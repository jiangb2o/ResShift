#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gaussian_diffusion import ModelMeanType
from models.script_util import create_gaussian_diffusion

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    import onnx
    from onnx.reference import ReferenceEvaluator


class OnnxRunner:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.backend = "onnxruntime" if ort is not None else "onnx-reference"
        if ort is not None:
            self.session = ort.InferenceSession(
                str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.input_names = [x.name for x in self.session.get_inputs()]
            self.output_name = self.session.get_outputs()[0].name
        else:
            model = onnx.load(str(model_path))
            self.session = ReferenceEvaluator(model)
            self.input_names = [x.name for x in model.graph.input]
            self.output_name = model.graph.output[0].name

    def run(self, feeds: Dict[str, np.ndarray]) -> np.ndarray:
        if ort is not None:
            outputs = self.session.run([self.output_name], feeds)
            return outputs[0]
        outputs = self.session.run([self.output_name], feeds)
        return outputs[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ResShift ONNX inference pipeline.")
    parser.add_argument("--config", type=str, default="configs/realsr_swinunet_realesrgan256.yaml")
    parser.add_argument("--unet_onnx", type=str, default="weights/resshift_model.onnx")
    parser.add_argument(
        "--encoder_onnx", type=str, default="onnx_inference/models/autoencoder_encoder.onnx"
    )
    parser.add_argument(
        "--decoder_onnx", type=str, default="onnx_inference/models/autoencoder_decoder.onnx"
    )
    parser.add_argument("--input", type=str, required=True, help="Input LQ image path.")
    parser.add_argument("--output", type=str, required=True, help="Output SR image path.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--allow_reference",
        action="store_true",
        help="Allow onnx.reference backend when onnxruntime is unavailable (very slow).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Only run first N reverse steps for quick validation. Default: full steps.",
    )
    return parser.parse_args()


def preprocess_image(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


def postprocess_image(tensor: torch.Tensor) -> np.ndarray:
    image = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0.0, 1.0)
    image = (image * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _pad_to_multiple(y: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int], bool]:
    h, w = y.shape[2], y.shape[3]
    if h % multiple == 0 and w % multiple == 0:
        return y, (0, 0), False
    pad_h = int(np.ceil(h / multiple) * multiple - h)
    pad_w = int(np.ceil(w / multiple) * multiple - w)
    pad_mode = "reflect"
    if pad_h >= h or pad_w >= w:
        pad_mode = "replicate"
    y = F.pad(y, (0, pad_w, 0, pad_h), mode=pad_mode)
    return y, (pad_h, pad_w), True


def _scale_input(x: torch.Tensor, t_index: int, diffusion) -> torch.Tensor:
    # 对应 models/gaussian_diffusion.py 中的 GaussianDiffusion._scale_input。
    # ResShift 会在送入 UNet 前对 x_t 做时刻相关归一化，提高数值稳定性。
    if not diffusion.normalize_input:
        return x
    if diffusion.latent_flag:
        std = float(np.sqrt(diffusion.etas[t_index] * diffusion.kappa**2 + 1.0))
        return x / std
    inputs_max = float(np.sqrt(diffusion.etas[t_index]) * diffusion.kappa * 3.0 + 1.0)
    return x / inputs_max


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if ort is None and not args.dry_run and not args.allow_reference:
        raise RuntimeError(
            "onnxruntime is not installed. "
            "Install onnxruntime to run inference, or use --dry_run for interface validation."
        )

    cfg = OmegaConf.load((PROJECT_ROOT / args.config).resolve())
    # 构建与原生 ResShift 推理一致的扩散调度器对象。
    diffusion = create_gaussian_diffusion(**cfg.diffusion.params)
    # SpacedDiffusion 场景下，循环步索引可能需要映射到原训练时间步 id。
    timestep_map = getattr(diffusion, "timestep_map", None)

    y0 = preprocess_image((PROJECT_ROOT / args.input).resolve())
    y0, (pad_h, pad_w), padded = _pad_to_multiple(y0, int(cfg.model.params.get("lq_size", 64)))
    sf = int(cfg.diffusion.params.get("sf", 4))
    y_up = F.interpolate(y0, scale_factor=sf, mode="bicubic")

    unet = OnnxRunner((PROJECT_ROOT / args.unet_onnx).resolve())
    encoder = OnnxRunner((PROJECT_ROOT / args.encoder_onnx).resolve())
    decoder = OnnxRunner((PROJECT_ROOT / args.decoder_onnx).resolve())

    print(f"Backend: UNet={unet.backend}, Encoder={encoder.backend}, Decoder={decoder.backend}")
    print(f"LQ shape: {tuple(y0.shape)}, upsampled shape: {tuple(y_up.shape)}")
    print(
        "Model IO names:\n"
        f" unet_in={unet.input_names}, unet_out={unet.output_name};\n"
        f" enc_in={encoder.input_names}, enc_out={encoder.output_name};\n"
        f" dec_in={decoder.input_names}, dec_out={decoder.output_name}"
    )

    begin_time = time.perf_counter()

    z_y = encoder.run({"image": y_up.numpy().astype(np.float32)})
    z_y = torch.from_numpy(z_y).float()
    scale_factor = float(cfg.diffusion.params.get("scale_factor", 1.0))
    # 与 encode_first_stage 对齐：编码后会对 latent 乘以 scale_factor。
    z_y = z_y * scale_factor

    # q(x_T | y)：从最后一步的先验分布初始化反向过程起点。
    t_last = diffusion.num_timesteps - 1
    init_noise = torch.randn_like(z_y)
    z_t = z_y + float(diffusion.kappa * diffusion.sqrt_etas[t_last]) * init_noise

    total_steps = diffusion.num_timesteps
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    indices = list(range(diffusion.num_timesteps))[::-1][:total_steps]
    # print(f"Sampling reverse steps: {len(indices)} (of total {diffusion.num_timesteps})")

    model_mean_type = diffusion.model_mean_type
    for i in indices:
        # `i`：反向采样循环索引。
        # `model_t`：实际喂给 UNet 的时间步（SpacedDiffusion 时会做映射）。
        model_t = timestep_map[i] if timestep_map is not None else i
        t_in = np.full((z_t.shape[0],), model_t, dtype=np.int64)

        # UNet 输入 x 是当前 latent 状态 x_t，并做时刻相关归一化。
        x_in = _scale_input(z_t, i, diffusion).numpy().astype(np.float32)
        # UNet 条件输入 lq 是原始低质量图（等价于 sampler 里的 model_kwargs["lq"]）。
        lq_in = y0.numpy().astype(np.float32)

        model_out = unet.run({"x": x_in, "lq": lq_in, "timesteps": t_in})
        model_out = torch.from_numpy(model_out).float()

        # `model_mean_type` 定义了 ResShift 训练时 UNet 的预测目标：
        # START_X：直接预测 x_0
        # RESIDUAL：预测 (y - x_0)
        # EPSILON / EPSILON_SCALE：预测噪声形式，再解析还原 x_0
        if model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_out
        elif model_mean_type == ModelMeanType.RESIDUAL:
            pred_xstart = z_y - model_out
        elif model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = (
                z_t
                - float(diffusion.sqrt_etas[i] * diffusion.kappa) * model_out
                - float(diffusion.etas[i]) * z_y
            ) / float(1.0 - diffusion.etas[i])
        elif model_mean_type == ModelMeanType.EPSILON_SCALE:
            pred_xstart = (z_t - model_out - float(diffusion.etas[i]) * z_y) / float(
                1.0 - diffusion.etas[i]
            )
        else:
            raise NotImplementedError(f"Unsupported model_mean_type: {model_mean_type}")

        # `coef1/coef2` 是 q(x_{t-1} | x_t, x_0) 的后验均值系数：
        # mean = coef1 * x_t + coef2 * x_0_pred
        coef1 = float(diffusion.posterior_mean_coef1[i])
        coef2 = float(diffusion.posterior_mean_coef2[i])
        mean = coef1 * z_t + coef2 * pred_xstart

        if i != 0:
            noise = torch.randn_like(z_t)
            # 第 i 步随机反向采样使用的后验 sigma。
            sigma = float(np.exp(0.5 * diffusion.posterior_log_variance_clipped[i]))
            z_t = mean + sigma * noise
        else:
            # 最后一步确定性更新（不再注入额外噪声）。
            z_t = mean

    # 与 decode_first_stage 对齐：进入 decoder 前先除以 scale_factor。
    z_out = z_t / max(scale_factor, 1e-8)
    sr = decoder.run({"latent": z_out.numpy().astype(np.float32)})
    sr_tensor = torch.from_numpy(sr).float().clamp(-1.0, 1.0)

    if padded:
        h0 = y0.shape[2] - pad_h
        w0 = y0.shape[3] - pad_w
        sr_tensor = sr_tensor[:, :, : h0 * sf, : w0 * sf]

    elapsed_time = time.perf_counter() - begin_time
    print(f"Total inference time: {elapsed_time:.4f} s")
    output_path = (PROJECT_ROOT / args.output).resolve()
    cv2.imwrite(str(output_path), postprocess_image(sr_tensor))
    print(f"Saved SR result to: {output_path}")


if __name__ == "__main__":
    main()
