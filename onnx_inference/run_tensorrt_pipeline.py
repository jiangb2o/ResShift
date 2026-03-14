#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
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

from onnx_inference.tensorrt_runner import TensorRTRunner
from models.gaussian_diffusion import ModelMeanType
from models.script_util import create_gaussian_diffusion


class _Log:
    def __init__(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("tensorrt_inference")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)
        self.print(f"\n\n====================Log initialized. Logging to: {file_path}====================")

    def print(self, message: str) -> None:
        self._logger.info(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ResShift inference with native TensorRT engines.")
    parser.add_argument("--config", type=str, default="configs/realsr_swinunet_realesrgan256.yaml")
    parser.add_argument(
        "--encoder_engine",
        type=str,
        default="onnx_inference/engines/autoencoder_encoder.engine",
    )
    parser.add_argument(
        "--unet_engine",
        type=str,
        default="onnx_inference/engines/resshift_model.engine",
    )
    parser.add_argument(
        "--decoder_engine",
        type=str,
        default="onnx_inference/engines/autoencoder_decoder.engine",
    )
    parser.add_argument("--input", type=str, required=True, help="Input image path or folder.")
    parser.add_argument("--output", type=str, required=True, help="Output image path or folder.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--warmup_images",
        type=int,
        default=3,
        help="Number of synthetic warmup runs excluded from timing stats.",
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


def pad_to_multiple(image: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int], bool]:
    h, w = image.shape[2], image.shape[3]
    if h % multiple == 0 and w % multiple == 0:
        return image, (0, 0), False
    pad_h = int(np.ceil(h / multiple) * multiple - h)
    pad_w = int(np.ceil(w / multiple) * multiple - w)
    pad_mode = "reflect"
    if pad_h >= h or pad_w >= w:
        pad_mode = "replicate"
    image = F.pad(image, (0, pad_w, 0, pad_h), mode=pad_mode)
    return image, (pad_h, pad_w), True


def scale_input(x: torch.Tensor, t_index: int, diffusion) -> torch.Tensor:
    if not diffusion.normalize_input:
        return x
    if diffusion.latent_flag:
        std = float(np.sqrt(diffusion.etas[t_index] * diffusion.kappa**2 + 1.0))
        return x / std
    inputs_max = float(np.sqrt(diffusion.etas[t_index]) * diffusion.kappa * 3.0 + 1.0)
    return x / inputs_max


def _run_pipeline_from_lq(
    y0_raw: torch.Tensor,
    image_label: str,
    encoder: TensorRTRunner,
    unet: TensorRTRunner,
    decoder: TensorRTRunner,
    diffusion,
    timestep_map,
    sf: int,
    scale_factor: float,
    lq_size: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    y0, (pad_h, pad_w), padded = pad_to_multiple(y0_raw, lq_size)
    y_up = F.interpolate(y0, scale_factor=sf, mode="bicubic")

    unet_shape = unet.get_tensor_shapes()["x"]
    expected_h, expected_w = int(unet_shape[2]), int(unet_shape[3])
    if (y0.shape[2], y0.shape[3]) != (expected_h, expected_w):
        raise RuntimeError(
            f"Current UNet engine expects latent/LQ shape {(expected_h, expected_w)}, "
            f"but padded input for `{image_label}` is {(int(y0.shape[2]), int(y0.shape[3]))}. "
            "Rebuild the UNet engine with matching profile, or use an input that pads to this size."
        )

    encoder_begin = time.perf_counter()
    z_y = encoder.run({"image": y_up.numpy().astype(np.float32)})
    encoder_elapsed = time.perf_counter() - encoder_begin
    z_y = torch.from_numpy(z_y).float() * scale_factor

    t_last = diffusion.num_timesteps - 1
    z_t = z_y + float(diffusion.kappa * diffusion.sqrt_etas[t_last]) * torch.randn_like(z_y)
    model_mean_type = diffusion.model_mean_type

    unet_elapsed = 0.0
    indices = list(range(diffusion.num_timesteps))[::-1]
    for i in indices:
        model_t = timestep_map[i] if timestep_map is not None else i
        t_in = np.full((z_t.shape[0],), model_t, dtype=np.int64)
        x_in = scale_input(z_t, i, diffusion).numpy().astype(np.float32)
        lq_in = y0.numpy().astype(np.float32)

        unet_begin = time.perf_counter()
        model_out = torch.from_numpy(unet.run({"x": x_in, "lq": lq_in, "timesteps": t_in})).float()
        unet_elapsed += time.perf_counter() - unet_begin

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

        mean = (
            float(diffusion.posterior_mean_coef1[i]) * z_t
            + float(diffusion.posterior_mean_coef2[i]) * pred_xstart
        )
        if i != 0:
            sigma = float(np.exp(0.5 * diffusion.posterior_log_variance_clipped[i]))
            z_t = mean + sigma * torch.randn_like(z_t)
        else:
            z_t = mean

    decoder_begin = time.perf_counter()
    sr = decoder.run({"latent": (z_t / max(scale_factor, 1e-8)).numpy().astype(np.float32)})
    decoder_elapsed = time.perf_counter() - decoder_begin
    sr_tensor = torch.from_numpy(sr).float().clamp(-1.0, 1.0)

    if padded:
        h0 = int(y0.shape[2]) - pad_h
        w0 = int(y0.shape[3]) - pad_w
        sr_tensor = sr_tensor[:, :, : h0 * sf, : w0 * sf]

    profile = {
        "encoder": encoder_elapsed,
        "unet": unet_elapsed,
        "decoder": decoder_elapsed,
        "unet_steps": len(indices),
    }
    return sr_tensor, profile


def main() -> None:
    args = parse_args()
    log = _Log((PROJECT_ROOT / "onnx_inference/log/tensorrt_log.log").resolve())
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(str((PROJECT_ROOT / args.config).resolve()))
    diffusion = create_gaussian_diffusion(**cfg.diffusion.params)
    timestep_map = getattr(diffusion, "timestep_map", None)
    sf = int(cfg.diffusion.params.get("sf", 4))
    scale_factor = float(cfg.diffusion.params.get("scale_factor", 1.0))
    lq_size = int(cfg.model.params.get("lq_size", 64))

    encoder = TensorRTRunner((PROJECT_ROOT / args.encoder_engine).resolve(), device=args.device)
    unet = TensorRTRunner((PROJECT_ROOT / args.unet_engine).resolve(), device=args.device)
    decoder = TensorRTRunner((PROJECT_ROOT / args.decoder_engine).resolve(), device=args.device)

    log.print(f"Device: {args.device}")
    log.print(f"Encoder engine: {encoder.engine_path}")
    log.print(f"UNet engine: {unet.engine_path}")
    log.print(f"Decoder engine: {decoder.engine_path}")
    log.print(
        "Engine tensor shapes:\n"
        f" encoder={encoder.get_tensor_shapes()}\n"
        f" unet={unet.get_tensor_shapes()}\n"
        f" decoder={decoder.get_tensor_shapes()}"
    )

    input_path = (PROJECT_ROOT / args.input).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()

    if input_path.is_dir():
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_paths = sorted(
            [path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in valid_exts]
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in folder: {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        image_paths = [input_path]
        output_path.parent.mkdir(parents=True, exist_ok=True)

    total_elapsed = 0.0
    total_encoder_elapsed = 0.0
    total_unet_elapsed = 0.0
    total_decoder_elapsed = 0.0
    total_unet_steps = 0
    measured_images = 0
    total_images = len(image_paths)
    warmup_images = max(0, int(args.warmup_images))

    if warmup_images > 0:
        first_shape = preprocess_image(image_paths[0]).shape
        warmup_lq = torch.randn(first_shape, dtype=torch.float32)
        log.print(f"Warmup enabled: {warmup_images} run(s) with synthetic input shape={tuple(first_shape)}")
        for i in range(warmup_images):
            _run_pipeline_from_lq(
                y0_raw=warmup_lq,
                image_label=f"warmup_{i+1}",
                encoder=encoder,
                unet=unet,
                decoder=decoder,
                diffusion=diffusion,
                timestep_map=timestep_map,
                sf=sf,
                scale_factor=scale_factor,
                lq_size=lq_size,
            )
            log.print(f"[warmup {i + 1}/{warmup_images}] done")

    for idx, image_path in enumerate(image_paths, start=1):
        y0_raw = preprocess_image(image_path)

        if idx == 1:
            y0_show, _, _ = pad_to_multiple(y0_raw, lq_size)
            y_up_show = F.interpolate(y0_show, scale_factor=sf, mode="bicubic")
            log.print(f"LQ shape: {tuple(y0_show.shape)}, upsampled shape: {tuple(y_up_show.shape)}")

        begin_time = time.perf_counter()
        sr_tensor, profile = _run_pipeline_from_lq(
            y0_raw=y0_raw,
            image_label=image_path.name,
            encoder=encoder,
            unet=unet,
            decoder=decoder,
            diffusion=diffusion,
            timestep_map=timestep_map,
            sf=sf,
            scale_factor=scale_factor,
            lq_size=lq_size,
        )
        elapsed_time = time.perf_counter() - begin_time

        if input_path.is_dir():
            current_output = output_path / image_path.name
        else:
            current_output = output_path
        current_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(current_output), postprocess_image(sr_tensor))

        total_elapsed += elapsed_time
        total_encoder_elapsed += profile["encoder"]
        total_unet_elapsed += profile["unet"]
        total_decoder_elapsed += profile["decoder"]
        total_unet_steps += int(profile["unet_steps"])
        measured_images += 1

        if(idx % (total_images / 10) == 0):
            log.print(
                f"[{idx}/{total_images}] {image_path.name} -> {current_output.name}, "
                f"total={elapsed_time:.4f} s, encoder={profile['encoder']:.4f} s, "
                f"unet={profile['unet']:.4f} s, decoder={profile['decoder']:.4f} s, "
                f"unet_steps={int(profile['unet_steps'])}"
            )

    avg_time = total_elapsed / max(measured_images, 1)
    avg_encoder_time = total_encoder_elapsed / max(measured_images, 1)
    avg_unet_time = total_unet_elapsed / max(measured_images, 1)
    avg_decoder_time = total_decoder_elapsed / max(measured_images, 1)
    avg_unet_step_time = total_unet_elapsed / max(total_unet_steps, 1)
    log.print(f"Processed {total_images} image(s)")
    log.print(f"Measured images (excluding warmup): {measured_images}")
    log.print(f"Total measured inference time: {total_elapsed:.4f} s")
    log.print(f"Average inference time per image: {avg_time:.4f} s")
    log.print(f"Average encoder time per image: {avg_encoder_time:.4f} s")
    log.print(f"Average UNet time per image: {avg_unet_time:.4f} s")
    log.print(f"Average decoder time per image: {avg_decoder_time:.4f} s")
    log.print(f"Average UNet time per step: {avg_unet_step_time:.6f} s")
    log.print(f"Saved SR result(s) to: {output_path}")


if __name__ == "__main__":
    main()
