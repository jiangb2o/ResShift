#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import util_common, util_net
from ldm.modules.diffusionmodules import model as diffusion_model


class EncoderExportWrapper(nn.Module):
    def __init__(self, autoencoder: nn.Module):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)


class DecoderExportWrapper(nn.Module):
    def __init__(self, autoencoder: nn.Module):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ResShift autoencoder encoder/decoder to ONNX."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/realsr_swinunet_realesrgan256.yaml",
        help="Path to config yaml containing autoencoder and diffusion.",
    )
    parser.add_argument(
        "--encoder_output",
        type=str,
        default="weights/autoencoder_encoder.onnx",
        help="Output path for encoder ONNX.",
    )
    parser.add_argument(
        "--decoder_output",
        type=str,
        default="weights/autoencoder_decoder.onnx",
        help="Output path for decoder ONNX.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used for export.",
    )
    parser.add_argument(
        "--lq_shape",
        type=int,
        nargs=4,
        default=None,
        metavar=("B", "C", "H", "W"),
        help="LQ tensor shape [B,C,H,W]. Default uses config export input_shape.",
    )
    return parser.parse_args()


def resolve_path(path_str: str, base: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    from_project = (PROJECT_ROOT / path).resolve()
    if from_project.exists():
        return from_project
    return (base / path).resolve()


def main() -> None:
    args = parse_args()
    # Exporting on CPU cannot run xformers memory_efficient_attention.
    diffusion_model.XFORMERS_IS_AVAILBLE = False
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = OmegaConf.load(str(cfg_path))

    if "autoencoder" not in cfg:
        raise KeyError(
            f"Config {cfg_path} does not contain `autoencoder`. "
            "Please use a training/inference config such as configs/realsr_swinunet_realesrgan256.yaml."
        )
    ae = util_common.instantiate_from_config(cfg.autoencoder)

    ckpt_path = resolve_path(cfg.autoencoder.ckpt_path, cfg_path.parent)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=args.device)
    if "state_dict" in ckpt:
        util_net.reload_model(ae, ckpt["state_dict"])
    else:
        util_net.reload_model(ae, ckpt)

    ae.eval()
    if args.device == "cuda":
        ae = ae.cuda()

    if args.lq_shape is not None:
        b, c, h, w = args.lq_shape
    else:
        lq_size = int(cfg.model.params.get("lq_size", 64))
        b, c, h, w = 1, 3, lq_size, lq_size

    sf = int(cfg.diffusion.params.get("sf", 4)) if "diffusion" in cfg else 4
    scale_factor = (
        float(cfg.diffusion.params.get("scale_factor", 1.0)) if "diffusion" in cfg else 1.0
    )

    enc_input = torch.randn(b, c, h * sf, w * sf, dtype=torch.float32)
    dec_input = torch.randn(b, c, h, w, dtype=torch.float32) / max(scale_factor, 1e-8)
    if args.device == "cuda":
        enc_input = enc_input.cuda()
        dec_input = dec_input.cuda()

    encoder_output = resolve_path(args.encoder_output, PROJECT_ROOT)
    decoder_output = resolve_path(args.decoder_output, PROJECT_ROOT)
    encoder_output.parent.mkdir(parents=True, exist_ok=True)
    decoder_output.parent.mkdir(parents=True, exist_ok=True)

    encoder_wrapper = EncoderExportWrapper(ae).eval()
    decoder_wrapper = DecoderExportWrapper(ae).eval()
    if args.device == "cuda":
        encoder_wrapper = encoder_wrapper.cuda()
        decoder_wrapper = decoder_wrapper.cuda()

    opset_version = int(cfg.export.opset_version) if "export" in cfg else 17
    do_constant_folding = bool(cfg.export.do_constant_folding) if "export" in cfg else True

    with torch.no_grad():
        torch.onnx.export(
            encoder_wrapper,
            (enc_input,),
            str(encoder_output),
            opset_version=opset_version,
            input_names=["image"],
            output_names=["latent"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "latent": {0: "batch", 2: "height", 3: "width"},
            },
            do_constant_folding=do_constant_folding,
            export_params=True,
        )

        torch.onnx.export(
            decoder_wrapper,
            (dec_input,),
            str(decoder_output),
            opset_version=opset_version,
            input_names=["latent"],
            output_names=["image"],
            dynamic_axes={
                "latent": {0: "batch", 2: "height", 3: "width"},
                "image": {0: "batch", 2: "height", 3: "width"},
            },
            do_constant_folding=do_constant_folding,
            export_params=True,
        )

    print("Export finished:")
    print(f"  Encoder: {encoder_output}")
    print(f"  Decoder: {decoder_output}")


if __name__ == "__main__":
    main()
