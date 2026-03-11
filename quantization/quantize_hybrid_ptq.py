#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.script_util import create_gaussian_diffusion
from quantization.awq import AWQConfig, collect_awq_activations
from quantization.calibration import build_calibration_batches
from quantization.hybrid_ptq import quantize_model_hybrid, save_hybrid_checkpoint
from quantization.int8_ptq import INT8PTQConfig
from utils import util_common, util_net
from utils.util_opts import str2bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize ResShift UNet with AWQ + INT8 PTQ.")
    parser.add_argument("--config", type=str, default="configs/realsr_swinunet_realesrgan256.yaml")
    parser.add_argument("--checkpoint", type=str, default="weights/resshift_realsrx4_s15_v1_default.pth")
    parser.add_argument("--calibration_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--samples_per_image", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--alpha_samples", type=int, default=11)
    parser.add_argument("--max_tokens_per_layer", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--awq_include", type=str, nargs="*", default=["qkv", "proj", "reduction"])
    parser.add_argument("--awq_exclude", type=str, nargs="*", default=[])
    parser.add_argument("--int8_include", type=str, nargs="*", default=[])
    parser.add_argument("--int8_exclude", type=str, nargs="*", default=[])
    parser.add_argument("--quantize_conv", type=str2bool, default="True")
    parser.add_argument("--quantize_linear", type=str2bool, default="True")
    parser.add_argument("--use_linfusion", type=str2bool, default="False")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = OmegaConf.load((PROJECT_ROOT / args.config).resolve())
    cfg.model.params.use_linfusion = bool(args.use_linfusion)

    if device.type != "cuda":
        from ldm.modules.diffusionmodules import model as diffusion_model
        diffusion_model.XFORMERS_IS_AVAILBLE = False

    model = util_common.instantiate_from_config(cfg.model).to(device).eval()
    ckpt = torch.load((PROJECT_ROOT / args.checkpoint).resolve(), map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    util_net.reload_model(model, state_dict)

    autoencoder = util_common.instantiate_from_config(cfg.autoencoder).to(device).eval()
    ae_ckpt = torch.load((PROJECT_ROOT / cfg.autoencoder.ckpt_path).resolve(), map_location=device)
    ae_state = ae_ckpt["state_dict"] if "state_dict" in ae_ckpt else ae_ckpt
    util_net.reload_model(autoencoder, ae_state)

    diffusion = create_gaussian_diffusion(**cfg.diffusion.params)
    calibration_batches = build_calibration_batches(
        image_dir=str((PROJECT_ROOT / args.calibration_dir).resolve()),
        diffusion=diffusion,
        autoencoder=autoencoder,
        lq_size=int(cfg.model.params.get("lq_size", 64)),
        sf=int(cfg.diffusion.params.get("sf", 4)),
        scale_factor=float(cfg.diffusion.params.get("scale_factor", 1.0)),
        num_images=args.num_images,
        samples_per_image=args.samples_per_image,
        device=device,
        seed=args.seed,
    )
    print(f"calibration_batches {len(calibration_batches)}")

    awq_config = AWQConfig(
        n_bits=args.w_bits,
        group_size=args.group_size,
        alpha_samples=args.alpha_samples,
        max_tokens_per_layer=args.max_tokens_per_layer,
        include_patterns=tuple(args.awq_include),
        exclude_patterns=tuple(args.awq_exclude),
    )
    int8_config = INT8PTQConfig(
        n_bits=8,
        quantize_conv=bool(args.quantize_conv),
        quantize_linear=bool(args.quantize_linear),
        include_patterns=tuple(args.int8_include),
        exclude_patterns=tuple(args.int8_exclude),
    )

    activation_map = collect_awq_activations(model, calibration_batches, awq_config, device)
    print(f"collected_awq_layers {len(activation_map)}")

    quantized_model, awq_metadata, int8_metadata = quantize_model_hybrid(
        model,
        activation_map,
        awq_config,
        int8_config,
    )
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_hybrid_checkpoint(
        str(output_path),
        quantized_model,
        awq_metadata,
        int8_metadata,
        awq_config,
        int8_config,
        args.config,
    )
    print(f"quantized_awq_layers {len(awq_metadata)}")
    print(f"quantized_int8_layers {len(int8_metadata)}")
    print(f"saved_to {output_path}")


if __name__ == "__main__":
    main()
