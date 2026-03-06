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
from awq import AWQConfig, collect_awq_activations, quantize_model_awq, save_awq_checkpoint
from calibration import build_calibration_batches
from utils import util_common, util_net


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize ResShift UNet with AWQ.")
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
    parser.add_argument("--include", type=str, nargs="*", default=["qkv", "proj", "reduction"])
    parser.add_argument("--exclude", type=str, nargs="*", default=[])
    parser.add_argument("--use_linfusion", action="store_true")
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
        include_patterns=tuple(args.include),
        exclude_patterns=tuple(args.exclude),
    )
    activation_map = collect_awq_activations(model, calibration_batches, awq_config, device)
    print(f"collected_layers {len(activation_map)}")

    quantized_model, metadata = quantize_model_awq(model, activation_map, awq_config)
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_awq_checkpoint(str(output_path), quantized_model, metadata, awq_config, args.config)
    print(f"quantized_layers {len(metadata)}")
    print(f"saved_to {output_path}")


if __name__ == "__main__":
    main()
