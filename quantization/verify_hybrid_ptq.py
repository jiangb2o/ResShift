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

from quantization.hybrid_ptq import load_hybrid_quantized_model
from quantization.int8_ptq import INT8Conv2d, INT8Linear
from quantization.awq import AWQLinear
from utils import util_common, util_net


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and verify a hybrid AWQ + INT8 PTQ ResShift UNet.")
    parser.add_argument("--config", type=str, default="configs/realsr_swinunet_realesrgan256.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--reference_checkpoint", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = OmegaConf.load((PROJECT_ROOT / args.config).resolve())

    model = util_common.instantiate_from_config(cfg.model)
    model, awq_layers, int8_layers = load_hybrid_quantized_model(
        model,
        str((PROJECT_ROOT / args.checkpoint).resolve()),
        device,
    )

    ref_out = None
    if args.reference_checkpoint:
        ref_model = util_common.instantiate_from_config(cfg.model).to(device).eval()
        ref_ckpt = torch.load((PROJECT_ROOT / args.reference_checkpoint).resolve(), map_location=device)
        ref_state = ref_ckpt["state_dict"] if "state_dict" in ref_ckpt else ref_ckpt
        util_net.reload_model(ref_model, ref_state)
    else:
        ref_model = None

    with torch.no_grad():
        x = torch.randn(1, 3, 64, 64, device=device)
        lq = torch.randn(1, 3, 64, 64, device=device)
        t = torch.tensor([10], dtype=torch.long, device=device)
        out = model(x=x, timesteps=t, lq=lq, mask=None)
        if ref_model is not None:
            ref_out = ref_model(x=x, timesteps=t, lq=lq, mask=None)

    num_awq_modules = sum(1 for m in model.modules() if isinstance(m, AWQLinear))
    num_int8_linear = sum(1 for m in model.modules() if isinstance(m, INT8Linear))
    num_int8_conv = sum(1 for m in model.modules() if isinstance(m, INT8Conv2d))
    print(f"loaded_awq_layers {len(awq_layers)}")
    print(f"loaded_int8_layers {len(int8_layers)}")
    print(f"module_counts awq={num_awq_modules} int8_linear={num_int8_linear} int8_conv={num_int8_conv}")
    print(f"forward_ok {tuple(out.shape)} {out.dtype} {torch.isfinite(out).all().item()}")
    if ref_out is not None:
        diff = (out - ref_out).abs()
        print(f"max_abs_diff {diff.max().item()}")
        print(f"mean_abs_diff {diff.mean().item()}")


if __name__ == "__main__":
    main()
