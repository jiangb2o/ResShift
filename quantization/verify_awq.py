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

from quantization.awq import load_awq_quantized_model
from utils import util_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and verify an AWQ-quantized ResShift UNet.")
    parser.add_argument("--config", type=str, default="configs/realsr_swinunet_realesrgan256.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = OmegaConf.load((PROJECT_ROOT / args.config).resolve())
    model = util_common.instantiate_from_config(cfg.model)
    model, quantized_layers = load_awq_quantized_model(model, str((PROJECT_ROOT / args.checkpoint).resolve()), device)

    with torch.no_grad():
        x = torch.randn(1, 3, 64, 64, device=device)
        lq = torch.randn(1, 3, 64, 64, device=device)
        t = torch.tensor([10], dtype=torch.long, device=device)
        out = model(x=x, timesteps=t, lq=lq, mask=None)
    print(f"loaded_quantized_layers {len(quantized_layers)}")
    print(f"forward_ok {tuple(out.shape)} {out.dtype} {torch.isfinite(out).all().item()}")


if __name__ == "__main__":
    main()
