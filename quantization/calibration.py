from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import torch
import torch.nn.functional as F


def _load_image(path: Path, lq_size: int) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (lq_size, lq_size), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0) * 2.0 - 1.0


def _scan_images(path: str) -> List[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def build_calibration_batches(
    image_dir: str,
    diffusion,
    autoencoder,
    *,
    lq_size: int,
    sf: int,
    scale_factor: float,
    num_images: int,
    samples_per_image: int,
    device: torch.device,
    seed: int,
) -> List[Dict[str, torch.Tensor]]:
    rng = random.Random(seed)
    image_paths = _scan_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No calibration images found in: {image_dir}")
    image_paths = image_paths[:num_images]

    autoencoder = autoencoder.to(device).eval()
    batches: List[Dict[str, torch.Tensor]] = []
    candidate_timesteps = list(range(diffusion.num_timesteps))

    with torch.no_grad():
        for image_path in image_paths:
            lq = _load_image(image_path, lq_size).to(device)
            y_up = F.interpolate(lq, scale_factor=sf, mode="bicubic")
            z_y = autoencoder.encode(y_up) * scale_factor
            for _ in range(samples_per_image):
                timestep = rng.choice(candidate_timesteps)
                t = torch.tensor([timestep], device=device, dtype=torch.long)
                noise = torch.randn_like(z_y)
                x_t = z_y + float(diffusion.kappa * diffusion.sqrt_etas[timestep]) * noise
                x_in = diffusion._scale_input(x_t, t)
                batches.append(
                    {
                        "x": x_in.detach().cpu(),
                        "timesteps": t.detach().cpu(),
                        "lq": lq.detach().cpu(),
                    }
                )
    return batches
