#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List

from loguru import logger
import numpy as np
from PIL import Image
import torch


SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
SUPPORTED_METRICS = ("clipiqa", "musiq")


def find_images(dir_path: Path) -> List[Path]:
    return sorted(
        p for p in dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU.")
        return "cpu"

    return device_arg


def load_image_tensor(image_path: Path, device: str) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_np = np.asarray(image, dtype=np.float32) / 255.0

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return image_tensor.to(device=device, dtype=torch.float32)


def create_metrics(metric_names: List[str], device: str) -> Dict[str, torch.nn.Module]:
    try:
        import pyiqa
    except ImportError as exc:
        raise RuntimeError("pyiqa is required to compute CLIPIQA/MUSIQ.") from exc

    metrics: Dict[str, torch.nn.Module] = {}
    for metric_name in metric_names:
        try:
            metrics[metric_name] = pyiqa.create_metric(metric_name, device=device)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize metric '{metric_name}': {exc}") from exc
    return metrics


def summarize_scores(metric_name: str, scores: List[float]) -> None:
    if not scores:
        logger.warning(f"{metric_name.upper()}: no valid scores were produced.")
        return

    scores_np = np.asarray(scores, dtype=np.float32)
    valid_scores = scores_np[~np.isnan(scores_np)]
    if valid_scores.size == 0:
        logger.warning(f"{metric_name.upper()}: all evaluations failed.")
        return

    logger.info(
        f"{metric_name.upper():8s}: mean={valid_scores.mean():.6f}  "
        f"std={valid_scores.std():.6f}  count={valid_scores.size}"
    )

    failed = int(scores_np.size - valid_scores.size)
    if failed > 0:
        logger.warning(f"{metric_name.upper()}: {failed} images failed during evaluation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate no-reference IQA metrics for SR results.")
    parser.add_argument("--sr_dir", type=str, required=True, help="SR / inference results folder")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device, e.g. auto / cpu / cuda:0",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_METRICS,
        default=list(SUPPORTED_METRICS),
        help="Metrics to evaluate. Default: clipiqa musiq",
    )
    args = parser.parse_args()

    sr_dir = Path(args.sr_dir)
    if not sr_dir.exists():
        raise FileNotFoundError(f"sr_dir not found: {sr_dir}")

    sr_files = find_images(sr_dir)
    if not sr_files:
        raise FileNotFoundError(f"No supported images found in: {sr_dir}")

    device = resolve_device(args.device)
    logger.info(f"SR dir: {sr_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Metrics: {', '.join(args.metrics)}")
    logger.info(f"Found {len(sr_files)} images")

    metrics = create_metrics(args.metrics, device)
    results = {metric_name: [] for metric_name in args.metrics}
    evaluated_images = 0

    with torch.inference_mode():
        for image_path in sr_files:
            try:
                image_tensor = load_image_tensor(image_path, device)
            except Exception as exc:
                logger.warning(f"Skip {image_path}: image load failed ({exc})")
                continue

            evaluated_images += 1
            for metric_name, metric in metrics.items():
                try:
                    score = float(metric(image_tensor).reshape(-1).mean().item())
                except Exception as exc:
                    logger.warning(f"{metric_name.upper()} failed on {image_path.name}: {exc}")
                    score = np.nan
                results[metric_name].append(score)

    if evaluated_images == 0:
        raise RuntimeError("No images were successfully loaded for evaluation.")

    logger.info("==== RESULTS SUMMARY ====>")
    logger.info(f"Images evaluated: {evaluated_images}")
    for metric_name in args.metrics:
        summarize_scores(metric_name, results[metric_name])


if __name__ == "__main__":
    main()
