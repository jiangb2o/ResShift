#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Calculate SR evaluation metrics with a unified entrypoint.
Supports full-reference metrics (PSNR, SSIM, LPIPS) and
no-reference metrics (CLIPIQA, MUSIQ).
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from utils import util_image


SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
SUPPORTED_METRIC_TOKENS = ("all", "psnr", "ssim", "lpips", "clipiqa", "musiq")
DEFAULT_METRICS = ("psnr", "ssim", "lpips", "clipiqa", "musiq")
FULL_REFERENCE_METRICS = ("psnr", "ssim", "lpips")
NO_REFERENCE_METRICS = ("clipiqa", "musiq")
DISPLAY_NAMES = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "lpips": "LPIPS",
    "clipiqa": "CLIPIQA",
    "musiq": "MUSIQ",
}


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    @property
    def encoding(self) -> str:
        return getattr(self.streams[0], "encoding", "utf-8")

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def tee_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def build_parser(
    default_metric_tokens: Optional[Sequence[str]] = None,
    description: str = "Calculate SR evaluation metrics.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gt_dir", type=str, default=None, help="Ground-truth image folder")
    parser.add_argument("--sr_dir", type=str, required=True, help="SR / inference results folder")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device for learned metrics, e.g. auto / cpu / cuda:0",
    )
    parser.add_argument("--ycbcr", action="store_true", help="Use Y channel for PSNR/SSIM")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_METRIC_TOKENS,
        default=list(default_metric_tokens or ("all",)),
        help="Metrics to evaluate. Use 'all' for the default full set.",
    )
    return parser


def find_images(dir_path: Path) -> List[Path]:
    return sorted(
        p for p in dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def expand_metric_tokens(metric_tokens: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for token in metric_tokens:
        current_metrics = DEFAULT_METRICS if token == "all" else (token,)
        for metric_name in current_metrics:
            if metric_name not in expanded:
                expanded.append(metric_name)
    return expanded


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA is not available, falling back to CPU.")
        return "cpu"

    return device_arg


def load_image_uint8(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def image_to_tensor01(image_uint8: np.ndarray, device: str) -> torch.Tensor:
    image_np = image_uint8.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return image_tensor.to(device=device, dtype=torch.float32)


def image_to_lpips_tensor(image_uint8: np.ndarray, device: str) -> torch.Tensor:
    return image_to_tensor01(image_uint8, device) * 2.0 - 1.0


def build_gt_indices(gt_dir: Path) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    rel_stem_map: Dict[str, List[Path]] = defaultdict(list)
    stem_map: Dict[str, List[Path]] = defaultdict(list)

    for image_path in find_images(gt_dir):
        rel_path = image_path.relative_to(gt_dir)
        rel_stem_key = f"{rel_path.parent.as_posix()}::{image_path.stem}"
        rel_stem_map[rel_stem_key].append(image_path)
        stem_map[image_path.stem].append(image_path)

    return rel_stem_map, stem_map


def find_gt_match(
    sr_path: Path,
    sr_dir: Path,
    rel_stem_map: Dict[str, List[Path]],
    stem_map: Dict[str, List[Path]],
) -> Optional[Path]:
    rel_path = sr_path.relative_to(sr_dir)
    rel_stem_key = f"{rel_path.parent.as_posix()}::{sr_path.stem}"
    rel_matches = rel_stem_map.get(rel_stem_key, [])
    if len(rel_matches) == 1:
        return rel_matches[0]
    if len(rel_matches) > 1:
        print(f"WARNING: Ambiguous GT matches for {sr_path.name} under relative path, skip FR metrics.")
        return None

    stem_matches = stem_map.get(sr_path.stem, [])
    if len(stem_matches) == 1:
        return stem_matches[0]
    if len(stem_matches) > 1:
        print(f"WARNING: Ambiguous GT matches for {sr_path.name} by stem, skip FR metrics.")
        return None
    return None


def create_lpips_metric(device: str) -> torch.nn.Module:
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("lpips is required to compute the LPIPS metric.") from exc
    return lpips.LPIPS(net="vgg").to(device).eval()


def create_nr_metrics(metric_names: Iterable[str], device: str) -> Dict[str, torch.nn.Module]:
    selected = [metric_name for metric_name in metric_names if metric_name in NO_REFERENCE_METRICS]
    if not selected:
        return {}

    try:
        import pyiqa
    except ImportError as exc:
        raise RuntimeError("pyiqa is required to compute CLIPIQA/MUSIQ.") from exc

    metrics: Dict[str, torch.nn.Module] = {}
    for metric_name in selected:
        try:
            metrics[metric_name] = pyiqa.create_metric(metric_name, device=device)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize metric '{metric_name}': {exc}") from exc
    return metrics


def summarize_metric(metric_name: str, scores: List[float], skipped: int) -> None:
    score_array = np.asarray(scores, dtype=np.float32)
    valid_scores = score_array[~np.isnan(score_array)]
    failed = int(score_array.size - valid_scores.size)

    if valid_scores.size == 0:
        mean = float("nan")
        std = float("nan")
    else:
        mean = float(valid_scores.mean())
        std = 0.0 if np.all(valid_scores == valid_scores[0]) else float(valid_scores.std())

    print(
        f"{DISPLAY_NAMES[metric_name]:<10s} "
        f"mean={mean:>10.4f}  std={std:>10.4f}  "
        f"count={valid_scores.size:>5d}  failed={failed:>5d}  skipped={skipped:>5d}"
    )


def main(
    argv: Optional[Sequence[str]] = None,
    default_metric_tokens: Optional[Sequence[str]] = None,
    description: str = "Calculate SR evaluation metrics.",
) -> int:
    parser = build_parser(default_metric_tokens=default_metric_tokens, description=description)
    args = parser.parse_args(argv)

    sr_dir = Path(args.sr_dir)
    log_path = sr_dir.parent / "metrics.log"
    with tee_output(log_path):
        print(f"Metrics log: {log_path.resolve()}")

        selected_metrics = expand_metric_tokens(args.metrics)
        needs_gt = any(metric_name in FULL_REFERENCE_METRICS for metric_name in selected_metrics)
        device = resolve_device(args.device)

        if not sr_dir.exists():
            raise FileNotFoundError(f"sr_dir not found: {sr_dir}")

        gt_dir = None
        if args.gt_dir is not None:
            gt_dir = Path(args.gt_dir)
            if not gt_dir.exists():
                raise FileNotFoundError(f"gt_dir not found: {gt_dir}")
        elif needs_gt:
            raise ValueError("--gt_dir is required when evaluating PSNR, SSIM, or LPIPS.")

        sr_files = find_images(sr_dir)
        if not sr_files:
            raise FileNotFoundError(f"No supported images found in: {sr_dir}")

        lpips_metric = create_lpips_metric(device) if "lpips" in selected_metrics else None
        nr_metrics = create_nr_metrics(selected_metrics, device)

        gt_rel_stem_map: Dict[str, List[Path]] = {}
        gt_stem_map: Dict[str, List[Path]] = {}
        if gt_dir is not None and needs_gt:
            gt_rel_stem_map, gt_stem_map = build_gt_indices(gt_dir)

        print(f"SR dir: {sr_dir}")
        print(f"GT dir: {gt_dir if gt_dir is not None else 'N/A'}")
        print(f"Device: {device}")
        print(f"Metrics: {', '.join(selected_metrics)}")
        print(f"Found {len(sr_files)} SR images")

        results = {metric_name: [] for metric_name in selected_metrics}
        skipped = {metric_name: 0 for metric_name in selected_metrics}
        loaded_images = 0
        paired_images = 0

        with torch.inference_mode():
            for sr_path in sr_files:
                try:
                    sr_uint8 = load_image_uint8(sr_path)
                except Exception as exc:
                    print(f"WARNING: Skip {sr_path}: image load failed ({exc})")
                    for metric_name in selected_metrics:
                        skipped[metric_name] += 1
                    continue

                loaded_images += 1

                sr_tensor_01 = None
                if nr_metrics:
                    sr_tensor_01 = image_to_tensor01(sr_uint8, device)
                    for metric_name, metric in nr_metrics.items():
                        try:
                            score = float(metric(sr_tensor_01).reshape(-1).mean().item())
                        except Exception as exc:
                            print(f"WARNING: {metric_name.upper()} failed on {sr_path.name}: {exc}")
                            score = np.nan
                        results[metric_name].append(score)

                if not needs_gt:
                    continue

                gt_path = find_gt_match(sr_path, sr_dir, gt_rel_stem_map, gt_stem_map)
                if gt_path is None:
                    print(f"WARNING: GT not found for {sr_path.name}, skip FR metrics.")
                    for metric_name in FULL_REFERENCE_METRICS:
                        if metric_name in results:
                            skipped[metric_name] += 1
                    continue

                try:
                    gt_uint8 = load_image_uint8(gt_path)
                except Exception as exc:
                    print(f"WARNING: Skip FR metrics for {sr_path.name}: GT load failed ({exc})")
                    for metric_name in FULL_REFERENCE_METRICS:
                        if metric_name in results:
                            skipped[metric_name] += 1
                    continue

                paired_images += 1
                sr_fr_uint8 = sr_uint8
                if sr_fr_uint8.shape != gt_uint8.shape:
                    print(f"WARNING: Shape mismatch for {sr_path.name}, resize SR to GT.")
                    sr_fr_uint8 = cv2.resize(
                        sr_fr_uint8,
                        (gt_uint8.shape[1], gt_uint8.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )

                if "psnr" in results:
                    try:
                        score = util_image.calculate_psnr(sr_fr_uint8, gt_uint8, border=0, ycbcr=args.ycbcr)
                    except Exception as exc:
                        print(f"WARNING: PSNR failed on {sr_path.name}: {exc}")
                        score = np.nan
                    results["psnr"].append(float(score))

                if "ssim" in results:
                    try:
                        score = util_image.calculate_ssim(sr_fr_uint8, gt_uint8, border=0, ycbcr=args.ycbcr)
                    except Exception as exc:
                        print(f"WARNING: SSIM failed on {sr_path.name}: {exc}")
                        score = np.nan
                    results["ssim"].append(float(score))

                if "lpips" in results and lpips_metric is not None:
                    try:
                        sr_lpips_tensor = image_to_lpips_tensor(sr_fr_uint8, device)
                        gt_lpips_tensor = image_to_lpips_tensor(gt_uint8, device)
                        score = float(lpips_metric(gt_lpips_tensor, sr_lpips_tensor).reshape(-1).mean().item())
                    except Exception as exc:
                        print(f"WARNING: LPIPS failed on {sr_path.name}: {exc}")
                        score = np.nan
                    results["lpips"].append(score)

        if loaded_images == 0:
            raise RuntimeError("No images were successfully loaded for evaluation.")

        print("==== RESULTS SUMMARY ====>")
        print(f"Images found   : {len(sr_files)}")
        print(f"Images loaded  : {loaded_images}")
        if needs_gt:
            print(f"GT pairs used  : {paired_images}")
        print("Metric     mean             std             count        failed        skipped")
        for metric_name in selected_metrics:
            summarize_metric(metric_name, results[metric_name], skipped[metric_name])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
