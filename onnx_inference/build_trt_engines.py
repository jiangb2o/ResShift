#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import tensorrt as trt
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _shape_str(shape: Tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build native TensorRT engines from ResShift ONNX models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/realsr_swinunet_realesrgan256.yaml",
        help="ResShift config used to derive default latent/image shapes.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["encoder", "unet", "decoder", "all"],
        default=["all"],
        help="Which engines to build.",
    )
    parser.add_argument(
        "--encoder_onnx",
        type=str,
        default="onnx_inference/models/autoencoder_encoder.onnx",
    )
    parser.add_argument(
        "--unet_onnx",
        type=str,
        default="onnx_inference/models/resshift_model.onnx",
    )
    parser.add_argument(
        "--decoder_onnx",
        type=str,
        default="onnx_inference/models/autoencoder_decoder.onnx",
    )
    parser.add_argument(
        "--engine_dir",
        type=str,
        default="onnx_inference/engines",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--latent_min_hw",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
    )
    parser.add_argument(
        "--latent_opt_hw",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
    )
    parser.add_argument(
        "--latent_max_hw",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
    )
    parser.add_argument(
        "--workspace_gb",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--optimization_level",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable TensorRT FP16 builder flag when supported.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
    )
    return parser.parse_args()


def _collect_model_names(raw_models) -> Tuple[str, ...]:
    if "all" in raw_models:
        return ("encoder", "unet", "decoder")
    return tuple(raw_models)


def _default_latent_shapes(cfg, args: argparse.Namespace):
    base = int(cfg.model.params.get("lq_size", 64))
    min_hw = tuple(args.latent_min_hw) if args.latent_min_hw is not None else (base, base)
    opt_hw = tuple(args.latent_opt_hw) if args.latent_opt_hw is not None else (base * 2, base * 2)
    max_hw = tuple(args.latent_max_hw) if args.latent_max_hw is not None else (base * 4, base * 4)
    return min_hw, opt_hw, max_hw


def _engine_profiles(batch_size: int, sf: int, latent_min_hw, latent_opt_hw, latent_max_hw):
    lh_min, lw_min = latent_min_hw
    lh_opt, lw_opt = latent_opt_hw
    lh_max, lw_max = latent_max_hw
    return {
        "encoder": {
            "image": {
                "min": (batch_size, 3, lh_min * sf, lw_min * sf),
                "opt": (batch_size, 3, lh_opt * sf, lw_opt * sf),
                "max": (batch_size, 3, lh_max * sf, lw_max * sf),
            }
        },
        "unet": {
            "x": {
                "min": (batch_size, 3, lh_min, lw_min),
                "opt": (batch_size, 3, lh_opt, lw_opt),
                "max": (batch_size, 3, lh_max, lw_max),
            },
            "lq": {
                "min": (batch_size, 3, lh_min, lw_min),
                "opt": (batch_size, 3, lh_opt, lw_opt),
                "max": (batch_size, 3, lh_max, lw_max),
            },
            "timesteps": {
                "min": (batch_size,),
                "opt": (batch_size,),
                "max": (batch_size,),
            },
        },
        "decoder": {
            "latent": {
                "min": (batch_size, 3, lh_min, lw_min),
                "opt": (batch_size, 3, lh_opt, lw_opt),
                "max": (batch_size, 3, lh_max, lw_max),
            }
        },
    }


def _parse_onnx(parser: trt.OnnxParser, onnx_path: Path) -> None:
    if not parser.parse(onnx_path.read_bytes()):
        errors = []
        for idx in range(parser.num_errors):
            errors.append(str(parser.get_error(idx)))
        raise RuntimeError(
            f"Failed to parse ONNX: {onnx_path}\n" + "\n".join(errors)
        )


def _create_builder(logger: trt.Logger) -> trt.Builder:
    try:
        builder = trt.Builder(logger)
    except TypeError as exc:
        raise RuntimeError(
            "Failed to create TensorRT builder. "
            "This usually means TensorRT/CUDA runtime initialization failed in the current environment."
        ) from exc
    if builder is None:
        raise RuntimeError("TensorRT builder creation returned None.")
    return builder


def _add_profile(
    builder: trt.Builder,
    network: trt.INetworkDefinition,
    config: trt.IBuilderConfig,
    shape_spec: Dict[str, Dict[str, Tuple[int, ...]]],
) -> None:
    profile = builder.create_optimization_profile()
    for idx in range(network.num_inputs):
        tensor = network.get_input(idx)
        if tensor.name not in shape_spec:
            continue
        shapes = shape_spec[tensor.name]
        profile.set_shape(tensor.name, shapes["min"], shapes["opt"], shapes["max"])
    if not profile:
        raise RuntimeError("Failed to create TensorRT optimization profile.")
    config.add_optimization_profile(profile)


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    shape_spec: Dict[str, Dict[str, Tuple[int, ...]]],
    args: argparse.Namespace,
) -> None:
    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = _create_builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    _parse_onnx(parser, onnx_path)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1024 ** 3))
    )
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = int(args.optimization_level)
    if args.fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print(f"[warn] FP16 requested but platform_has_fast_fp16 is False: {onnx_path.name}")

    _add_profile(builder, network, config, shape_spec)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"Failed to build TensorRT engine from: {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)

    print(f"[ok] Built engine: {engine_path}")
    for input_name, profile in shape_spec.items():
        print(
            f"  {input_name}: min={_shape_str(profile['min'])}, "
            f"opt={_shape_str(profile['opt'])}, max={_shape_str(profile['max'])}"
        )


def main() -> None:
    args = _parse_args()
    cfg = OmegaConf.load(str(_resolve_path(args.config)))
    engine_dir = _resolve_path(args.engine_dir)
    sf = int(cfg.diffusion.params.get("sf", 4))
    latent_min_hw, latent_opt_hw, latent_max_hw = _default_latent_shapes(cfg, args)
    profiles = _engine_profiles(args.batch_size, sf, latent_min_hw, latent_opt_hw, latent_max_hw)

    model_names = _collect_model_names(args.models)
    onnx_paths = {
        "encoder": _resolve_path(args.encoder_onnx),
        "unet": _resolve_path(args.unet_onnx),
        "decoder": _resolve_path(args.decoder_onnx),
    }

    for model_name in model_names:
        onnx_path = onnx_paths[model_name]
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found for {model_name}: {onnx_path}")

        engine_path = engine_dir / f"{onnx_path.stem}.engine"
        if args.skip_existing and engine_path.exists():
            print(f"[skip] Engine already exists: {engine_path}")
            continue

        print(f"[build] {model_name}: {onnx_path}")
        _build_engine(onnx_path, engine_path, profiles[model_name], args)


if __name__ == "__main__":
    main()
