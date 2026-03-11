from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from quantization.awq import AWQConfig, AWQLinear, materialize_awq_caches, quantize_model_awq
from quantization.int8_ptq import INT8PTQConfig, INT8Conv2d, INT8Linear, materialize_int8_caches, quantize_model_int8_ptq


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def quantize_model_hybrid(
    model: nn.Module,
    activation_map: Dict[str, torch.Tensor],
    awq_config: AWQConfig,
    int8_config: INT8PTQConfig,
) -> Tuple[nn.Module, Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    awq_model, awq_metadata = quantize_model_awq(model, activation_map, awq_config)
    hybrid_model, int8_metadata = quantize_model_int8_ptq(awq_model, int8_config, skip_names=awq_metadata.keys())
    return hybrid_model, awq_metadata, int8_metadata


def save_hybrid_checkpoint(
    output_path: str,
    quantized_model: nn.Module,
    awq_metadata: Dict[str, Dict[str, object]],
    int8_metadata: Dict[str, Dict[str, object]],
    awq_config: AWQConfig,
    int8_config: INT8PTQConfig,
    source_config_path: str,
) -> None:
    payload = {
        "format": "reshift-hybrid-ptq-v1",
        "awq_config": awq_config.to_dict(),
        "int8_config": int8_config.to_dict(),
        "model_state": quantized_model.state_dict(),
        "awq_layers": awq_metadata,
        "int8_layers": int8_metadata,
        "source_config": source_config_path,
    }
    torch.save(payload, output_path)


def is_hybrid_checkpoint_payload(payload: object) -> bool:
    return isinstance(payload, dict) and payload.get("format") == "reshift-hybrid-ptq-v1"


def _replace_awq_modules(model: nn.Module, awq_layers: Dict[str, Dict[str, object]], dequant_mode: str) -> None:
    for name, info in awq_layers.items():
        parent, child_name = _get_parent_module(model, name)
        quant_layer = AWQLinear(
            int(info["in_features"]),
            int(info["out_features"]),
            bool(info["bias"]),
            n_bits=int(info["n_bits"]),
            group_size=int(info["group_size"]),
        )
        quant_layer.set_dequant_mode(dequant_mode)
        setattr(parent, child_name, quant_layer)


def _replace_int8_modules(model: nn.Module, int8_layers: Dict[str, Dict[str, object]], dequant_mode: str) -> None:
    for name, info in int8_layers.items():
        parent, child_name = _get_parent_module(model, name)
        if info["module_type"] == "conv2d":
            quant_layer = INT8Conv2d(
                int(info["in_channels"]),
                int(info["out_channels"]),
                tuple(info["kernel_size"]),
                tuple(info["stride"]),
                tuple(info["padding"]),
                tuple(info["dilation"]),
                int(info["groups"]),
                bool(info["bias"]),
                str(info["padding_mode"]),
                n_bits=int(info["n_bits"]),
            )
        elif info["module_type"] == "linear":
            quant_layer = INT8Linear(
                int(info["in_features"]),
                int(info["out_features"]),
                bool(info["bias"]),
                n_bits=int(info["n_bits"]),
            )
        else:
            raise ValueError(f"Unsupported INT8 module type: {info['module_type']}")
        quant_layer.set_dequant_mode(dequant_mode)
        setattr(parent, child_name, quant_layer)


def load_hybrid_quantized_model_from_payload(
    model: nn.Module,
    payload: Dict[str, object],
    device: torch.device,
    dequant_mode: str = "cached",
) -> Tuple[nn.Module, Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    if not is_hybrid_checkpoint_payload(payload):
        raise ValueError(f"Unsupported hybrid checkpoint format: {payload.get('format')}")

    awq_layers = payload["awq_layers"]
    int8_layers = payload["int8_layers"]
    _replace_awq_modules(model, awq_layers, dequant_mode)
    _replace_int8_modules(model, int8_layers, dequant_mode)

    current_state = model.state_dict()
    payload_state = payload["model_state"]
    filtered_state = {key: value for key, value in payload_state.items() if key in current_state}
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    if dequant_mode == "cached":
        materialize_awq_caches(model, device)
        materialize_int8_caches(model, device)
    model.eval()
    return model, awq_layers, int8_layers


def load_hybrid_quantized_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    dequant_mode: str = "cached",
) -> Tuple[nn.Module, Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    return load_hybrid_quantized_model_from_payload(model, payload, device, dequant_mode=dequant_mode)
