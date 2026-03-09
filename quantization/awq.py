from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AWQConfig:
    n_bits: int = 4
    group_size: int = 128
    alpha_samples: int = 11
    max_tokens_per_layer: int = 256
    include_patterns: Tuple[str, ...] = ("qkv", "proj", "reduction")
    exclude_patterns: Tuple[str, ...] = ()
    quantize_bias: bool = False
    eps: float = 1e-6

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class AWQLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        n_bits: int,
        group_size: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.group_size = group_size

        num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, num_groups, dtype=torch.float32))
        self.register_buffer("awq_scale", torch.ones(in_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None
        self.register_buffer("weight_cache", torch.empty(0), persistent=False)
        self.register_buffer("bias_cache", torch.empty(0), persistent=False)
        self.dequant_mode = "cached"

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        awq_scale: torch.Tensor,
    ) -> "AWQLinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            n_bits=4,
            group_size=awq_scale.shape[0] if awq_scale.numel() == linear.in_features else linear.in_features,
        )
        module.n_bits = 4
        module.group_size = int((linear.in_features + scales.shape[1] - 1) // scales.shape[1])
        module.qweight = qweight
        module.scales = scales
        module.awq_scale = awq_scale
        if linear.bias is not None:
            module.bias = linear.bias.detach().float()
        return module

    def _dequantize_weight(self) -> torch.Tensor:
        weight = torch.empty_like(self.qweight, dtype=torch.float32)
        for group_idx in range(self.scales.shape[1]):
            start = group_idx * self.group_size
            end = min(start + self.group_size, self.in_features)
            scale = self.scales[:, group_idx].unsqueeze(1)
            weight[:, start:end] = self.qweight[:, start:end].float() * scale
        weight = weight / self.awq_scale.unsqueeze(0)
        return weight

    def _compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        return torch.float16 if x.is_cuda else torch.float32

    def set_dequant_mode(self, mode: str) -> None:
        if mode not in {"cached", "dynamic"}:
            raise ValueError(f"Unsupported dequant mode: {mode}")
        self.dequant_mode = mode

    def materialize_cache(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        target_device = device if device is not None else self.qweight.device
        target_dtype = dtype if dtype is not None else (torch.float16 if target_device.type == "cuda" else torch.float32)
        self.weight_cache = self._dequantize_weight().to(device=target_device, dtype=target_dtype)
        if self.bias is not None:
            self.bias_cache = self.bias.to(device=target_device, dtype=target_dtype)
        else:
            self.bias_cache = torch.empty(0, device=target_device, dtype=target_dtype)

    def clear_cache(self) -> None:
        self.weight_cache = torch.empty(0, device=self.qweight.device)
        self.bias_cache = torch.empty(0, device=self.qweight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = self._compute_dtype(x)
        x_compute = x.to(dtype=compute_dtype)
        if self.dequant_mode == "cached":
            if (
                self.weight_cache.numel() == 0
                or self.weight_cache.device != x.device
                or self.weight_cache.dtype != compute_dtype
            ):
                self.materialize_cache(device=x.device, dtype=compute_dtype)
            weight = self.weight_cache
            bias = self.bias_cache if self.bias is not None else None
        else:
            weight = self._dequantize_weight().to(dtype=compute_dtype, device=x.device)
            bias = self.bias
            if bias is not None:
                bias = bias.to(dtype=compute_dtype, device=x.device)
        out = F.linear(x_compute, weight, bias)
        return out.to(dtype=x.dtype)


def _should_quantize(name: str, layer: nn.Module, config: AWQConfig) -> bool:
    if not isinstance(layer, nn.Linear):
        return False
    if config.include_patterns and not any(pattern in name for pattern in config.include_patterns):
        return False
    if config.exclude_patterns and any(pattern in name for pattern in config.exclude_patterns):
        return False
    return True


def _groupwise_quantize(weight: torch.Tensor, n_bits: int, group_size: int, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -2 ** (n_bits - 1)
    out_features, in_features = weight.shape
    num_groups = (in_features + group_size - 1) // group_size
    qweight = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty(out_features, num_groups, device=weight.device, dtype=torch.float32)

    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min(start + group_size, in_features)
        chunk = weight[:, start:end]
        max_abs = chunk.abs().amax(dim=1, keepdim=True).clamp(min=eps)
        scale = max_abs / float(qmax)
        qchunk = torch.round(chunk / scale).clamp(qmin, qmax).to(torch.int8)
        qweight[:, start:end] = qchunk
        scales[:, group_idx] = scale.squeeze(1)

    return qweight.cpu(), scales.cpu()


def _dequantize_groupwise(qweight: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    out_features, in_features = qweight.shape
    weight = torch.empty(out_features, in_features, dtype=torch.float32)
    for group_idx in range(scales.shape[1]):
        start = group_idx * group_size
        end = min(start + group_size, in_features)
        scale = scales[:, group_idx].unsqueeze(1)
        weight[:, start:end] = qweight[:, start:end].float() * scale
    return weight


def _sample_tokens(x: torch.Tensor, max_tokens: int) -> torch.Tensor:
    x = x.detach().float().reshape(-1, x.shape[-1])
    if x.shape[0] <= max_tokens:
        return x.cpu()
    indices = torch.linspace(0, x.shape[0] - 1, steps=max_tokens, device=x.device).long()
    return x[indices].cpu()


class ActivationCollector:
    def __init__(self, target_layers: Dict[str, nn.Linear], max_tokens_per_layer: int) -> None:
        self.target_layers = target_layers
        self.max_tokens_per_layer = max_tokens_per_layer
        self.activations: Dict[str, List[torch.Tensor]] = {name: [] for name in target_layers}
        self.handles = []

    def _hook(self, name: str):
        def fn(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            tensor = _sample_tokens(inputs[0], self.max_tokens_per_layer)
            self.activations[name].append(tensor)
        return fn

    def register(self) -> None:
        for name, module in self.target_layers.items():
            self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finalize(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for name, chunks in self.activations.items():
            if not chunks:
                continue
            out[name] = torch.cat(chunks, dim=0)
        return out


def _search_awq_scale(
    weight: torch.Tensor,
    activations: torch.Tensor,
    config: AWQConfig,
) -> torch.Tensor:
    if activations.numel() == 0:
        return torch.ones(weight.shape[1], dtype=torch.float32)

    act_mean = activations.abs().mean(dim=0).clamp(min=config.eps)
    reference = activations @ weight.t().cpu()
    best_error = None
    best_scale = torch.ones_like(act_mean)

    for alpha in torch.linspace(0.0, 1.0, steps=config.alpha_samples):
        scale = act_mean.pow(alpha.item())
        norm = torch.sqrt(scale.max() * scale.min()).clamp(min=config.eps)
        scale = (scale / norm).clamp(min=1e-4, max=1e4)
        scaled_weight = weight.cpu() * scale.unsqueeze(0)
        qweight, qscales = _groupwise_quantize(scaled_weight, config.n_bits, config.group_size, config.eps)
        dequant = _dequantize_groupwise(qweight, qscales, config.group_size) / scale.unsqueeze(0)
        candidate = activations @ dequant.t()
        error = F.mse_loss(candidate, reference).item()
        if best_error is None or error < best_error:
            best_error = error
            best_scale = scale

    return best_scale.float()


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def collect_target_layers(model: nn.Module, config: AWQConfig) -> Dict[str, nn.Linear]:
    return {
        name: module
        for name, module in model.named_modules()
        if _should_quantize(name, module, config)
    }


def collect_awq_activations(
    model: nn.Module,
    calibration_batches: Iterable[Dict[str, torch.Tensor]],
    config: AWQConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    targets = collect_target_layers(model, config)
    collector = ActivationCollector(targets, config.max_tokens_per_layer)
    collector.register()
    model.eval()
    with torch.no_grad():
        for batch in calibration_batches:
            model(
                x=batch["x"].to(device),
                timesteps=batch["timesteps"].to(device),
                lq=batch["lq"].to(device),
                mask=None,
            )
    collector.remove()
    return collector.finalize()


def quantize_model_awq(
    model: nn.Module,
    activation_map: Dict[str, torch.Tensor],
    config: AWQConfig,
) -> Tuple[nn.Module, Dict[str, Dict[str, object]]]:
    quantized_model = copy.deepcopy(model).cpu().eval()
    metadata: Dict[str, Dict[str, object]] = {}

    for name, module in list(quantized_model.named_modules()):
        if not _should_quantize(name, module, config):
            continue
        activations = activation_map.get(name)
        if activations is None:
            continue

        weight = module.weight.detach().float().cpu()
        awq_scale = _search_awq_scale(weight, activations.cpu(), config)
        scaled_weight = weight * awq_scale.unsqueeze(0)
        qweight, scales = _groupwise_quantize(scaled_weight, config.n_bits, config.group_size, config.eps)

        quant_layer = AWQLinear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            n_bits=config.n_bits,
            group_size=config.group_size,
        )
        quant_layer.qweight.copy_(qweight)
        quant_layer.scales.copy_(scales)
        quant_layer.awq_scale.copy_(awq_scale)
        if module.bias is not None:
            quant_layer.bias.copy_(module.bias.detach().float().cpu())

        parent, child_name = _get_parent_module(quantized_model, name)
        setattr(parent, child_name, quant_layer)
        metadata[name] = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "bias": module.bias is not None,
            "group_size": config.group_size,
            "n_bits": config.n_bits,
        }

    return quantized_model, metadata


def save_awq_checkpoint(
    output_path: str,
    quantized_model: nn.Module,
    metadata: Dict[str, Dict[str, object]],
    config: AWQConfig,
    source_config_path: str,
) -> None:
    payload = {
        "format": "reshift-awq-v1",
        "awq_config": config.to_dict(),
        "model_state": quantized_model.state_dict(),
        "quantized_layers": metadata,
        "source_config": source_config_path,
    }
    torch.save(payload, output_path)


def is_awq_checkpoint_payload(payload: object) -> bool:
    return isinstance(payload, dict) and payload.get("format") == "reshift-awq-v1"


def load_awq_quantized_model_from_payload(
    model: nn.Module,
    payload: Dict[str, object],
    device: torch.device,
    dequant_mode: str = "cached",
) -> Tuple[nn.Module, Dict[str, Dict[str, object]]]:
    if not is_awq_checkpoint_payload(payload):
        raise ValueError(f"Unsupported AWQ checkpoint format: {payload.get('format')}")

    quantized_layers = payload["quantized_layers"]
    for name, info in quantized_layers.items():
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

    current_state = model.state_dict()
    payload_state = payload["model_state"]
    filtered_state = {key: value for key, value in payload_state.items() if key in current_state}
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    if dequant_mode == "cached":
        materialize_awq_caches(model, device)
    model.eval()
    return model, quantized_layers


def load_awq_quantized_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    dequant_mode: str = "cached",
) -> Tuple[nn.Module, Dict[str, Dict[str, object]]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    return load_awq_quantized_model_from_payload(model, payload, device, dequant_mode=dequant_mode)


def materialize_awq_caches(model: nn.Module, device: torch.device) -> None:
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    for module in model.modules():
        if isinstance(module, AWQLinear):
            module.materialize_cache(device=device, dtype=target_dtype)
