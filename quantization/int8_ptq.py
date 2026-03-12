from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class INT8PTQConfig:
    n_bits: int = 8
    quantize_conv: bool = True
    quantize_linear: bool = True
    include_patterns: Tuple[str, ...] = ()
    exclude_patterns: Tuple[str, ...] = ()
    eps: float = 1e-6

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class INT8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        *,
        n_bits: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits

        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None
        self.register_buffer("weight_cache", torch.empty(0), persistent=False)
        self.register_buffer("bias_cache", torch.empty(0), persistent=False)
        self.dequant_mode = "cached"

    def _dequantize_weight(self) -> torch.Tensor:
        return self.qweight.float() * self.scales.unsqueeze(1)

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
            weight = self._dequantize_weight().to(device=x.device, dtype=compute_dtype)
            bias = self.bias
            if bias is not None:
                bias = bias.to(device=x.device, dtype=compute_dtype)
        out = F.linear(x_compute, weight, bias)
        return out.to(dtype=x.dtype)


class INT8Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
        bias: bool,
        padding_mode: str,
        *,
        n_bits: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.n_bits = n_bits
        self._reversed_padding_repeated_twice = (
            padding[1],
            padding[1],
            padding[0],
            padding[0],
        )

        self.register_buffer(
            "qweight",
            torch.zeros(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], dtype=torch.int8),
        )
        self.register_buffer("scales", torch.ones(out_channels, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.bias = None
        self.register_buffer("weight_cache", torch.empty(0), persistent=False)
        self.register_buffer("bias_cache", torch.empty(0), persistent=False)
        self.dequant_mode = "cached"

    def _dequantize_weight(self) -> torch.Tensor:
        return self.qweight.float() * self.scales.view(-1, 1, 1, 1)

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

    def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return F.conv2d(x, weight, bias, self.stride, padding, self.dilation, self.groups)

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
            weight = self._dequantize_weight().to(device=x.device, dtype=compute_dtype)
            bias = self.bias
            if bias is not None:
                bias = bias.to(device=x.device, dtype=compute_dtype)
        out = self._conv_forward(x_compute, weight, bias)
        return out.to(dtype=x.dtype)


def _should_quantize(name: str, module: nn.Module, config: INT8PTQConfig) -> bool:
    is_conv = isinstance(module, nn.Conv2d)
    is_linear = isinstance(module, nn.Linear)
    if is_conv and not config.quantize_conv:
        return False
    if is_linear and not config.quantize_linear:
        return False
    if not (is_conv or is_linear):
        return False
    if config.include_patterns and not any(pattern in name for pattern in config.include_patterns):
        return False
    if config.exclude_patterns and any(pattern in name for pattern in config.exclude_patterns):
        return False
    return True


def _quantize_per_output_channel(weight: torch.Tensor, n_bits: int, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -2 ** (n_bits - 1)
    flat = weight.reshape(weight.shape[0], -1)
    max_abs = flat.abs().amax(dim=1, keepdim=True).clamp(min=eps)
    scale = max_abs / float(qmax)
    qflat = torch.round(flat / scale).clamp(qmin, qmax).to(torch.int8)
    qweight = qflat.reshape_as(weight)
    return qweight.cpu(), scale.squeeze(1).cpu()


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def collect_int8_target_layers(
    model: nn.Module,
    config: INT8PTQConfig,
    skip_names: Iterable[str] = (),
) -> Dict[str, nn.Module]:
    skip = set(skip_names)
    return {
        name: module
        for name, module in model.named_modules()
        if name not in skip and _should_quantize(name, module, config)
    }


def quantize_model_int8_ptq(
    model: nn.Module,
    config: INT8PTQConfig,
    skip_names: Iterable[str] = (),
) -> Tuple[nn.Module, Dict[str, Dict[str, object]]]:
    quantized_model = copy.deepcopy(model).cpu().eval()
    metadata: Dict[str, Dict[str, object]] = {}

    for name, module in list(quantized_model.named_modules()):
        if name in skip_names or not _should_quantize(name, module, config):
            continue

        qweight, scales = _quantize_per_output_channel(module.weight.detach().float().cpu(), config.n_bits, config.eps)
        if isinstance(module, nn.Conv2d):
            quant_layer = INT8Conv2d(
                module.in_channels,
                module.out_channels,
                tuple(module.kernel_size),
                tuple(module.stride),
                tuple(module.padding),
                tuple(module.dilation),
                int(module.groups),
                module.bias is not None,
                module.padding_mode,
                n_bits=config.n_bits,
            )
            metadata[name] = {
                "module_type": "conv2d",
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": tuple(module.kernel_size),
                "stride": tuple(module.stride),
                "padding": tuple(module.padding),
                "dilation": tuple(module.dilation),
                "groups": int(module.groups),
                "bias": module.bias is not None,
                "padding_mode": module.padding_mode,
                "n_bits": config.n_bits,
            }
        else:
            quant_layer = INT8Linear(
                module.in_features,
                module.out_features,
                module.bias is not None,
                n_bits=config.n_bits,
            )
            metadata[name] = {
                "module_type": "linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
                "n_bits": config.n_bits,
            }

        quant_layer.qweight.copy_(qweight)
        quant_layer.scales.copy_(scales)
        if module.bias is not None:
            quant_layer.bias.copy_(module.bias.detach().float().cpu())

        parent, child_name = _get_parent_module(quantized_model, name)
        setattr(parent, child_name, quant_layer)

    return quantized_model, metadata


def materialize_int8_caches(model: nn.Module, device: torch.device) -> None:
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    for module in model.modules():
        if isinstance(module, (INT8Linear, INT8Conv2d)):
            module.materialize_cache(device=device, dtype=target_dtype)


def restore_int8_conv_modules(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> int:
    target_dtype = dtype if dtype is not None else (torch.float16 if device.type == "cuda" else torch.float32)
    restored = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, INT8Conv2d):
            continue
        conv = nn.Conv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        ).to(device=device, dtype=target_dtype)
        with torch.no_grad():
            conv.weight.copy_(module._dequantize_weight().to(device=device, dtype=target_dtype))
            if module.bias is not None:
                conv.bias.copy_(module.bias.to(device=device, dtype=target_dtype))
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, conv)
        restored += 1
    return restored
