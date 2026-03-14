#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import tensorrt as trt
import torch


TRT_LOGGER_SEVERITY = {
    "internal_error": trt.Logger.INTERNAL_ERROR,
    "error": trt.Logger.ERROR,
    "warning": trt.Logger.WARNING,
    "info": trt.Logger.INFO,
    "verbose": trt.Logger.VERBOSE,
}


TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
}


def _ensure_contiguous(array: np.ndarray) -> np.ndarray:
    if array.flags.c_contiguous:
        return array
    return np.ascontiguousarray(array)


class TensorRTRunner:
    """
    Minimal native TensorRT runner for engines with explicit batch and dynamic shapes.

    The class intentionally mirrors the OnnxRunner interface used by
    onnx_inference/run_onnx_pipeline.py:
        runner = TensorRTRunner("model.engine")
        output = runner.run({"input": np_array})
    """

    def __init__(
        self,
        engine_path: Union[str, Path],
        device: str = "cuda:0",
        profile_index: int = 0,
        logger_severity: str = "error",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda is not available. TensorRTRunner requires a CUDA-capable PyTorch runtime."
            )

        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.device = torch.device(device)
        self.profile_index = int(profile_index)
        self.logger = trt.Logger(TRT_LOGGER_SEVERITY[logger_severity.lower()])
        self.runtime = trt.Runtime(self.logger)

        engine_bytes = self.engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")

        self.stream = torch.cuda.Stream(device=self.device)
        self.tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.input_names = [
            name
            for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name
            for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        self._torch_dtypes = {
            name: TRT_TO_TORCH[self.engine.get_tensor_dtype(name)] for name in self.tensor_names
        }

        torch.cuda.set_device(self.device)
        self.context.set_optimization_profile_async(self.profile_index, self.stream.cuda_stream)

    def _prepare_input(self, name: str, array: np.ndarray) -> torch.Tensor:
        trt_dtype = self.engine.get_tensor_dtype(name)
        np_dtype = np.dtype(trt.nptype(trt_dtype))
        host_array = np.asarray(array, dtype=np_dtype)
        host_array = _ensure_contiguous(host_array)
        device_tensor = torch.from_numpy(host_array).to(self.device, non_blocking=False)
        self.context.set_input_shape(name, tuple(host_array.shape))
        self.context.set_tensor_address(name, int(device_tensor.data_ptr()))
        return device_tensor

    def _allocate_output(self, name: str) -> torch.Tensor:
        shape = tuple(self.context.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(
                f"Output shape for tensor `{name}` is unresolved: {shape}. "
                "Input shapes/profile are likely incomplete."
            )
        output_tensor = torch.empty(shape, dtype=self._torch_dtypes[name], device=self.device)
        self.context.set_tensor_address(name, int(output_tensor.data_ptr()))
        return output_tensor

    def run(
        self,
        feeds: Dict[str, np.ndarray],
        output_names: Optional[Sequence[str]] = None,
        return_dict: bool = False,
    ):
        missing = [name for name in self.input_names if name not in feeds]
        if missing:
            raise KeyError(f"Missing TensorRT inputs: {missing}")

        selected_outputs: List[str] = list(output_names) if output_names is not None else self.output_names
        input_tensors: Dict[str, torch.Tensor] = {}
        output_tensors: Dict[str, torch.Tensor] = {}

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            for name in self.input_names:
                input_tensors[name] = self._prepare_input(name, feeds[name])

            unresolved = self.context.infer_shapes()
            if unresolved:
                raise RuntimeError(f"TensorRT shape inference failed for tensors: {unresolved}")

            for name in self.output_names:
                output_tensors[name] = self._allocate_output(name)

            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError(f"TensorRT execution failed: {self.engine_path}")

        self.stream.synchronize()
        outputs = {
            name: output_tensors[name].detach().cpu().numpy().copy() for name in selected_outputs
        }
        if return_dict:
            return outputs
        if len(outputs) == 1:
            return outputs[selected_outputs[0]]
        return [outputs[name] for name in selected_outputs]

    def get_tensor_shapes(self) -> Dict[str, tuple]:
        return {name: tuple(self.engine.get_tensor_shape(name)) for name in self.tensor_names}

