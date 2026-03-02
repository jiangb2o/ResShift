#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    from onnx.reference import ReferenceEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ONNX pipeline artifacts.")
    parser.add_argument("--unet_onnx", type=str, default="weights/resshift_model.onnx")
    parser.add_argument(
        "--encoder_onnx", type=str, default="onnx_inference/models/autoencoder_encoder.onnx"
    )
    parser.add_argument(
        "--decoder_onnx", type=str, default="onnx_inference/models/autoencoder_decoder.onnx"
    )
    parser.add_argument("--report", type=str, default="onnx_inference/outputs/validation_report.json")
    return parser.parse_args()


def check_model(path: Path) -> dict:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return {
        "path": str(path),
        "inputs": [x.name for x in model.graph.input],
        "outputs": [x.name for x in model.graph.output],
    }


def run_encoder_once(path: Path) -> dict:
    x = np.random.randn(1, 3, 256, 256).astype(np.float32)
    if ort is not None:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = session.run(None, {"image": x})[0]
        backend = "onnxruntime"
    else:
        model = onnx.load(str(path))
        session = ReferenceEvaluator(model)
        out = session.run(None, {"image": x})[0]
        backend = "onnx-reference"
    return {
        "backend": backend,
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
        "output_mean": float(out.mean()),
    }


def main() -> None:
    args = parse_args()
    unet = (PROJECT_ROOT / args.unet_onnx).resolve()
    encoder = (PROJECT_ROOT / args.encoder_onnx).resolve()
    decoder = (PROJECT_ROOT / args.decoder_onnx).resolve()
    report_path = (PROJECT_ROOT / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "onnxruntime_available": ort is not None,
        "models": {
            "unet": check_model(unet),
            "encoder": check_model(encoder),
            "decoder": check_model(decoder),
        },
        "encoder_runtime_test": run_encoder_once(encoder),
    }
    if ort is None:
        report["e2e_status"] = (
            "skipped: onnxruntime is not installed; install onnxruntime/onnxruntime-gpu to run full pipeline."
        )
    else:
        report["e2e_status"] = "ready: onnxruntime available, run run_onnx_pipeline.py for full inference."

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validation report saved to: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
