#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Complete ResShift Export and Conversion Workflow
PyTorch -> ONNX -> MNN
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional


class ResShiftExportWorkflow:
    """Complete workflow for exporting ResShift to MNN format."""
    
    def __init__(
        self,
        resshift_root: str = "/home/ubuntu/ResShift",
        mnn_root: str = "/home/ubuntu/MNN",
    ):
        """Initialize workflow.
        
        Args:
            resshift_root: ResShift project root
            mnn_root: MNN project root
        """
        self.resshift_root = Path(resshift_root)
        self.mnn_root = Path(mnn_root)
        self.steps_completed = []
    
    def log_step(self, step_num: int, name: str, status: str = "pending"):
        """Log workflow step."""
        status_symbol = {
            "pending": "⚬",
            "running": "⊙",
            "completed": "✓",
            "failed": "✗",
            "skipped": "⊘",
        }.get(status, "?")
        
        print(f"  {status_symbol} Step {step_num}: {name}")
    
    def step1_export_to_onnx(
        self,
        config_path: str,
        output_onnx: str = "resshift_model.onnx",
        force: bool = False,
    ) -> bool:
        """Step 1: Export PyTorch model to ONNX.
        
        Args:
            config_path: Path to export config YAML
            output_onnx: Output ONNX file path
            force: Force re-export even if file exists
            
        Returns:
            True if successful
        """
        output_path = self.resshift_root / output_onnx
        
        # Check if already exists
        if output_path.exists() and not force:
            print(f"\n✓ ONNX model already exists: {output_path}")
            self.steps_completed.append("export_onnx")
            return True
        
        print(f"\n{'='*60}")
        print(f"Step 1: Export to ONNX")
        print(f"{'='*60}")
        
        script_path = self.resshift_root / "export2onxx.py"
        if not script_path.exists():
            print(f"✗ Export script not found: {script_path}")
            return False
        
        cmd = [
            sys.executable,
            str(script_path),
            "--config", config_path,
        ]
        
        print(f"Running: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, cwd=self.resshift_root)
        
        if result.returncode == 0 and output_path.exists():
            size_mb = output_path.stat().st_size / (1024**2)
            print(f"\n✓ ONNX export successful: {size_mb:.2f} MB")
            self.steps_completed.append("export_onnx")
            return True
        else:
            print(f"\n✗ ONNX export failed")
            return False
    
    def step2_convert_to_mnn(
        self,
        input_onnx: str = "resshift_model.onnx",
        output_mnn: str = "resshift_model.mnn",
        preset: str = "normal",
        force: bool = False,
    ) -> bool:
        """Step 2: Convert ONNX to MNN.
        
        Args:
            input_onnx: Input ONNX file path
            output_mnn: Output MNN file path
            preset: Optimization preset (normal, optimize, quantized)
            force: Force re-conversion
            
        Returns:
            True if successful
        """
        input_path = self.resshift_root / input_onnx
        output_path = self.resshift_root / output_mnn
        
        # Check input exists
        if not input_path.exists():
            print(f"\n✗ ONNX model not found: {input_path}")
            return False
        
        # Check if already exists
        if output_path.exists() and not force:
            print(f"\n✓ MNN model already exists: {output_path}")
            self.steps_completed.append("convert_mnn")
            return True
        
        print(f"\n{'='*60}")
        print(f"Step 2: Convert ONNX to MNN")
        print(f"{'='*60}")
        
        script_path = self.resshift_root / "onnx_to_mnn.py"
        if not script_path.exists():
            print(f"✗ Conversion script not found: {script_path}")
            return False
        
        cmd = [
            sys.executable,
            str(script_path),
            "-i", str(input_path),
            "-o", str(output_path),
            "--preset", preset,
            "--mnn-root", str(self.mnn_root),
        ]
        
        print(f"Running: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, cwd=self.resshift_root)
        
        if result.returncode == 0 and output_path.exists():
            size_mb = output_path.stat().st_size / (1024**2)
            print(f"\n✓ MNN conversion successful: {size_mb:.2f} MB")
            self.steps_completed.append("convert_mnn")
            return True
        else:
            print(f"\n✗ MNN conversion failed")
            return False
    
    def step3_verify_mnn(self, mnn_model: str = "resshift_model.mnn") -> bool:
        """Step 3: Verify MNN model.
        
        Args:
            mnn_model: MNN model file path
            
        Returns:
            True if successful
        """
        model_path = self.resshift_root / mnn_model
        
        if not model_path.exists():
            print(f"\n✗ MNN model not found: {model_path}")
            return False
        
        print(f"\n{'='*60}")
        print(f"Step 3: Verify MNN Model")
        print(f"{'='*60}")
        
        script_path = self.resshift_root / "verify_mnn_model.py"
        if not script_path.exists():
            print(f"✗ Verification script not found: {script_path}")
            return False
        
        cmd = [
            sys.executable,
            str(script_path),
            "-m", str(model_path),
            "--mnn-root", str(self.mnn_root),
        ]
        
        print(f"Running verification...\n")
        
        result = subprocess.run(cmd, cwd=self.resshift_root)
        
        if result.returncode == 0:
            self.steps_completed.append("verify_mnn")
            return True
        else:
            print(f"\n✗ Verification failed")
            return False
    
    def run_full_pipeline(
        self,
        config_path: str = "configs/export_config.yaml",
        mnn_preset: str = "normal",
        verify: bool = True,
        force: bool = False,
    ) -> bool:
        """Run complete export pipeline.
        
        Args:
            config_path: Path to export config
            mnn_preset: MNN optimization preset
            verify: Whether to verify the result
            force: Force re-do all steps
            
        Returns:
            True if all steps successful
        """
        print("\n" + "="*60)
        print("ResShift Export & Conversion Workflow")
        print("="*60)
        print("\nWorkflow Steps:")
        print("  1. Export PyTorch model to ONNX")
        print("  2. Convert ONNX to MNN format")
        print("  3. Verify MNN model")
        print("\nStarting workflow...\n")
        
        # Step 1: Export to ONNX
        self.log_step(1, "Export to ONNX", "running")
        if not self.step1_export_to_onnx(config_path, force=force):
            self.log_step(1, "Export to ONNX", "failed")
            return False
        self.log_step(1, "Export to ONNX", "completed")
        
        # Step 2: Convert to MNN
        self.log_step(2, "Convert to MNN", "running")
        if not self.step2_convert_to_mnn(preset=mnn_preset, force=force):
            self.log_step(2, "Convert to MNN", "failed")
            return False
        self.log_step(2, "Convert to MNN", "completed")
        
        # Step 3: Verify
        if verify:
            self.log_step(3, "Verify MNN", "running")
            if not self.step3_verify_mnn():
                self.log_step(3, "Verify MNN", "failed")
                return False
            self.log_step(3, "Verify MNN", "completed")
        else:
            self.log_step(3, "Verify MNN", "skipped")
        
        # Summary
        print(f"\n{'='*60}")
        print("Workflow Complete!")
        print(f"{'='*60}")
        print(f"\nCompleted Steps: {len(self.steps_completed)}/3")
        for i, step in enumerate(self.steps_completed, 1):
            print(f"  {i}. {step}")
        
        print(f"\nOutput Files:")
        onnx_file = self.resshift_root / "resshift_model.onnx"
        mnn_file = self.resshift_root / "resshift_model.mnn"
        
        if onnx_file.exists():
            print(f"  • ONNX: {onnx_file.name} ({onnx_file.stat().st_size / (1024**2):.2f} MB)")
        if mnn_file.exists():
            print(f"  • MNN:  {mnn_file.name} ({mnn_file.stat().st_size / (1024**2):.2f} MB)")
        
        print(f"\nNext Steps:")
        print(f"  1. Deploy MNN model on mobile/edge devices")
        print(f"  2. Use MNN inference engine for deployment")
        print(f"  3. Quantize model for further optimization (optional)")
        
        return True


def get_parser():
    parser = argparse.ArgumentParser(
        description="Complete ResShift export and conversion workflow"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/export_config.yaml",
        help="Path to export config YAML"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["normal", "optimize", "quantized"],
        default="normal",
        help="MNN optimization preset"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification step"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-do all steps even if outputs exist"
    )
    parser.add_argument(
        "--resshift-root",
        type=str,
        default="/home/ubuntu/ResShift",
        help="ResShift project root directory"
    )
    parser.add_argument(
        "--mnn-root",
        type=str,
        default="/home/ubuntu/MNN",
        help="MNN project root directory"
    )
    parser.add_argument(
        "--only-onnx",
        action="store_true",
        help="Only perform ONNX export, skip MNN conversion"
    )
    parser.add_argument(
        "--only-mnn",
        action="store_true",
        help="Only perform MNN conversion, skip ONNX export"
    )
    
    return parser.parse_args()


def main():
    args = get_parser()
    
    # Create workflow
    workflow = ResShiftExportWorkflow(
        resshift_root=args.resshift_root,
        mnn_root=args.mnn_root,
    )
    
    # Run workflow
    try:
        if args.only_onnx:
            # Only export to ONNX
            success = workflow.step1_export_to_onnx(
                args.config,
                force=args.force
            )
        elif args.only_mnn:
            # Only convert to MNN
            success = workflow.step2_convert_to_mnn(
                preset=args.preset,
                force=args.force
            )
        else:
            # Full pipeline
            success = workflow.run_full_pipeline(
                config_path=args.config,
                mnn_preset=args.preset,
                verify=not args.skip_verify,
                force=args.force,
            )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nWorkflow error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
