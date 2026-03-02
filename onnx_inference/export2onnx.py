import torch
import torch.nn as nn
import torch.onnx
import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf

# Ensure project root is importable when this script is launched by path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import util_common, util_net, util_opts

# Check for onnx module
try:
    import onnx
except ImportError:
    onnx = None


def get_export_parser():
    """Get command line arguments for ONNX export"""
    parser = argparse.ArgumentParser(description="Export ResShift model to ONNX format")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/export_config.yaml",
        help="Path to export config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to export on",
    )
    parser.add_argument(
        "--unet",
        type=str,
        default="True",
        help="Whether to export the UNet model (default: True)",
    )
    parser.add_argument(
        "--autoencoder",
        type=str,
        default="True",
        help="Whether to export the autoencoder encoder/decoder (default: True)",
    )
    args = parser.parse_args()
    return args


class ResShiftExportWrapper(nn.Module):
    """
    Wrapper module for ResShift model export.
    Exports the UNet model for a single diffusion step.
    """
    def __init__(self, model, lq_size=64, scale_factor=1.0):
        super().__init__()
        self.model = model
        self.lq_size = lq_size
        self.scale_factor = scale_factor
    
    def forward(self, x, lq, timesteps):
        """
        Forward pass for single diffusion step.
        
        Args:
            x: Current latent representation [B, 3, H, W] - feature space
            lq: Low-quality condition image [B, 3, H, W]
            timesteps: Diffusion timestep [B]
        
        Returns:
            Denoised output [B, 3, H, W]
        """
        # Ensure timesteps is the right shape
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        
        # Call the model with all required parameters
        output = self.model(
            x=x,
            timesteps=timesteps,
            lq=lq,
            mask=None
        )
        
        return output

class EncoderExportWrapper(nn.Module):
    def __init__(self, autoencoder: nn.Module):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)


class DecoderExportWrapper(nn.Module):
    def __init__(self, autoencoder: nn.Module):
        super().__init__()
        self.autoencoder = autoencoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

def export_unet(args, configs):
    # Verify checkpoint exists
    ckpt_path = Path(configs.model.ckpt_path).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_from_project = (PROJECT_ROOT / ckpt_path).resolve()
        ckpt_from_config = (PROJECT_ROOT / ckpt_path).resolve()
        ckpt_path = ckpt_from_project if ckpt_from_project.exists() else ckpt_from_config
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    configs.model.ckpt_path = str(ckpt_path)

    output_path = Path(configs.export.unet_output_path).expanduser()
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    configs.export.unet_output_path = str(output_path)
    
    print(f"\n=== ONNX Export Configuration ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output: {configs.export.unet_output_path}")
    print(f"Input shape: {configs.export.input_shape}")
    print(f"Opset version: {configs.export.opset_version}")
    print(f"Device: {args.device}\n")

    if onnx is None:
        raise ImportError("onnx is not installed. Please run: pip install onnx onnxruntime")
    
    # Build model
    print("Building model from config...")
    model = util_common.instantiate_from_config(configs.model)
    if args.device == "cuda":
        model = model.cuda()
    
    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=f"{args.device}:0" if args.device == "cuda" else "cpu")
    if 'state_dict' in ckpt:
        util_net.reload_model(model, ckpt['state_dict'])
    else:
        util_net.reload_model(model, ckpt)
    
    model.eval()
    print("Model loaded successfully.")
    
    # Wrap the model for export
    wrapper_model = ResShiftExportWrapper(
        model, 
        lq_size=configs.model.params.lq_size,
    )
    if args.device == "cuda":
        wrapper_model = wrapper_model.cuda()
    wrapper_model.eval()
    
    # Create dummy inputs
    B, C, H, W = configs.export.input_shape
    lq_H, lq_W = configs.model.params.lq_size, configs.model.params.lq_size
    
    # 当前扩散步输入
    x_input = torch.randn(B, C, H, W)
    
    # lq: low quailty 条件输入
    lq_input = torch.randn(B, C, lq_H, lq_W)
    
    # timesteps: diffusion timestep
    timesteps_input = torch.full((B,), 10, dtype=torch.long)
    
    if args.device == "cuda":
        x_input = x_input.cuda()
        lq_input = lq_input.cuda()
        timesteps_input = timesteps_input.cuda()
    
    print(f"Creating dummy inputs:")
    print(f"  x shape: {x_input.shape}")
    print(f"  lq shape: {lq_input.shape}")
    print(f"  timesteps shape: {timesteps_input.shape}")
    
    # Export to ONNX
    print("\nExporting model to ONNX format...")

    # Disable gradients for export
    with torch.no_grad():
        torch.onnx.export(
            wrapper_model,
            (x_input, lq_input, timesteps_input),
            configs.export.unet_output_path,
            opset_version=configs.export.opset_version,
            input_names=['x', 'lq', 'timesteps'],
            output_names=[configs.export.output_name],
            dynamic_axes={
                'x': {0: 'batch', 2: 'height', 3: 'width'},
                'lq': {0: 'batch', 2: 'height', 3: 'width'},
                'timesteps': {0: 'batch'},
                configs.export.output_name: {0: 'batch', 2: 'height', 3: 'width'}
            },
            verbose=configs.export.verbose,
            do_constant_folding=configs.export.do_constant_folding,
            export_params=True,
        )
    
    output_file = Path(configs.export.unet_output_path)
    file_size = output_file.stat().st_size / (1024**2)  # Convert to MB
    print(f"\n✓ UNet Model exported successfully!")
    print(f"  Output file: {configs.export.unet_output_path}")
    print(f"\nInputs:")
    print(f"  - x: Latent representation [batch, 3, {H}, {W}]")
    print(f"  - lq: Low-quality condition [batch, 3, {H}, {W}]")
    print(f"  - timesteps: Diffusion timestep [batch]")
    print(f"\nOutput:")
    print(f"  - output: Denoised latent [batch, 3, {H}, {W}]")
    pass

def export_autoencoder(args, configs):
    if "autoencoder" not in configs:
        raise KeyError(
            f"Config does not contain `autoencoder`. "
            "Please use a training/inference config such as configs/realsr_swinunet_realesrgan256.yaml."
        )
    ae = util_common.instantiate_from_config(configs.autoencoder)

    ckpt_path = (PROJECT_ROOT / configs.autoencoder.ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=args.device)
    if "state_dict" in ckpt:
        util_net.reload_model(ae, ckpt["state_dict"])
    else:
        util_net.reload_model(ae, ckpt)

    ae.eval()
    if args.device == "cuda":
        ae = ae.cuda()

    lq_size = int(configs.model.params.get("lq_size", 64))
    b, c, h, w = 1, 3, lq_size, lq_size

    sf = int(configs.diffusion.params.get("sf", 4)) if "diffusion" in configs else 4
    scale_factor = (
        float(configs.diffusion.params.get("scale_factor", 1.0)) if "diffusion" in configs else 1.0
    )

    # lq 被超分到目标分辨率再进行 encoder
    enc_input = torch.randn(b, c, h * sf, w * sf, dtype=torch.float32)
    print(f"encoder input shape: {enc_input.shape}")
    dec_input = torch.randn(b, c, h, w, dtype=torch.float32) / max(scale_factor, 1e-8)
    print(f"decoder input shape: {dec_input.shape}")

    if args.device == "cuda":
        enc_input = enc_input.cuda()
        dec_input = dec_input.cuda()

    encoder_wrapper = EncoderExportWrapper(ae).eval()
    decoder_wrapper = DecoderExportWrapper(ae).eval()
    if args.device == "cuda":
        encoder_wrapper = encoder_wrapper.cuda()
        decoder_wrapper = decoder_wrapper.cuda()


    with torch.no_grad():
        torch.onnx.export(
            encoder_wrapper,
            (enc_input,),
            configs.export.encoder_output_path,
            opset_version=configs.export.opset_version,
            input_names=["image"],
            output_names=["latent"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "latent": {0: "batch", 2: "height", 3: "width"},
            },
            do_constant_folding=configs.export.do_constant_folding,
            export_params=True,
        )

        torch.onnx.export(
            decoder_wrapper,
            (dec_input,),
            configs.export.decoder_output_path,
            opset_version=configs.export.opset_version,
            input_names=["latent"],
            output_names=["image"],
            dynamic_axes={
                "latent": {0: "batch", 2: "height", 3: "width"},
                "image": {0: "batch", 2: "height", 3: "width"},
            },
            do_constant_folding=configs.export.do_constant_folding,
            export_params=True,
        )

    print("AutoEncoder Export successfully!")
    print(f"  Encoder: {configs.export.encoder_output_path}")
    print(f"  Decoder: {configs.export.decoder_output_path}")


def main():
    args = get_export_parser()
    
    # Load export config
    print(f"Loading config from {args.config}...")
    config_path = Path(args.config).expanduser().resolve()
    configs = OmegaConf.load(str(config_path))

    print(f"export args:\n export unet: {args.unet}\n export autoencoder: {args.autoencoder}\n")

    if(util_opts.str2bool(args.unet)):
        print("\n=================exporting Unet Model...=================")
        export_unet(args, configs)
    if(util_opts.str2bool(args.autoencoder)):
        print("\n=================exporting Autoencoder Encoder/Decoder...=================")
        export_autoencoder(args, configs)


if __name__ == '__main__':
    main()
