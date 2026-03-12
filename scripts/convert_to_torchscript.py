#!/usr/bin/env python3
"""Convert MobileFaceDetector PyTorch model to TorchScript (.pt) format."""

import argparse

import torch

from face_detection import MobileFaceDetector


def convert_to_torchscript(
    model_path: str,
    output_path: str,
    input_size: int = 256,
    method: str = "trace",
):
    print(f"Loading model from {model_path}...")
    model = MobileFaceDetector()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"Converting to TorchScript using {method}...")

    if method == "trace":
        scripted_model = torch.jit.trace(model, dummy_input)
    else:
        scripted_model = torch.jit.script(model)

    scripted_model.save(output_path)
    print(f"TorchScript model saved to {output_path}")

    print("\nVerifying TorchScript model...")
    loaded_model = torch.jit.load(output_path)

    with torch.no_grad():
        original_output = model(dummy_input)
        scripted_output = loaded_model(dummy_input)

    diff = torch.abs(original_output - scripted_output).max().item()
    print(f"Max difference between original and scripted: {diff:.6e}")

    if diff < 1e-5:
        print("TorchScript model verification passed")
    else:
        print("Warning: Output difference detected")

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {scripted_output.shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TorchScript")
    parser.add_argument(
        "--model",
        type=str,
        default="weights/mobile_face_detector_epoch_10.pth",
        help="Path to PyTorch model weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/mobile_face_detector.pt",
        help="Output TorchScript file path",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Input image size (default: 256, matches training)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["trace", "script"],
        default="trace",
        help="Conversion method (default: trace)",
    )
    args = parser.parse_args()

    convert_to_torchscript(
        model_path=args.model,
        output_path=args.output,
        input_size=args.input_size,
        method=args.method,
    )


if __name__ == "__main__":
    main()
