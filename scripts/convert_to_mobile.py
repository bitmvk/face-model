#!/usr/bin/env python3
"""Convert MobileFaceDetector PyTorch model to TFLite format for mobile."""

import argparse

import torch
import litert_torch

from face_detection import MobileFaceDetector


def convert_to_mobile(
    model_path: str,
    output_path: str,
    input_size: int = 256,
):
    print(f"Loading model from {model_path}...")
    model = MobileFaceDetector()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = (torch.randn(1, 3, input_size, input_size),)

    torch_out = model(*dummy_input)

    print("Converting to TFLite...")
    edge_model = litert_torch.convert(model, dummy_input)
    edge_model.export(output_path)

    print(f"TFLite model saved to {output_path}")

    out = edge_model(*dummy_input)
    print(f"Output shape: {out.shape}")
    print(f"Output values: {out}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TFLite")
    parser.add_argument(
        "--model",
        type=str,
        default="weights/mobile_face_detector_epoch_10.pth",
        help="Path to PyTorch model weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/mobile_face_detector.tflite",
        help="Output TFLite file path",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Input image size (default: 256, matches training)",
    )
    args = parser.parse_args()

    convert_to_mobile(
        model_path=args.model,
        output_path=args.output,
        input_size=args.input_size,
    )


if __name__ == "__main__":
    main()
