#!/usr/bin/env python3
"""Convert MobileFaceDetector PyTorch model to ONNX format."""

import argparse

import numpy as np
import torch

from face_detection import MobileFaceDetector


def convert_to_onnx(
    model_path: str,
    output_path: str,
    input_size: int = 256,
    opset_version: int = 12,
):
    print(f"Loading model from {model_path}...")
    model = MobileFaceDetector()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    if input_size == 256:
        print("Warning: Input size 256 may cause issues with AdaptiveAvgPool2d export.")
        print("Using 224 is recommended for full ONNX compatibility.")

    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"Converting to ONNX (opset version {opset_version})...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"ONNX model saved to {output_path}")

    print("\nVerifying ONNX model...")
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")

    print("\nTesting inference with ONNX Runtime...")
    try:
        import onnxruntime as ort

        ort_session = ort.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        with torch.no_grad():
            torch_out = model(dummy_input)

        np.testing.assert_allclose(
            torch_out.numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05
        )
        print("ONNX Runtime inference test passed")
        print(f"  Output shape: {ort_outputs[0].shape}")
    except ImportError:
        print("onnxruntime not installed, skipping inference test")
        print("  Install with: uv add onnxruntime")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="weights/mobile_face_detector_epoch_10.pth",
        help="Path to PyTorch model weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/mobile_face_detector.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size (default: 256, matches training)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12)",
    )
    args = parser.parse_args()

    convert_to_onnx(
        model_path=args.model,
        output_path=args.output,
        input_size=args.input_size,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
