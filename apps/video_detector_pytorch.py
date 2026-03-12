#!/usr/bin/env python3
"""
Real-time camera stream face detection using MobileFaceDetector (PyTorch TorchScript).
Shows FPS and frame processing time.
"""

import torch
import numpy as np
import cv2
import time
from pathlib import Path
from PIL import Image

from face_detection import letterbox_image, extract_detection


def load_model(model_path, device_type="cuda"):
    device = torch.device(device_type if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model, device


def main():
    model_path = Path(__file__).parent.parent / "weights" / "mobile_face_detector.pt"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: uv run python scripts/convert_to_torchscript.py")
        return

    print(f"Loading model from {model_path}...")
    model, device = load_model(str(model_path))
    print(f"Model loaded on {device}")

    camera_id = 0
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera started. Press 'q' to quit.")

    fps = 0.0
    frame_times = []
    fps_window = 30

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame from camera")
            break

        frame_start = time.time()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        orig_h, orig_w = frame.shape[:2]

        img_tensor, scale, pad_left, pad_top = letterbox_image(pil_image)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)

        x, y, w, h, conf, left_eye, right_eye = extract_detection(
            outputs, orig_w, orig_h, scale, pad_left, pad_top
        )

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, left_eye, 4, (255, 0, 0), -1)
        cv2.circle(frame, right_eye, 4, (0, 0, 255), -1)

        frame_end = time.time()
        frame_time_ms = (frame_end - frame_start) * 1000

        frame_times.append(frame_end)
        if len(frame_times) > fps_window:
            frame_times.pop(0)

        if len(frame_times) >= 2:
            time_diff = frame_times[-1] - frame_times[0]
            if time_diff > 0:
                fps = (len(frame_times) - 1) / time_diff

        info_text = f"FPS: {fps:.1f} | Frame Time: {frame_time_ms:.1f}ms"
        cv2.putText(
            frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        cv2.imshow("Face Detection (PyTorch)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()
