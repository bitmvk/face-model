#!/usr/bin/env python3
"""
Real-time camera stream face detection using MobileFaceDetector.
Shows FPS and frame processing time.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import time
from pathlib import Path


class InvertedResidual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []

        if expand_ratio != 1:
            layers.extend(
                [
                    torch.nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                    torch.nn.BatchNorm2d(hidden_dim),
                    torch.nn.ReLU6(inplace=True),
                ]
            )

        layers.extend(
            [
                torch.nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                torch.nn.BatchNorm2d(hidden_dim),
                torch.nn.ReLU6(inplace=True),
                torch.nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileFaceDetector(torch.nn.Module):
    def __init__(self):
        super(MobileFaceDetector, self).__init__()

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU6(inplace=True),
        )

        self.blocks = torch.nn.Sequential(
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6),
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 128, 2, 6),
            InvertedResidual(128, 128, 1, 6),
        )

        self.spatial_pool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = torch.nn.Flatten()

        self.shared_features = torch.nn.Sequential(
            torch.nn.Linear(128 * 7 * 7, 256),
            torch.nn.ReLU6(inplace=True),
            torch.nn.Dropout(0.2),
        )

        self.reg_head = torch.nn.Linear(256, 8)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.spatial_pool(x)
        x = self.flatten(x)

        features = self.shared_features(x)
        coords = self.reg_head(features)

        return coords


TARGET_SIZE = 256


def letterbox_image(image):
    orig_w, orig_h = image.size
    scale = min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    padded = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    pad_left = (TARGET_SIZE - new_w) // 2
    pad_top = (TARGET_SIZE - new_h) // 2
    padded.paste(resized, (pad_left, pad_top))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    tensor = transform(padded)
    return tensor, scale, pad_left, pad_top


def extract_detection(outputs, orig_w, orig_h, scale, pad_left, pad_top):
    outputs = outputs.squeeze(0).cpu().numpy()
    x_norm, y_norm, w_norm, h_norm, lex_norm, ley_norm, rex_norm, rey_norm = outputs

    x_pad = x_norm * TARGET_SIZE
    y_pad = y_norm * TARGET_SIZE
    w_pad = w_norm * TARGET_SIZE
    h_pad = h_norm * TARGET_SIZE
    lex_pad = lex_norm * TARGET_SIZE
    ley_pad = ley_norm * TARGET_SIZE
    rex_pad = rex_norm * TARGET_SIZE
    rey_pad = rey_norm * TARGET_SIZE

    x_orig = int((x_pad - pad_left) / scale)
    y_orig = int((y_pad - pad_top) / scale)
    w_orig = int(w_pad / scale)
    h_orig = int(h_pad / scale)
    lex_orig = int((lex_pad - pad_left) / scale)
    ley_orig = int((ley_pad - pad_top) / scale)
    rex_orig = int((rex_pad - pad_left) / scale)
    rey_orig = int((rey_pad - pad_top) / scale)

    conf = 1.0
    return (
        x_orig,
        y_orig,
        w_orig,
        h_orig,
        conf,
        (lex_orig, ley_orig),
        (rex_orig, rey_orig),
    )


def load_model(model_path, device_type="cuda"):
    device = torch.device(device_type if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model, device


def main():
    model_path = Path(__file__).parent / "mobile_face_detector.pt"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
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

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()
