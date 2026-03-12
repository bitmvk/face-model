#!/usr/bin/env python3
"""
Real-time camera stream face detection using ONNX MobileFaceDetector.
Shows FPS and frame processing time.
"""

import numpy as np
import cv2
import time
from pathlib import Path
from PIL import Image

from face_detection import TARGET_SIZE


def letterbox_image(image):
    orig_w, orig_h = image.size
    scale = min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    padded = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    pad_left = (TARGET_SIZE - new_w) // 2
    pad_top = (TARGET_SIZE - new_h) // 2
    padded.paste(resized, (pad_left, pad_top))

    img_np = np.array(padded).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)

    return img_np, scale, pad_left, pad_top


def extract_detection(outputs, orig_w, orig_h, scale, pad_left, pad_top):
    outputs = outputs.squeeze()
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


def load_model(model_path):
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    actual_provider = session.get_providers()[0]
    print(f"Using execution provider: {actual_provider}")

    return session


def main():
    model_path = Path(__file__).parent.parent / "weights" / "mobile_face_detector.onnx"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run: uv run python scripts/convert_to_onnx.py")
        return

    print(f"Loading ONNX model from {model_path}...")
    session = load_model(str(model_path))
    print("Model loaded")

    input_name = session.get_inputs()[0].name

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

        img_input, scale, pad_left, pad_top = letterbox_image(pil_image)

        outputs = session.run(None, {input_name: img_input})

        x, y, w, h, conf, left_eye, right_eye = extract_detection(
            outputs[0], orig_w, orig_h, scale, pad_left, pad_top
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

        cv2.imshow("Face Detection (ONNX)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()
