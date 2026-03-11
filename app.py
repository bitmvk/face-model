#!/usr/bin/env python3
"""
Gradio demo for MobileFaceDetector trained on CelebA.
Now uses letterboxing (256x256) to match training preprocessing.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import gradio as gr
import pandas as pd
from pathlib import Path

# -------------------- Dataset Paths --------------------
DATASET_DIR = Path(__file__).parent / "Dataset"

FULL_IMAGE_DIR = DATASET_DIR / "img_celeba"
FULL_ANNOTATIONS_FILE = DATASET_DIR / "bbox_and_eyes.csv"

ALIGNED_IMAGE_DIR = DATASET_DIR / "img_align_celeba_png"
ALIGNED_LANDMARKS_FILE = DATASET_DIR / "list_landmarks_align_celeba.txt"

full_annotations_df = None
aligned_landmarks_df = None


def load_full_annotations():
    global full_annotations_df
    if full_annotations_df is None:
        full_annotations_df = pd.read_csv(FULL_ANNOTATIONS_FILE)
    return full_annotations_df


def load_aligned_landmarks():
    global aligned_landmarks_df
    if aligned_landmarks_df is None:
        aligned_landmarks_df = pd.read_csv(
            ALIGNED_LANDMARKS_FILE,
            sep=r"\s+",
            skiprows=2,
            header=None,
            names=[
                "image_id",
                "lefteye_x",
                "lefteye_y",
                "righteye_x",
                "righteye_y",
                "nose_x",
                "nose_y",
                "leftmouth_x",
                "leftmouth_y",
                "rightmouth_x",
                "rightmouth_y",
            ],
        )
    return aligned_landmarks_df


def get_image_data_full(image_id):
    df = load_full_annotations()

    row = df[df["image_id"] == image_id]
    if row.empty:
        raise ValueError(f"Image {image_id} not found in full dataset annotations")

    landmarks = {
        "left_eye": (int(row["lefteye_x"].values[0]), int(row["lefteye_y"].values[0])),
        "right_eye": (
            int(row["righteye_x"].values[0]),
            int(row["righteye_y"].values[0]),
        ),
    }

    bbox = {
        "x": int(row["x_1"].values[0]),
        "y": int(row["y_1"].values[0]),
        "width": int(row["width"].values[0]),
        "height": int(row["height"].values[0]),
    }

    return landmarks, bbox


def get_image_data_aligned(image_id):
    df = load_aligned_landmarks()

    if image_id.endswith(".jpg"):
        png_id = image_id.replace(".jpg", ".png")
    else:
        png_id = image_id
        image_id = image_id.replace(".png", ".jpg")

    row = df[df["image_id"] == png_id]
    if row.empty:
        raise ValueError(f"Image {png_id} not found in aligned dataset landmarks")

    landmarks = {
        "left_eye": (int(row["lefteye_x"].values[0]), int(row["lefteye_y"].values[0])),
        "right_eye": (
            int(row["righteye_x"].values[0]),
            int(row["righteye_y"].values[0]),
        ),
    }

    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    eye_dist = abs(right_eye[0] - left_eye[0])
    eye_center_x = (left_eye[0] + right_eye[0]) // 2
    eye_center_y = (left_eye[1] + right_eye[1]) // 2

    face_width = int(eye_dist * 2.5)
    face_height = int(eye_dist * 3.0)

    bbox = {
        "x": max(0, eye_center_x - face_width // 2),
        "y": max(0, eye_center_y - face_height // 3),
        "width": face_width,
        "height": face_height,
    }

    return landmarks, bbox


def get_image_data(image_id, dataset_version):
    if dataset_version == "Full":
        return get_image_data_full(image_id)
    else:
        return get_image_data_aligned(image_id)


def get_image_path(image_id, dataset_version):
    if dataset_version == "Full":
        if not image_id.endswith(".jpg"):
            image_id = f"{int(image_id):06d}.jpg"
        return FULL_IMAGE_DIR / image_id
    else:
        if not image_id.endswith(".png"):
            if image_id.endswith(".jpg"):
                image_id = image_id.replace(".jpg", ".png")
            else:
                image_id = f"{int(image_id):06d}.png"
        return ALIGNED_IMAGE_DIR / image_id


# -------------------- Model Definition (same as training) --------------------
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


model = None
device = None

# -------------------- Inference helpers --------------------
TARGET_SIZE = 256  # Must match training


def letterbox_image(image):
    """
    Resize image with letterboxing to TARGET_SIZE and return tensor + parameters.
    """
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
    """
    Convert model outputs (normalized in padded image) to original image coordinates.
    """
    outputs = outputs.squeeze(0).cpu().numpy()
    x_norm, y_norm, w_norm, h_norm, lex_norm, ley_norm, rex_norm, rey_norm = outputs

    # Padded image coordinates
    x_pad = x_norm * TARGET_SIZE
    y_pad = y_norm * TARGET_SIZE
    w_pad = w_norm * TARGET_SIZE
    h_pad = h_norm * TARGET_SIZE
    lex_pad = lex_norm * TARGET_SIZE
    ley_pad = ley_norm * TARGET_SIZE
    rex_pad = rex_norm * TARGET_SIZE
    rey_pad = rey_norm * TARGET_SIZE

    # Reverse letterbox
    x_orig = int((x_pad - pad_left) / scale)
    y_orig = int((y_pad - pad_top) / scale)
    w_orig = int(w_pad / scale)
    h_orig = int(h_pad / scale)
    lex_orig = int((lex_pad - pad_left) / scale)
    ley_orig = int((ley_pad - pad_top) / scale)
    rex_orig = int((rex_pad - pad_left) / scale)
    rey_orig = int((rey_pad - pad_top) / scale)

    conf = 1.0  # confidence not predicted
    return (
        x_orig,
        y_orig,
        w_orig,
        h_orig,
        conf,
        (lex_orig, ley_orig),
        (rex_orig, rey_orig),
    )


def load_model_if_needed(model_path, device_type):
    global model, device
    if model is None or str(model_path) != getattr(model, "_model_path", None):
        device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        model = MobileFaceDetector()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model._model_path = str(model_path)
        model.to(device)
        model.eval()
    return model, device


# -------------------- Inference functions --------------------
def run_inference_from_image(
    image,
    model_path,
    device_type,
    show_bbox,
    show_left_eye,
    show_right_eye,
    line_thickness,
):
    if image is None:
        return None, "Please upload an image"

    load_model_if_needed(model_path, device_type)

    orig_img = Image.fromarray(image).convert("RGB")
    orig_w, orig_h = orig_img.size

    # Letterbox and run inference
    img_tensor, scale, pad_left, pad_top = letterbox_image(orig_img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        x, y, w, h, conf, left_eye, right_eye = extract_detection(
            outputs, orig_w, orig_h, scale, pad_left, pad_top
        )

    # Draw results
    img_np = np.array(orig_img)
    if show_bbox:
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), line_thickness)
    if show_left_eye:
        cv2.circle(img_np, left_eye, line_thickness * 2, (255, 0, 0), -1)
    if show_right_eye:
        cv2.circle(img_np, right_eye, line_thickness * 2, (0, 0, 255), -1)

    info = f"BBox: ({x}, {y}, {w}, {h}) | Left Eye: {left_eye} | Right Eye: {right_eye} | Conf: {conf:.4f}"
    return Image.fromarray(img_np), info


def run_inference_from_id(
    image_id,
    model_path,
    device_type,
    dataset_version,
    show_bbox,
    show_left_eye,
    show_right_eye,
    line_thickness,
):
    if not image_id:
        return None, None, "Please provide an image ID"

    try:
        # Resolve image ID to file path
        if dataset_version == "Full":
            image_id_str = (
                image_id if image_id.endswith(".jpg") else f"{int(image_id):06d}.jpg"
            )
        else:
            image_id_str = (
                image_id if image_id.endswith(".png") else f"{int(image_id):06d}.png"
            )

        image_path = get_image_path(image_id_str, dataset_version)
        if not image_path.exists():
            return None, None, f"Image not found: {image_path}"

        # Load ground truth annotations
        landmarks, bbox = get_image_data(image_id_str, dataset_version)
        orig_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig_img.size

        # Draw ground truth
        img_np_gt = np.array(orig_img)
        if show_bbox:
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            cv2.rectangle(
                img_np_gt, (x, y), (x + w, y + h), (0, 255, 0), line_thickness
            )
        if show_left_eye:
            cv2.circle(
                img_np_gt, landmarks["left_eye"], line_thickness * 2, (255, 0, 0), -1
            )
        if show_right_eye:
            cv2.circle(
                img_np_gt, landmarks["right_eye"], line_thickness * 2, (0, 0, 255), -1
            )
        dataset_output = Image.fromarray(img_np_gt)
        dataset_info = f"Dataset ({dataset_version}): BBox: ({bbox['x']}, {bbox['y']}, {bbox['width']}, {bbox['height']}) | Left Eye: {landmarks['left_eye']} | Right Eye: {landmarks['right_eye']}"

        # Load model
        load_model_if_needed(model_path, device_type)

        # Letterbox and run inference
        img_tensor, scale, pad_left, pad_top = letterbox_image(orig_img)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            x, y, w, h, conf, left_eye, right_eye = extract_detection(
                outputs, orig_w, orig_h, scale, pad_left, pad_top
            )

        # Draw inference
        img_np_infer = np.array(orig_img)
        if show_bbox:
            cv2.rectangle(
                img_np_infer, (x, y), (x + w, y + h), (0, 255, 0), line_thickness
            )
        if show_left_eye:
            cv2.circle(img_np_infer, left_eye, line_thickness * 2, (255, 0, 0), -1)
        if show_right_eye:
            cv2.circle(img_np_infer, right_eye, line_thickness * 2, (0, 0, 255), -1)

        infer_output = Image.fromarray(img_np_infer)
        infer_info = f"Inferred BBox: ({x}, {y}, {w}, {h}) | Left Eye: {left_eye} | Right Eye: {right_eye} | Conf: {conf:.4f}"
        return infer_output, dataset_output, f"{infer_info}\n{dataset_info}"

    except Exception as e:
        return None, None, f"Error: {str(e)}"


def run_inference(
    image_id,
    image,
    model_path,
    device_type,
    dataset_version,
    show_bbox,
    show_left_eye,
    show_right_eye,
    line_thickness,
):
    if image_id and str(image_id).strip():
        return run_inference_from_id(
            image_id,
            model_path,
            device_type,
            dataset_version,
            show_bbox,
            show_left_eye,
            show_right_eye,
            line_thickness,
        )
    elif image is not None:
        inferred, info = run_inference_from_image(
            image,
            model_path,
            device_type,
            show_bbox,
            show_left_eye,
            show_right_eye,
            line_thickness,
        )
        return inferred, None, info
    else:
        return None, None, "Please provide an image ID or upload an image"


# -------------------- Gradio Interface --------------------
with gr.Blocks(title="Face Detection Demo") as demo:
    gr.Markdown("# Face Detection Inference")
    gr.Markdown(
        "Provide an image ID (1-202599) or upload an image. Image ID takes precedence."
    )

    with gr.Row():
        with gr.Column():
            image_id_input = gr.Textbox(
                label="Image ID (1-202599)", placeholder="e.g., 1 or 000001"
            )
            image_input = gr.Image(type="numpy", label="Or Upload Image")

            with gr.Accordion("Parameters", open=True):
                dataset_version = gr.Radio(
                    ["Full", "Aligned"],
                    value="Full",
                    label="Dataset Version",
                    info="Full: uses bbox_and_eyes.csv (actual bbox) | Aligned: uses list_landmarks_align_celeba.txt (calculated bbox)",
                )
                model_path = gr.Textbox(
                    value="mobile_face_detector-2.pth", label="Model Path"
                )
                device_type = gr.Radio(["cuda", "cpu"], value="cuda", label="Device")
                show_bbox = gr.Checkbox(value=True, label="Show Bounding Box")
                show_left_eye = gr.Checkbox(value=True, label="Show Left Eye (Red)")
                show_right_eye = gr.Checkbox(value=True, label="Show Right Eye (Blue)")
                line_thickness = gr.Slider(
                    1, 10, value=3, step=1, label="Line Thickness"
                )

            run_btn = gr.Button("Detect Face", variant="primary")

        with gr.Column():
            infer_output = gr.Image(type="pil", label="Inferred Output")
            dataset_output = gr.Image(type="pil", label="Dataset Actual Output")
            info_output = gr.Textbox(label="Detection Info")

    run_btn.click(
        fn=run_inference,
        inputs=[
            image_id_input,
            image_input,
            model_path,
            device_type,
            dataset_version,
            show_bbox,
            show_left_eye,
            show_right_eye,
            line_thickness,
        ],
        outputs=[infer_output, dataset_output, info_output],
    )

    gr.Examples(
        examples=[],
        inputs=image_id_input,
    )


if __name__ == "__main__":
    demo.launch()
