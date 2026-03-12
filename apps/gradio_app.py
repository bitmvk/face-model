#!/usr/bin/env python3
"""
Gradio demo for MobileFaceDetector trained on CelebA.
Now uses letterboxing (256x256) to match training preprocessing.
"""

import torch
import numpy as np
import cv2
import gradio as gr
import pandas as pd
from pathlib import Path
from PIL import Image

from face_detection import (
    MobileFaceDetector,
    letterbox_image,
    extract_detection,
    TARGET_SIZE,
)


DATASET_DIR = Path(__file__).parent.parent / "Dataset"

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


model = None
device = None


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

    img_tensor, scale, pad_left, pad_top = letterbox_image(orig_img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        x, y, w, h, conf, left_eye, right_eye = extract_detection(
            outputs, orig_w, orig_h, scale, pad_left, pad_top
        )

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

        landmarks, bbox = get_image_data(image_id_str, dataset_version)
        orig_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig_img.size

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

        load_model_if_needed(model_path, device_type)

        img_tensor, scale, pad_left, pad_top = letterbox_image(orig_img)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            x, y, w, h, conf, left_eye, right_eye = extract_detection(
                outputs, orig_w, orig_h, scale, pad_left, pad_top
            )

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
                    value="weights/mobile_face_detector.pth", label="Model Path"
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
