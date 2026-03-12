#!/usr/bin/env python3
"""
Gradio demo for verifying CelebA dataset augmentations.
Visualizes rotation, scaling, and letterboxing transformations.
"""

import math
import cv2
import gradio as gr
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image


DATASET_DIR = Path(__file__).parent.parent / "Dataset"
FULL_IMAGE_DIR = DATASET_DIR / "img_celeba"
FULL_ANNOTATIONS_FILE = DATASET_DIR / "bbox_and_eyes.csv"

annotations_df = None


def load_annotations():
    global annotations_df
    if annotations_df is None:
        if not FULL_ANNOTATIONS_FILE.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {FULL_ANNOTATIONS_FILE}"
            )
        annotations_df = pd.read_csv(FULL_ANNOTATIONS_FILE)
        annotations_df.set_index("image_id", inplace=True)
    return annotations_df


def default_annotation(img_width, img_height):
    return {
        "x": img_width // 4,
        "y": img_height // 4,
        "w": img_width // 2,
        "h": img_height // 2,
        "left_eye": (img_width // 3, img_height // 2),
        "right_eye": (2 * img_width // 3, img_height // 2),
    }


def _rotate_point(x, y, angle, cx, cy, new_cx, new_cy):
    rad = math.radians(-angle)
    x_rel = x - cx
    y_rel = y - cy
    x_rot = x_rel * math.cos(rad) - y_rel * math.sin(rad)
    y_rot = x_rel * math.sin(rad) + y_rel * math.cos(rad)
    return x_rot + new_cx, y_rot + new_cy


def process_dataset_image(
    image_id,
    enable_rotation,
    rotation_angle,
    enable_scale,
    scale_factor,
    target_size,
    show_bbox,
    show_eyes,
    line_thickness,
):
    if not image_id:
        return None, "Provide an image ID."

    image_id_str = image_id if image_id.endswith(".jpg") else f"{int(image_id):06d}.jpg"
    img_path = FULL_IMAGE_DIR / image_id_str

    if not img_path.exists():
        return None, f"Image not found: {img_path}"

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    df = load_annotations()
    if image_id_str in df.index:
        row = df.loc[image_id_str]
        ann = {
            "x": int(row["x_1"]),
            "y": int(row["y_1"]),
            "w": int(row["width"]),
            "h": int(row["height"]),
            "left_eye": (int(row["lefteye_x"]), int(row["lefteye_y"])),
            "right_eye": (int(row["righteye_x"]), int(row["righteye_y"])),
        }
    else:
        ann = default_annotation(orig_w, orig_h)

    if enable_rotation and rotation_angle != 0:
        angle = rotation_angle
        rad = math.radians(angle)
        new_w = int(round(orig_w * abs(math.cos(rad)) + orig_h * abs(math.sin(rad))))
        new_h = int(round(orig_w * abs(math.sin(rad)) + orig_h * abs(math.cos(rad))))

        img = img.rotate(angle, expand=True, fillcolor=0, resample=Image.BILINEAR)

        cx_orig = orig_w / 2.0
        cy_orig = orig_h / 2.0
        new_cx = new_w / 2.0
        new_cy = new_h / 2.0

        corners = [
            (ann["x"], ann["y"]),
            (ann["x"] + ann["w"], ann["y"]),
            (ann["x"], ann["y"] + ann["h"]),
            (ann["x"] + ann["w"], ann["y"] + ann["h"]),
        ]

        new_corners = [
            _rotate_point(x, y, angle, cx_orig, cy_orig, new_cx, new_cy)
            for (x, y) in corners
        ]

        xs = [p[0] for p in new_corners]
        ys = [p[1] for p in new_corners]

        new_x = max(0, min(xs))
        new_y = max(0, min(ys))
        max_x = min(new_w, max(xs))
        max_y = min(new_h, max(ys))

        new_w_bbox = max_x - new_x
        new_h_bbox = max_y - new_y

        new_left_eye = _rotate_point(
            ann["left_eye"][0],
            ann["left_eye"][1],
            angle,
            cx_orig,
            cy_orig,
            new_cx,
            new_cy,
        )
        new_right_eye = _rotate_point(
            ann["right_eye"][0],
            ann["right_eye"][1],
            angle,
            cx_orig,
            cy_orig,
            new_cx,
            new_cy,
        )

        ann.update(
            {
                "x": new_x,
                "y": new_y,
                "w": new_w_bbox,
                "h": new_h_bbox,
                "left_eye": new_left_eye,
                "right_eye": new_right_eye,
            }
        )
        orig_w, orig_h = new_w, new_h

    if enable_scale and scale_factor != 1.0:
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        ann["x"] *= scale_factor
        ann["y"] *= scale_factor
        ann["w"] *= scale_factor
        ann["h"] *= scale_factor
        ann["left_eye"] = (
            ann["left_eye"][0] * scale_factor,
            ann["left_eye"][1] * scale_factor,
        )
        ann["right_eye"] = (
            ann["right_eye"][0] * scale_factor,
            ann["right_eye"][1] * scale_factor,
        )

        orig_w, orig_h = new_w, new_h

    scale = min(target_size / orig_w, target_size / orig_h)
    new_w2 = int(orig_w * scale)
    new_h2 = int(orig_h * scale)
    img_resized = img.resize((new_w2, new_h2), Image.BILINEAR)

    padded_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_left = (target_size - new_w2) // 2
    pad_top = (target_size - new_h2) // 2
    padded_img.paste(img_resized, (pad_left, pad_top))

    final_x = int(ann["x"] * scale + pad_left)
    final_y = int(ann["y"] * scale + pad_top)
    final_w = int(ann["w"] * scale)
    final_h = int(ann["h"] * scale)
    final_lex = int(ann["left_eye"][0] * scale + pad_left)
    final_ley = int(ann["left_eye"][1] * scale + pad_top)
    final_rex = int(ann["right_eye"][0] * scale + pad_left)
    final_rey = int(ann["right_eye"][1] * scale + pad_top)

    out_np = np.array(padded_img)

    if show_bbox:
        cv2.rectangle(
            out_np,
            (final_x, final_y),
            (final_x + final_w, final_y + final_h),
            (0, 255, 0),
            line_thickness,
        )
    if show_eyes:
        cv2.circle(out_np, (final_lex, final_ley), line_thickness * 2, (255, 0, 0), -1)
        cv2.circle(out_np, (final_rex, final_rey), line_thickness * 2, (0, 0, 255), -1)

    info = f"Final Dimensions: {target_size}x{target_size}\n"
    info += f"BBox: ({final_x}, {final_y}, {final_w}, {final_h})\n"
    info += f"Left Eye: ({final_lex}, {final_ley})\n"
    info += f"Right Eye: ({final_rex}, {final_rey})"

    return out_np, info


with gr.Blocks(title="Dataset Augmentation Verifier") as demo:
    gr.Markdown("# Dataset Transformation Pipeline Verifier")

    with gr.Row():
        with gr.Column():
            image_id_input = gr.Textbox(label="Image ID (1-202599)", value="000001")

            with gr.Accordion("Augmentation Parameters", open=True):
                target_size = gr.Slider(
                    64, 512, value=256, step=32, label="Target Size (Letterbox)"
                )

                enable_rotation = gr.Checkbox(value=False, label="Apply Rotation")
                rotation_angle = gr.Slider(
                    -45, 45, value=0, step=1, label="Rotation Angle (Degrees)"
                )

                enable_scale = gr.Checkbox(value=False, label="Apply Scale")
                scale_factor = gr.Slider(
                    0.5, 1.5, value=1.0, step=0.05, label="Scale Factor"
                )

            with gr.Accordion("Rendering Options", open=True):
                show_bbox = gr.Checkbox(value=True, label="Draw Bounding Box")
                show_eyes = gr.Checkbox(
                    value=True, label="Draw Eyes (Red=Left, Blue=Right)"
                )
                line_thickness = gr.Slider(
                    1, 10, value=2, step=1, label="Line Thickness"
                )

            run_btn = gr.Button("Process Image", variant="primary")

        with gr.Column():
            preview_output = gr.Image(type="numpy", label="Augmented Tensor Preview")
            info_output = gr.Textbox(label="Final Coordinates")

    run_btn.click(
        fn=process_dataset_image,
        inputs=[
            image_id_input,
            enable_rotation,
            rotation_angle,
            enable_scale,
            scale_factor,
            target_size,
            show_bbox,
            show_eyes,
            line_thickness,
        ],
        outputs=[preview_output, info_output],
    )

if __name__ == "__main__":
    demo.launch()
