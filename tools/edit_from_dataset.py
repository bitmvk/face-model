#!/usr/bin/env python3
"""Edit CelebA images with face detection annotations."""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


DATASET_DIR = Path(__file__).parent.parent / "Dataset"
IMAGE_DIR = DATASET_DIR / "img_celeba"
LANDMARKS_FILE = DATASET_DIR / "list_landmarks_align_celeba.txt"
BBOX_FILE = DATASET_DIR / "list_bbox_celeba.txt"


def load_dataset_data():
    landmarks_df = pd.read_csv(
        LANDMARKS_FILE,
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
    bbox_df = pd.read_csv(BBOX_FILE, delim_whitespace=True)
    return landmarks_df, bbox_df


def get_image_data(image_id, landmarks_df, bbox_df):
    if not image_id.endswith(".jpg"):
        image_id = f"{int(image_id):06d}.jpg"

    landmarks_row = landmarks_df[landmarks_df["image_id"] == image_id]
    bbox_row = bbox_df[bbox_df["image_id"] == image_id]

    if landmarks_row.empty:
        raise ValueError(f"Image {image_id} not found in landmarks dataset")
    if bbox_row.empty:
        raise ValueError(f"Image {image_id} not found in bbox dataset")

    landmarks = {
        "left_eye": (
            int(landmarks_row["lefteye_x"].values[0]),
            int(landmarks_row["lefteye_y"].values[0]),
        ),
        "right_eye": (
            int(landmarks_row["righteye_x"].values[0]),
            int(landmarks_row["righteye_y"].values[0]),
        ),
        "nose": (
            int(landmarks_row["nose_x"].values[0]),
            int(landmarks_row["nose_y"].values[0]),
        ),
        "left_mouth": (
            int(landmarks_row["leftmouth_x"].values[0]),
            int(landmarks_row["leftmouth_y"].values[0]),
        ),
        "right_mouth": (
            int(landmarks_row["rightmouth_x"].values[0]),
            int(landmarks_row["rightmouth_y"].values[0]),
        ),
    }

    bbox = {
        "x": int(bbox_row["x_1"].values[0]),
        "y": int(bbox_row["y_1"].values[0]),
        "width": int(bbox_row["width"].values[0]),
        "height": int(bbox_row["height"].values[0]),
    }

    return landmarks, bbox


def edit_image(
    image_path,
    landmarks,
    bbox,
    show_bbox=True,
    show_left_eye=True,
    show_right_eye=True,
    line_thickness=3,
):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if show_bbox:
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), line_thickness)

    if show_left_eye:
        cv2.circle(img, landmarks["left_eye"], line_thickness * 2, (255, 0, 0), -1)

    if show_right_eye:
        cv2.circle(img, landmarks["right_eye"], line_thickness * 2, (0, 0, 255), -1)

    return img


def process_image_id(
    image_id,
    show_bbox=True,
    show_left_eye=True,
    show_right_eye=True,
    line_thickness=3,
    output_path=None,
):
    landmarks_df, bbox_df = load_dataset_data()

    if not image_id.endswith(".jpg"):
        image_id = f"{int(image_id):06d}.jpg"

    image_path = IMAGE_DIR / image_id
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    landmarks, bbox = get_image_data(image_id, landmarks_df, bbox_df)

    edited_img = edit_image(
        image_path,
        landmarks,
        bbox,
        show_bbox=show_bbox,
        show_left_eye=show_left_eye,
        show_right_eye=show_right_eye,
        line_thickness=line_thickness,
    )

    if output_path:
        cv2.imwrite(str(output_path), cv2.cvtColor(edited_img, cv2.COLOR_RGB2BGR))
        print(f"Saved edited image to: {output_path}")
    else:
        return edited_img

    return edited_img


def main():
    parser = argparse.ArgumentParser(
        description="Edit CelebA image with face detection annotations"
    )
    parser.add_argument("image_id", help="Image ID (e.g., '000001' or '000001.jpg')")
    parser.add_argument("--no-bbox", action="store_true", help="Hide bounding box")
    parser.add_argument(
        "--no-left-eye", action="store_true", help="Hide left eye marker"
    )
    parser.add_argument(
        "--no-right-eye", action="store_true", help="Hide right eye marker"
    )
    parser.add_argument("-t", "--thickness", type=int, default=3, help="Line thickness")
    parser.add_argument("-o", "--output", type=str, help="Output file path")

    args = parser.parse_args()

    try:
        process_image_id(
            args.image_id,
            show_bbox=not args.no_bbox,
            show_left_eye=not args.no_left_eye,
            show_right_eye=not args.no_right_eye,
            line_thickness=args.thickness,
            output_path=args.output,
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
