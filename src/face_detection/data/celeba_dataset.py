import csv
import math
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def default_annotation(img_width, img_height):
    """Return a default face region (centered 50% of image) when annotation is missing."""
    return {
        "x": img_width // 4,
        "y": img_height // 4,
        "w": img_width // 2,
        "h": img_height // 2,
        "left_eye": (img_width // 3, img_height // 2),
        "right_eye": (2 * img_width // 3, img_height // 2),
    }


class CelebADataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        start_idx=0,
        max_samples=None,
        target_size=224,
        augment_scale=False,
        augment_rotation=False,
        max_rotation_angle=30,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        self.augment_scale = augment_scale
        self.augment_rotation = augment_rotation
        self.max_rotation_angle = max_rotation_angle

        img_dir = self.data_dir / "img_celeba"
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        self.file_list = sorted([f for f in img_dir.glob("*.jpg")])
        end_idx = start_idx + max_samples if max_samples else len(self.file_list)
        self.file_list = self.file_list[start_idx:end_idx]

        bbox_eyes_file = self.data_dir / "bbox_and_eyes.csv"
        if not bbox_eyes_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {bbox_eyes_file}")

        self.annotations = {}
        with open(bbox_eyes_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.annotations[row["image_id"]] = {
                    "x": int(row["x_1"]),
                    "y": int(row["y_1"]),
                    "w": int(row["width"]),
                    "h": int(row["height"]),
                    "left_eye": (int(row["lefteye_x"]), int(row["lefteye_y"])),
                    "right_eye": (int(row["righteye_x"]), int(row["righteye_y"])),
                }

    def __len__(self):
        return len(self.file_list)

    def _rotate_point(self, x, y, angle, cx, cy, new_cx, new_cy):
        """Rotate point (x,y) about (cx,cy) by angle degrees and translate to new center.
        Uses -angle to match PIL's counter-clockwise rotation in Y-down coordinate system."""
        rad = math.radians(-angle)
        x_rel = x - cx
        y_rel = y - cy
        x_rot = x_rel * math.cos(rad) - y_rel * math.sin(rad)
        y_rot = x_rel * math.sin(rad) + y_rel * math.cos(rad)
        x_new = x_rot + new_cx
        y_new = y_rot + new_cy
        return x_new, y_new

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        img_id = img_path.name
        if img_id in self.annotations:
            ann = self.annotations[img_id].copy()
        else:
            ann = default_annotation(orig_w, orig_h)

        if self.augment_rotation:
            angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)

            rad = math.radians(angle)
            new_w = int(
                round(orig_w * abs(math.cos(rad)) + orig_h * abs(math.sin(rad)))
            )
            new_h = int(
                round(orig_w * abs(math.sin(rad)) + orig_h * abs(math.cos(rad)))
            )

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
            new_corners = []
            for x, y in corners:
                nx, ny = self._rotate_point(
                    x, y, angle, cx_orig, cy_orig, new_cx, new_cy
                )
                new_corners.append((nx, ny))

            xs = [p[0] for p in new_corners]
            ys = [p[1] for p in new_corners]

            new_x = max(0, min(xs))
            new_y = max(0, min(ys))
            max_x = min(new_w, max(xs))
            max_y = min(new_h, max(ys))

            new_w_bbox = max_x - new_x
            new_h_bbox = max_y - new_y

            new_left_eye = self._rotate_point(
                ann["left_eye"][0],
                ann["left_eye"][1],
                angle,
                cx_orig,
                cy_orig,
                new_cx,
                new_cy,
            )
            new_right_eye = self._rotate_point(
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

        if self.augment_scale:
            scale_factor = random.uniform(0.5, 1.0)
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

        scale = min(self.target_size / orig_w, self.target_size / orig_h)
        new_w2 = int(orig_w * scale)
        new_h2 = int(orig_h * scale)
        img_resized = img.resize((new_w2, new_h2), Image.BILINEAR)

        padded_img = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        pad_left = (self.target_size - new_w2) // 2
        pad_top = (self.target_size - new_h2) // 2
        padded_img.paste(img_resized, (pad_left, pad_top))

        if self.transform:
            padded_img = self.transform(padded_img)

        x = ann["x"] * scale + pad_left
        y = ann["y"] * scale + pad_top
        w = ann["w"] * scale
        h = ann["h"] * scale
        le_x = ann["left_eye"][0] * scale + pad_left
        le_y = ann["left_eye"][1] * scale + pad_top
        re_x = ann["right_eye"][0] * scale + pad_left
        re_y = ann["right_eye"][1] * scale + pad_top

        target = torch.tensor(
            [
                x / self.target_size,
                y / self.target_size,
                w / self.target_size,
                h / self.target_size,
                le_x / self.target_size,
                le_y / self.target_size,
                re_x / self.target_size,
                re_y / self.target_size,
            ],
            dtype=torch.float32,
        )

        return padded_img, target, 1
