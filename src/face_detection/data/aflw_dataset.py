import math
import random
import sqlite3
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class AFLWDataset(Dataset):
    """
    AFLW face dataset with bounding box and eye landmark annotations.

    Reads from aflw.sqlite. For images with multiple faces, the largest face
    (by bounding box area) that has both eye center coordinates is used.
    Only samples with both LeftEyeCenter and RightEyeCenter annotations are included.

    Args:
        data_dir (str or Path): AFLW root directory (contains data/aflw.sqlite and data/flickr/).
        transform (callable, optional): Transform applied to the image.
        start_idx (int): Start index into the sorted sample list.
        max_samples (int, optional): Maximum number of samples to use.
        target_size (int): Output image size after letterboxing.
        augment_scale (bool): Enable random scale augmentation.
        augment_rotation (bool): Enable random rotation augmentation.
        max_rotation_angle (float): Maximum rotation angle in degrees.
    """

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

        db_path = self.data_dir / "data" / "aflw.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"AFLW database not found: {db_path}")

        self.images_dir = self.data_dir / "data" / "flickr"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"AFLW images directory not found: {self.images_dir}")

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT fi.filepath,
                   fr.x, fr.y, fr.w, fr.h,
                   fc_le.x, fc_le.y,
                   fc_re.x, fc_re.y,
                   fr.w * fr.h AS area
            FROM FaceImages fi
            JOIN Faces f ON f.file_id = fi.file_id AND f.db_id = fi.db_id
            JOIN FaceRect fr ON fr.face_id = f.face_id
            JOIN FeatureCoords fc_le ON fc_le.face_id = f.face_id AND fc_le.feature_id = 8
            JOIN FeatureCoords fc_re ON fc_re.face_id = f.face_id AND fc_re.feature_id = 11
            WHERE fr.w > 0 AND fr.h > 0
            ORDER BY fi.filepath, area DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        # One sample per image: keep the largest face
        seen = {}
        for row in rows:
            filepath = row[0]
            if filepath not in seen:
                seen[filepath] = row

        records = [seen[k] for k in sorted(seen.keys())]
        end_idx = start_idx + max_samples if max_samples else len(records)
        self.records = records[start_idx:end_idx]

    def __len__(self):
        return len(self.records)

    def _rotate_point(self, x, y, angle, cx, cy, new_cx, new_cy):
        rad = math.radians(-angle)
        x_rel, y_rel = x - cx, y - cy
        x_rot = x_rel * math.cos(rad) - y_rel * math.sin(rad)
        y_rot = x_rel * math.sin(rad) + y_rel * math.cos(rad)
        return x_rot + new_cx, y_rot + new_cy

    def __getitem__(self, idx):
        filepath, bx, by, bw, bh, le_x, le_y, re_x, re_y, _ = self.records[idx]

        img_path = self.images_dir / filepath
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        ann = {
            "x": max(0.0, float(bx)),
            "y": max(0.0, float(by)),
            "w": float(bw),
            "h": float(bh),
            "left_eye": (float(le_x), float(le_y)),
            "right_eye": (float(re_x), float(re_y)),
        }
        ann["w"] = min(ann["w"], orig_w - ann["x"])
        ann["h"] = min(ann["h"], orig_h - ann["y"])

        if self.augment_rotation:
            angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
            rad = math.radians(angle)
            new_w = int(round(orig_w * abs(math.cos(rad)) + orig_h * abs(math.sin(rad))))
            new_h = int(round(orig_w * abs(math.sin(rad)) + orig_h * abs(math.cos(rad))))
            img = img.rotate(angle, expand=True, fillcolor=0, resample=Image.BILINEAR)

            cx_orig, cy_orig = orig_w / 2.0, orig_h / 2.0
            new_cx, new_cy = new_w / 2.0, new_h / 2.0

            corners = [
                (ann["x"], ann["y"]),
                (ann["x"] + ann["w"], ann["y"]),
                (ann["x"], ann["y"] + ann["h"]),
                (ann["x"] + ann["w"], ann["y"] + ann["h"]),
            ]
            new_corners = [
                self._rotate_point(x, y, angle, cx_orig, cy_orig, new_cx, new_cy)
                for x, y in corners
            ]
            xs = [p[0] for p in new_corners]
            ys = [p[1] for p in new_corners]

            new_x = max(0, min(xs))
            new_y = max(0, min(ys))
            max_x = min(new_w, max(xs))
            max_y = min(new_h, max(ys))

            ann.update({
                "x": new_x,
                "y": new_y,
                "w": max_x - new_x,
                "h": max_y - new_y,
                "left_eye": self._rotate_point(
                    ann["left_eye"][0], ann["left_eye"][1],
                    angle, cx_orig, cy_orig, new_cx, new_cy
                ),
                "right_eye": self._rotate_point(
                    ann["right_eye"][0], ann["right_eye"][1],
                    angle, cx_orig, cy_orig, new_cx, new_cy
                ),
            })
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
        le_x_out = ann["left_eye"][0] * scale + pad_left
        le_y_out = ann["left_eye"][1] * scale + pad_top
        re_x_out = ann["right_eye"][0] * scale + pad_left
        re_y_out = ann["right_eye"][1] * scale + pad_top

        target = torch.tensor(
            [
                x / self.target_size,
                y / self.target_size,
                w / self.target_size,
                h / self.target_size,
                le_x_out / self.target_size,
                le_y_out / self.target_size,
                re_x_out / self.target_size,
                re_y_out / self.target_size,
            ],
            dtype=torch.float32,
        )

        return padded_img, target, 1
