#!/usr/bin/env python3
"""
Train MobileFaceDetector on CelebA dataset with letterboxing and scale augmentation.
Fixed transform: annotation coordinates now correctly account for random scaling.
"""

import argparse
import csv
import os
import random
import time  # <-- added for elapsed time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# -------------------- Helper for default annotation --------------------
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


# -------------------- Model Definition --------------------
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))
        layers = []

        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileFaceDetector(nn.Module):
    def __init__(self):
        super(MobileFaceDetector, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.blocks = nn.Sequential(
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

        self.spatial_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()

        self.shared_features = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256), nn.ReLU6(inplace=True), nn.Dropout(0.2)
        )

        self.reg_head = nn.Linear(256, 8)

        # Optional confidence head (commented out)
        # self.conf_head = nn.Sequential(
        #     nn.Linear(256, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.spatial_pool(x)
        x = self.flatten(x)

        features = self.shared_features(x)
        coords = self.reg_head(features)
        return coords


# -------------------- Dataset --------------------
class CelebADataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        start_idx=0,
        max_samples=None,
        target_size=224,
        augment_scale=False,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        self.augment_scale = augment_scale

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

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Random scaling augmentation (simulate different face sizes)
        if self.augment_scale:
            scale_factor = random.uniform(0.5, 1.0)
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            img = img.resize((new_w, new_h), Image.BILINEAR)
        else:
            scale_factor = 1.0
            new_w, new_h = orig_w, orig_h

        # Letterbox to target_size
        scale = min(self.target_size / new_w, self.target_size / new_h)
        new_w2 = int(new_w * scale)
        new_h2 = int(new_h * scale)
        img_resized = img.resize((new_w2, new_h2), Image.BILINEAR)

        padded_img = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        pad_left = (self.target_size - new_w2) // 2
        pad_top = (self.target_size - new_h2) // 2
        padded_img.paste(img_resized, (pad_left, pad_top))

        if self.transform:
            padded_img = self.transform(padded_img)

        # Get annotation (or default)
        img_id = img_path.name
        if img_id in self.annotations:
            ann = self.annotations[img_id]
        else:
            ann = default_annotation(orig_w, orig_h)

        # Apply random scale to annotation coordinates
        x = ann["x"] * scale_factor
        y = ann["y"] * scale_factor
        w = ann["w"] * scale_factor
        h = ann["h"] * scale_factor
        le_x = ann["left_eye"][0] * scale_factor
        le_y = ann["left_eye"][1] * scale_factor
        re_x = ann["right_eye"][0] * scale_factor
        re_y = ann["right_eye"][1] * scale_factor

        # Apply letterbox scale and padding
        x_pad = x * scale + pad_left
        y_pad = y * scale + pad_top
        w_pad = w * scale
        h_pad = h * scale
        le_x_pad = le_x * scale + pad_left
        le_y_pad = le_y * scale + pad_top
        re_x_pad = re_x * scale + pad_left
        re_y_pad = re_y * scale + pad_top

        # Normalize to [0,1] relative to padded image size
        target = torch.tensor(
            [
                x_pad / self.target_size,
                y_pad / self.target_size,
                w_pad / self.target_size,
                h_pad / self.target_size,
                le_x_pad / self.target_size,
                le_y_pad / self.target_size,
                re_x_pad / self.target_size,
                re_y_pad / self.target_size,
            ],
            dtype=torch.float32,
        )

        metadata = (orig_w, orig_h, scale, pad_left, pad_top)
        return padded_img, target, metadata


# -------------------- Evaluation Metrics --------------------
def calculate_iou(pred_box, true_box):
    x1 = torch.max(pred_box[:, 0], true_box[:, 0])
    y1 = torch.max(pred_box[:, 1], true_box[:, 1])
    x2 = torch.min(pred_box[:, 0] + pred_box[:, 2], true_box[:, 0] + true_box[:, 2])
    y2 = torch.min(pred_box[:, 1] + pred_box[:, 3], true_box[:, 1] + true_box[:, 3])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = pred_box[:, 2] * pred_box[:, 3]
    true_area = true_box[:, 2] * true_box[:, 3]
    union = pred_area + true_area - intersection

    return intersection / (union + 1e-6)


def calculate_eye_accuracy(pred_eyes, true_eyes, threshold=0.03):
    """
    Returns:
        left_correct (int): number of samples with left eye within threshold
        right_correct (int): number of samples with right eye within threshold
    """
    left_eye_pred = pred_eyes[:, 0:2]
    left_eye_true = true_eyes[:, 0:2]
    right_eye_pred = pred_eyes[:, 2:4]
    right_eye_true = true_eyes[:, 2:4]

    left_dist = torch.norm(left_eye_pred - left_eye_true, dim=1)
    right_dist = torch.norm(right_eye_pred - right_eye_true, dim=1)

    left_correct = (left_dist < threshold).sum().item()
    right_correct = (right_dist < threshold).sum().item()
    return left_correct, right_correct


# -------------------- Training Loop --------------------
def train_model(
    model,
    train_loader,
    val_loader,
    target_iou=85.0,
    target_eye_acc=80.0,
    lr=0.001,
    device="cuda",
    start_epoch=0,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    criterion_reg = nn.SmoothL1Loss()
    landmark_weight = 5.0

    start_time = time.time()  # <-- start time for elapsed time calculation

    epoch = start_epoch
    while True:
        model.train()
        total_train_loss = 0
        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_bbox = criterion_reg(outputs[:, :4], targets[:, :4])
            loss_eyes = criterion_reg(outputs[:, 4:8], targets[:, 4:8])
            loss = loss_bbox + (loss_eyes * landmark_weight)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[{minutes:02d}:{seconds:02d}] Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        iou_sum = 0
        left_eye_correct = 0
        right_eye_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)

                loss_bbox = criterion_reg(outputs[:, :4], targets[:, :4])
                loss_eyes = criterion_reg(outputs[:, 4:8], targets[:, 4:8])
                loss = loss_bbox + (loss_eyes * landmark_weight)

                total_val_loss += loss.item()

                ious = calculate_iou(outputs[:, :4], targets[:, :4])
                iou_sum += ious.sum().item()

                left_correct, right_correct = calculate_eye_accuracy(
                    outputs[:, 4:8], targets[:, 4:8], threshold=0.01
                )
                left_eye_correct += left_correct
                right_eye_correct += right_correct

                total_samples += targets.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        mean_iou_pct = (iou_sum / total_samples) * 100
        left_eye_acc_pct = (left_eye_correct / total_samples) * 100
        right_eye_acc_pct = (right_eye_correct / total_samples) * 100

        scheduler.step(avg_val_loss)

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print(
            f"[{minutes:02d}:{seconds:02d}] Epoch {epoch + 1} Completed | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Mean BBox IoU: {mean_iou_pct:.2f}%% | "
            f"Left Eye Acc: {left_eye_acc_pct:.2f}%% | "
            f"Right Eye Acc: {right_eye_acc_pct:.2f}%%"
        )

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"mobile_face_detector_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Stop condition: both eyes meet target individually
        if mean_iou_pct >= target_iou and left_eye_acc_pct >= target_eye_acc and right_eye_acc_pct >= target_eye_acc:
            print(f"\nTarget metrics achieved. Stopping training.")
            break

        epoch += 1

    return model


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Train MobileFaceDetector on CelebA")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory (containing img_celeba and bbox_and_eyes.csv)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained weights (.pth file) to start from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mobile_face_detector.pth",
        help="Where to save the final model",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Input image size after letterboxing (default: 256)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--target_iou",
        type=float,
        default=80.0,
        help="Target IoU%% to stop (default: 80)",
    )
    parser.add_argument(
        "--target_eye_acc",
        type=float,
        default=80.0,
        help="Target eye accuracy%% to stop (default: 80)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum number of epochs (if not set, train until targets are met)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0, safe for notebooks)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Transform (no Resize here, done in dataset)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Datasets (using fixed splits from the notebook)
    train_dataset = CelebADataset(
        args.data_dir,
        transform=transform,
        start_idx=0,
        max_samples=200000,
        target_size=args.target_size,
        augment_scale=True,
    )
    val_dataset = CelebADataset(
        args.data_dir,
        transform=transform,
        start_idx=200000,
        max_samples=20000,
        target_size=args.target_size,
        augment_scale=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset:   {len(val_dataset)} images")

    # Create model
    model = MobileFaceDetector()

    # Load pretrained weights if provided
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained weights from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
    elif args.pretrained:
        print(
            f"Warning: Pretrained file {args.pretrained} not found. Training from scratch."
        )

    # Optionally freeze early layers (uncomment if desired)
    # for name, param in model.named_parameters():
    #     if 'stem' in name or 'blocks.0' in name:
    #         param.requires_grad = False

    # Train
    model = train_model(
        model,
        train_loader,
        val_loader,
        target_iou=args.target_iou,
        target_eye_acc=args.target_eye_acc,
        lr=args.lr,
        device=device,
    )

    # Save final model
    torch.save(model.state_dict(), args.output)
    print(f"Final model saved to {args.output}")


if __name__ == "__main__":
    main()