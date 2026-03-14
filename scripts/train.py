#!/usr/bin/env python3
"""
Train MobileFaceDetector on CelebA dataset with letterboxing, scale augmentation,
and rotation augmentation.
"""

import argparse
import os
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from face_detection import (
    CelebADataset,
    COCONoHumanDataset,
    MixedDataset,
    MobileFaceDetector,
    train_model,
)


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
        default="weights/mobile_face_detector.pth",
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
    parser.add_argument(
        "--augment_rotation",
        action="store_true",
        help="Enable random rotation augmentation",
    )
    parser.add_argument(
        "--max_rotation_angle",
        type=float,
        default=30,
        help="Maximum rotation angle in degrees (default: 30)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="weights",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--train_only_head",
        action="store_true",
        help="Freeze backbone and only train the detection head",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/training.log",
        help="Path to log file for training metrics (default: logs/training.log)",
    )
    parser.add_argument(
        "--coco_dir",
        type=str,
        default=None,
        help="Path to COCO dataset root (e.g., Dataset/). If provided, mixes COCO images at 2:1 ratio",
    )
    parser.add_argument(
        "--coco_ratio",
        type=int,
        default=2,
        help="Number of COCO images to use per face image (default: 2)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    face_train_dataset = CelebADataset(
        args.data_dir,
        transform=transform,
        start_idx=0,
        max_samples=200000,
        target_size=args.target_size,
        augment_scale=True,
        augment_rotation=args.augment_rotation,
        max_rotation_angle=args.max_rotation_angle,
    )

    # Create training dataset - either mixed with COCO or face-only
    if args.coco_dir:
        coco_images_dir = os.path.join(args.coco_dir, "train2017")
        coco_ann_file = os.path.join(
            args.coco_dir, "annotations", "instances_train2017.json"
        )

        if os.path.exists(coco_images_dir) and os.path.exists(coco_ann_file):
            coco_dataset = COCONoHumanDataset(
                coco_images_dir,
                coco_ann_file,
                transform=transform,
                target_size=args.target_size,
            )
            train_dataset = MixedDataset(
                face_train_dataset,
                coco_dataset,
                ratio=args.coco_ratio,
            )
            print(f"COCO dataset: {len(coco_dataset)} images (no humans)")
            print(f"Mix ratio: {args.coco_ratio} COCO images per 1 face image")
        else:
            print(
                f"Warning: COCO paths not found: {coco_images_dir} or {coco_ann_file}"
            )
            print("Training with face dataset only")
            train_dataset = face_train_dataset
    else:
        train_dataset = face_train_dataset
    val_dataset = CelebADataset(
        args.data_dir,
        transform=transform,
        start_idx=200000,
        max_samples=20000,
        target_size=args.target_size,
        augment_scale=False,
        augment_rotation=False,
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

    if args.coco_dir and "coco_dataset" in locals():
        print(
            f"Train dataset: {len(train_dataset)} images ({len(face_train_dataset)} face + {len(coco_dataset)} COCO at {args.coco_ratio}:1 ratio)"
        )
    else:
        print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset:   {len(val_dataset)} images")

    model = MobileFaceDetector()

    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Loading pretrained weights from {args.pretrained}")
        if args.train_only_head:
            pretrained_state = torch.load(args.pretrained, map_location=device)
            backbone_keys = ["stem", "blocks", "spatial_pool"]
            backbone_state = {
                k: v
                for k, v in pretrained_state.items()
                if any(k.startswith(key) for key in backbone_keys)
            }
            model.load_state_dict(backbone_state, strict=False)
            print(f"Loaded backbone weights. Head layers will be trained from scratch.")
        else:
            model.load_state_dict(torch.load(args.pretrained, map_location=device))
    elif args.pretrained:
        print(
            f"Warning: Pretrained file {args.pretrained} not found. Training from scratch."
        )

    config = {
        "pretrained": args.pretrained if args.pretrained else "None",
        "train_only_head": args.train_only_head,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "target_iou": args.target_iou,
        "target_eye_acc": args.target_eye_acc,
        "device": device,
        "data_dir": args.data_dir,
        "target_size": args.target_size,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "augment_scale": True,
        "augment_rotation": args.augment_rotation,
        "max_rotation_angle": args.max_rotation_angle,
        "checkpoint_dir": args.checkpoint_dir,
        "log_file": args.log_file,
        "coco_dir": args.coco_dir if args.coco_dir else "None",
        "coco_ratio": args.coco_ratio if args.coco_dir else "N/A",
    }

    model = train_model(
        model,
        train_loader,
        val_loader,
        config=config,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Final model saved to {args.output}")


if __name__ == "__main__":
    main()
