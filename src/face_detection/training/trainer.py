import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    left_eye_pred = pred_eyes[:, 0:2]
    left_eye_true = true_eyes[:, 0:2]
    right_eye_pred = pred_eyes[:, 2:4]
    right_eye_true = true_eyes[:, 2:4]

    left_dist = torch.norm(left_eye_pred - left_eye_true, dim=1)
    right_dist = torch.norm(right_eye_pred - right_eye_true, dim=1)

    left_correct = (left_dist < threshold).sum().item()
    right_correct = (right_dist < threshold).sum().item()
    return left_correct, right_correct


def _write_log_header(log_file, config):
    with open(log_file, "a") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: MobileFaceDetector\n")
        f.write(f"Pretrained: {config.get('pretrained', 'None')}\n")
        f.write(f"Train Only Head: {config.get('train_only_head', False)}\n")
        f.write(f"Learning Rate: {config.get('lr', 0.001)}\n")
        f.write(f"Batch Size: {config.get('batch_size', 32)}\n")
        f.write(f"Target IoU: {config.get('target_iou', 80.0)}%\n")
        f.write(f"Target Eye Accuracy: {config.get('target_eye_acc', 80.0)}%\n")
        f.write(f"Landmark Weight: {config.get('landmark_weight', 5.0)}\n")
        f.write(f"Device: {config.get('device', 'cuda')}\n")
        f.write(f"Dataset: {config.get('data_dir', 'N/A')}\n")
        f.write(f"Target Size: {config.get('target_size', 256)}\n")
        f.write(f"Train Samples: {config.get('train_samples', 'N/A')}\n")
        f.write(f"Val Samples: {config.get('val_samples', 'N/A')}\n")
        if config.get("coco_dir") and config["coco_dir"] != "None":
            f.write(f"COCO Dataset: {config['coco_dir']}\n")
            f.write(f"COCO Ratio: {config.get('coco_ratio', 'N/A')}:1\n")
        f.write("-" * 80 + "\n")
        f.write("Augmentation Settings:\n")
        f.write(f"  Scale Augmentation: {config.get('augment_scale', True)}\n")
        f.write(f"  Rotation Augmentation: {config.get('augment_rotation', False)}\n")
        f.write(
            f"  Max Rotation Angle: {config.get('max_rotation_angle', 30)} degrees\n"
        )
        f.write("=" * 80 + "\n\n")


def _write_log_epoch(
    log_file, epoch, epoch_duration, train_loss, val_loss, iou, left_acc, right_acc, lr
):
    with open(log_file, "a") as f:
        f.write(
            f"[Epoch {epoch:03d}] "
            f"Duration: {epoch_duration:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"IoU: {iou:.2f}% | "
            f"Left Eye Acc: {left_acc:.2f}% | "
            f"Right Eye Acc: {right_acc:.2f}% | "
            f"LR: {lr:.6f}\n"
        )


def _write_log_footer(log_file, config, final_epoch, total_duration, final_metrics):
    with open(log_file, "a") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"Training Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        f.write(f"Total Duration: {hours}h {minutes}m {seconds}s\n")
        f.write(f"Final Epoch: {final_epoch}\n")
        f.write(f"Final Train Loss: {final_metrics['train_loss']:.4f}\n")
        f.write(f"Final Val Loss: {final_metrics['val_loss']:.4f}\n")
        f.write(f"Final IoU: {final_metrics['iou']:.2f}%\n")
        f.write(f"Final Left Eye Acc: {final_metrics['left_eye_acc']:.2f}%\n")
        f.write(f"Final Right Eye Acc: {final_metrics['right_eye_acc']:.2f}%\n")
        f.write(f"Final LR: {final_metrics['lr']:.6f}\n")
        f.write("=" * 80 + "\n")


def train_model(
    model,
    train_loader,
    val_loader,
    config=None,
):
    if config is None:
        config = {}

    target_iou = config.get("target_iou", 85.0)
    target_eye_acc = config.get("target_eye_acc", 80.0)
    lr = config.get("lr", 0.001)
    device = config.get("device", "cuda")
    start_epoch = config.get("start_epoch", 0)
    checkpoint_dir = config.get("checkpoint_dir", ".")
    checkpoint_interval = config.get("checkpoint_interval", 10)
    train_only_head = config.get("train_only_head", False)
    log_file = config.get("log_file", None)
    landmark_weight = config.get("landmark_weight", 5.0)

    model = model.to(device)

    head_layers = [model.shared_features, model.reg_head, model.conf_head]
    if train_only_head:
        for param in model.parameters():
            param.requires_grad = False
        for layer in head_layers:
            for param in layer.parameters():
                param.requires_grad = True
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    criterion_bbox = nn.SmoothL1Loss(reduction="none")
    criterion_eyes = nn.SmoothL1Loss(reduction="none")

    if log_file:
        os.makedirs(
            os.path.dirname(log_file) if os.path.dirname(log_file) else ".",
            exist_ok=True,
        )
        _write_log_header(log_file, config)

    start_time = time.time()
    final_metrics = {}

    epoch = start_epoch
    while True:
        epoch_start = time.time()
        model.train()
        total_train_loss = 0
        total_conf_loss = 0
        total_coord_loss = 0

        for batch_idx, (images, targets, has_face) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            has_face = has_face.to(device).float()

            optimizer.zero_grad()

            coords_pred, conf_logits = model(images)

            conf_loss = F.binary_cross_entropy_with_logits(
                conf_logits.squeeze(1), has_face
            )

            mask = has_face.unsqueeze(1)

            # Only compute coord loss and backprop through reg_head when faces exist
            if mask.sum() > 0:
                bbox_loss_element = criterion_bbox(coords_pred[:, :4], targets[:, :4])
                bbox_loss = (bbox_loss_element * mask).sum() / (mask.sum() * 4)

                eyes_loss_element = criterion_eyes(coords_pred[:, 4:8], targets[:, 4:8])
                eyes_loss = (eyes_loss_element * mask).sum() / (mask.sum() * 4)

                coord_loss = bbox_loss + landmark_weight * eyes_loss
                loss = conf_loss + coord_loss
            else:
                # No faces in batch: only train conf_head
                bbox_loss = torch.tensor(0.0, device=device)
                eyes_loss = torch.tensor(0.0, device=device)
                coord_loss = torch.tensor(0.0, device=device)
                loss = conf_loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_conf_loss += conf_loss.item()
            total_coord_loss += coord_loss.item()

            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[{minutes:02d}:{seconds:02d}] Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f} (conf: {conf_loss.item():.4f}, coord: {coord_loss.item():.4f}), "
                    f"LR: {current_lr:.6f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)
        avg_conf_loss = total_conf_loss / len(train_loader)
        avg_coord_loss = total_coord_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        total_val_conf_loss = 0
        total_val_coord_loss = 0
        iou_sum = 0
        left_eye_correct = 0
        right_eye_correct = 0
        total_samples = 0
        total_face_samples = 0

        with torch.no_grad():
            for images, targets, has_face in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                has_face = has_face.to(device).float()

                coords_pred, conf_logits = model(images)

                conf_loss = F.binary_cross_entropy_with_logits(
                    conf_logits.squeeze(1), has_face
                )
                total_val_conf_loss += conf_loss.item()

                mask = has_face.unsqueeze(1)

                if mask.sum() > 0:
                    bbox_loss_element = criterion_bbox(
                        coords_pred[:, :4], targets[:, :4]
                    )
                    bbox_loss = (bbox_loss_element * mask).sum() / (mask.sum() * 4)
                    eyes_loss_element = criterion_eyes(
                        coords_pred[:, 4:8], targets[:, 4:8]
                    )
                    eyes_loss = (eyes_loss_element * mask).sum() / (mask.sum() * 4)
                    coord_loss = bbox_loss + landmark_weight * eyes_loss
                else:
                    bbox_loss = torch.tensor(0.0, device=device)
                    eyes_loss = torch.tensor(0.0, device=device)
                    coord_loss = torch.tensor(0.0, device=device)

                total_val_coord_loss += coord_loss.item()
                loss = conf_loss + coord_loss
                total_val_loss += loss.item()

                face_mask = has_face.bool()
                if face_mask.any():
                    face_coords_pred = coords_pred[face_mask]
                    face_targets = targets[face_mask]

                    ious = calculate_iou(face_coords_pred[:, :4], face_targets[:, :4])
                    iou_sum += ious.sum().item()

                    left_correct, right_correct = calculate_eye_accuracy(
                        face_coords_pred[:, 4:8], face_targets[:, 4:8], threshold=0.01
                    )
                    left_eye_correct += left_correct
                    right_eye_correct += right_correct

                    total_face_samples += face_mask.sum().item()
                total_samples += images.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_conf_loss = total_val_conf_loss / len(val_loader)
        avg_val_coord_loss = total_val_coord_loss / len(val_loader)

        if total_face_samples > 0:
            mean_iou_pct = (iou_sum / total_face_samples) * 100
            left_eye_acc_pct = (left_eye_correct / total_face_samples) * 100
            right_eye_acc_pct = (right_eye_correct / total_face_samples) * 100
        else:
            mean_iou_pct = 0.0
            left_eye_acc_pct = 0.0
            right_eye_acc_pct = 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_val_loss)

        epoch_duration = time.time() - epoch_start

        print(
            f"[Epoch {epoch + 1}] "
            f"Duration: {epoch_duration:.1f}s | "
            f"Train Loss: {avg_train_loss:.4f} (conf: {avg_conf_loss:.4f}, coord: {avg_coord_loss:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} (conf: {avg_val_conf_loss:.4f}, coord: {avg_val_coord_loss:.4f}) | "
            f"IoU: {mean_iou_pct:.2f}% | "
            f"Left Eye Acc: {left_eye_acc_pct:.2f}% | "
            f"Right Eye Acc: {right_eye_acc_pct:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        if log_file:
            _write_log_epoch(
                log_file,
                epoch + 1,
                epoch_duration,
                avg_train_loss,
                avg_val_loss,
                mean_iou_pct,
                left_eye_acc_pct,
                right_eye_acc_pct,
                current_lr,
            )

        final_metrics = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "iou": mean_iou_pct,
            "left_eye_acc": left_eye_acc_pct,
            "right_eye_acc": right_eye_acc_pct,
            "lr": current_lr,
        }

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"mobile_face_detector_epoch_{epoch + 1}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        if (
            mean_iou_pct >= target_iou
            and left_eye_acc_pct >= target_eye_acc
            and right_eye_acc_pct >= target_eye_acc
        ):
            print(f"\nTarget metrics achieved. Stopping training.")
            break

        epoch += 1

    if log_file:
        total_duration = time.time() - start_time
        _write_log_footer(log_file, config, epoch, total_duration, final_metrics)

    return model
