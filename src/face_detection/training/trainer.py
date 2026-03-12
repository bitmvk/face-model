import time

import torch
import torch.nn as nn


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


def train_model(
    model,
    train_loader,
    val_loader,
    target_iou=85.0,
    target_eye_acc=80.0,
    lr=0.001,
    device="cuda",
    start_epoch=0,
    checkpoint_dir=".",
    checkpoint_interval=10,
    train_only_head=False,
):
    model = model.to(device)

    head_layers = [model.shared_features, model.reg_head]
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

    criterion_reg = nn.SmoothL1Loss()
    landmark_weight = 5.0

    start_time = time.time()

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

        if (epoch + 1) % checkpoint_interval == 0:
            import os

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

    return model
