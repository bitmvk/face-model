import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class COCONoHumanDataset(Dataset):
    """
    Dataset that returns COCO images without person annotations.
    Returns format compatible with CelebADataset: (image, target, has_face)
    where has_face is always 0 (no face) and target is zeros.

    Args:
        root_dir (str or Path): Directory containing the images (e.g., 'train2017/').
        ann_file (str): Path to the original COCO JSON (instances_train2017.json).
        transform (callable, optional): Transform to apply to images.
        start_idx (int): Start index for slicing the dataset.
        max_samples (int, optional): Maximum number of samples to use.
    """

    def __init__(
        self,
        root_dir,
        ann_file,
        transform=None,
        start_idx=0,
        max_samples=None,
        target_size=224,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size

        # Load original annotations
        with open(ann_file, "r") as f:
            data = json.load(f)

        # Build mapping: image id -> file name
        self.img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

        # Find category id for 'person'
        person_cat_id = None
        for cat in data["categories"]:
            if cat["name"] == "person":
                person_cat_id = cat["id"]
                break
        if person_cat_id is None:
            raise ValueError("No 'person' category found in annotation file.")

        # Collect all image ids that contain a person annotation
        images_with_person = set()
        for ann in data["annotations"]:
            if ann["category_id"] == person_cat_id:
                images_with_person.add(ann["image_id"])

        # All image ids in the dataset
        all_img_ids = set(self.img_id_to_file.keys())

        # Keep only those with NO person annotations
        self.img_ids = sorted(all_img_ids - images_with_person)

        # Apply slicing
        if max_samples is not None:
            end_idx = start_idx + max_samples
            self.img_ids = self.img_ids[start_idx:end_idx]
        elif start_idx > 0:
            self.img_ids = self.img_ids[start_idx:]

        # Target is always zeros for no-face images
        # Format: [x, y, w, h, le_x, le_y, re_x, re_y]
        self.target = torch.zeros(8, dtype=torch.float32)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.root_dir / self.img_id_to_file[img_id]
        image = Image.open(img_path).convert("RGB")

        # Letterbox to target_size (same as CelebADataset)
        orig_w, orig_h = image.size
        scale = min(self.target_size / orig_w, self.target_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)

        padded = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        padded.paste(image, (pad_left, pad_top))

        if self.transform is not None:
            padded = self.transform(padded)

        # Return format: (image, target, has_face)
        # has_face = 0 since these are no-human images
        return padded, self.target.clone(), 0


class MixedDataset(Dataset):
    """
    Dataset that mixes two datasets at a specified ratio.
    Samples 'ratio' items from dataset_b for every 1 item from dataset_a.

    Args:
        dataset_a: First dataset (e.g., CelebADataset with faces)
        dataset_b: Second dataset (e.g., COCONoHumanDataset without faces)
        ratio: Number of samples from dataset_b per 1 sample from dataset_a
    """

    def __init__(self, dataset_a, dataset_b, ratio=2):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.ratio = ratio

        # Calculate total length
        # For each sample in A, we take 'ratio' samples from B
        len_a = len(dataset_a)
        len_b = len(dataset_b)

        # Determine how many complete cycles we can do
        # Each cycle: 1 from A + ratio from B
        max_cycles = min(len_a, len_b // ratio)
        self.num_cycles = max_cycles
        self.total_length = max_cycles * (1 + ratio)

        if max_cycles == 0:
            raise ValueError(
                f"Not enough data to mix at ratio {ratio}: "
                f"dataset_a has {len_a}, dataset_b has {len_b} "
                f"(need at least {ratio} in B per 1 in A)"
            )

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx >= self.total_length:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.total_length}"
            )

        # Determine which cycle and position within cycle
        cycle = idx // (1 + self.ratio)
        position_in_cycle = idx % (1 + self.ratio)

        if position_in_cycle == 0:
            # Return from dataset_a
            return self.dataset_a[cycle]
        else:
            # Return from dataset_b
            b_idx = cycle * self.ratio + (position_in_cycle - 1)
            return self.dataset_b[b_idx]
