import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_transforms(image_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """
    Returns data transforms for train or inference/validation phases.
    Uses ImageNet standard normalization for transfer learning backbones.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


class ChestXRayDataset(Dataset):
    """
    Chest Radiograph PyTorch Dataset supporting directory structures:
      data_dir/
        NORMAL/
        PNEUMONIA/
    """
    def __init__(self, file_paths: List[Path], labels: List[int], transform: Optional[transforms.Compose] = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def scan_dataset(data_dir: str, classes: List[str] = ("NORMAL", "PNEUMONIA")) -> Tuple[List[Path], List[int]]:
    """
    Scans a directory for images matching the given class subfolders.
    """
    file_paths = []
    labels = []
    base_path = Path(data_dir)

    for class_idx, class_name in enumerate(classes):
        class_folder = base_path / class_name
        if not class_folder.exists():
            continue
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            for img_file in class_folder.glob(ext):
                file_paths.append(img_file)
                labels.append(class_idx)

    return file_paths, labels


def compute_class_weights(labels: List[int], num_classes: int = 2) -> torch.Tensor:
    """
    Computes inverse frequency class weights to address radiograph dataset class imbalance:
      weight_c = total_samples / (num_classes * count_c)
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = [total / (num_classes * (count if count > 0 else 1)) for count in counts]
    return torch.tensor(weights, dtype=torch.float32)


def generate_mock_radiograph(output_path: Path, is_pneumonia: bool = False, image_size: int = 224):
    """
    Generates an anatomically structured synthetic chest radiograph for smoke testing,
    unit testing, and out-of-the-box demo without requiring 2GB download.
    Simulates thoracic cavity, rib silhouettes, spinal column, and opacities.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Background dark lung field
    img = Image.new("L", (image_size, image_size), color=25)
    draw = ImageDraw.Draw(img)

    # Spine center
    draw.rectangle([image_size * 0.46, 10, image_size * 0.54, image_size - 10], fill=110)

    # Left and Right Lung Fields (translucent darker ellipses)
    draw.ellipse([25, 30, image_size * 0.44, image_size * 0.85], fill=40, outline=90, width=3)
    draw.ellipse([image_size * 0.56, 30, image_size - 25, image_size * 0.85], fill=40, outline=90, width=3)

    # Rib shadows
    for y in range(45, int(image_size * 0.8), 24):
        draw.arc([15, y, image_size * 0.48, y + 30], start=180, end=360, fill=85, width=2)
        draw.arc([image_size * 0.52, y, image_size - 15, y + 30], start=180, end=360, fill=85, width=2)

    # Clavicles
    draw.line([25, 45, image_size * 0.45, 60], fill=120, width=4)
    draw.line([image_size - 25, 45, image_size * 0.55, 60], fill=120, width=4)

    # Heart silhouette on left hemithorax
    draw.ellipse([image_size * 0.40, image_size * 0.50, image_size * 0.65, image_size * 0.82], fill=130)

    # If pneumonia, add consolidated infiltrates/opacities in lung field
    if is_pneumonia:
        draw.ellipse([image_size * 0.60, image_size * 0.45, image_size * 0.85, image_size * 0.70], fill=125)
        draw.ellipse([image_size * 0.15, image_size * 0.55, image_size * 0.35, image_size * 0.75], fill=110)

    # Save as RGB
    img.convert("RGB").save(output_path)
