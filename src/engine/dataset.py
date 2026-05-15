"""
dataset.py — Dual-Pipeline Data Loader
=======================================

This file handles loading and preprocessing images from the State Farm dataset.

KEY DESIGN DECISION — Two Pipelines:
- The CustomDriverCNN uses GRAYSCALE (1-channel) input because it was built that way.
- The transfer learning models (MobileNet, ResNet, EfficientNet) are pre-trained on
  ImageNet which uses RGB (3-channel) images. We must match the format they expect.

This module automatically switches between both pipelines based on the model's input_mode.

Data Augmentation (Training Only):
- We apply random transformations to training images to make the model more robust.
- The validation set uses NO augmentation — we always evaluate on clean, original images.
- This gap between train and val transforms is what prevents overfitting.
"""

import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class DriverDistractionDataset(Dataset):
    """
    Custom PyTorch Dataset for the State Farm Distracted Driver images.
    Stores file paths (not images) to keep memory usage low.
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels     = labels
        self.transform  = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image    = Image.open(img_path).convert("RGB")  # Always open as RGB first
        label    = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    data_dir   = "./data/processed/state-farm/train",
    batch_size = 32,
    test_split = 0.2,
    input_mode = "grayscale",   # "grayscale" for CustomCNN | "rgb" for transfer learning
    quick_mode = False          # True = use only 20% of data for fast UI testing
):
    """
    Build and return train/validation DataLoaders.

    Args:
        data_dir:   Path to the processed dataset folder (must have c0-c9 subfolders)
        batch_size: Number of images processed per training step
        test_split: Fraction of data held out for validation (default 20%)
        input_mode: 'grayscale' or 'rgb' — determines preprocessing pipeline
        quick_mode: If True, only loads 20% of data for fast iteration/testing

    Returns:
        (train_loader, val_loader): Two PyTorch DataLoader objects
    """
    classes      = [f"c{i}" for i in range(10)]
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    all_paths, all_labels = [], []

    for cls_name in classes:
        cls_dir   = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        img_paths = glob.glob(os.path.join(cls_dir, "*.jpg"))
        all_paths.extend(img_paths)
        all_labels.extend([class_to_idx[cls_name]] * len(img_paths))

    if len(all_paths) == 0:
        raise ValueError(
            f"No images found in '{data_dir}'.\n"
            f"Please run the dataset download script first:\n"
            f"  python src/data_pipeline/download_dataset.py"
        )

    # Quick mode: use only 20% of data for fast UI/architecture testing
    if quick_mode:
        all_paths, _, all_labels, _ = train_test_split(
            all_paths, all_labels, test_size=0.8, random_state=42, stratify=all_labels
        )
        print(f"⚡ QUICK MODE: Using {len(all_paths)} images (20% of dataset)")

    # Split dataset: 80% train, 20% validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels,
        test_size=test_split, random_state=42, stratify=all_labels
    )

    # ------------------------------------------------------------------
    # PIPELINE A: Grayscale (for CustomDriverCNN)
    # ------------------------------------------------------------------
    if input_mode == "grayscale":
        # Training: augmentation applied to prevent overfitting
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),        # Mirror the image 50% of the time
            transforms.RandomRotation(degrees=15),          # Tilt image ±15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random lighting changes
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])    # Normalize to [-1, 1] range
        ])
        # Validation: no augmentation — always evaluate on clean images
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    # ------------------------------------------------------------------
    # PIPELINE B: RGB (for MobileNetV2, ResNet18, EfficientNet-B0)
    # ImageNet normalization values are standard for all pre-trained models.
    # ------------------------------------------------------------------
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            # ImageNet mean & std — pre-trained models REQUIRE these exact values
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    train_dataset = DriverDistractionDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset   = DriverDistractionDataset(val_paths,   val_labels,   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"📦 Dataset loaded | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Mode: {input_mode.upper()}")
    return train_loader, val_loader
