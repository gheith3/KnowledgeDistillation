"""
Data Loading Module
===================
CIFAR-100 data loading with configurable augmentation.

Author: Gheith Alrawahi
Institution: Nankai University
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment
from torch.utils.data import DataLoader
from typing import Tuple

from config import AugmentationConfig, DATA_DIR


# CIFAR-100 normalization values
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_train_transform(aug_config: AugmentationConfig) -> transforms.Compose:
    """Build training transform based on augmentation config."""
    transform_list = []
    
    # Basic augmentation
    if aug_config.random_crop:
        transform_list.append(
            transforms.RandomCrop(32, padding=aug_config.random_crop_padding)
        )
    
    if aug_config.random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    # AutoAugment
    if aug_config.auto_augment:
        policy = getattr(autoaugment.AutoAugmentPolicy, aug_config.auto_augment_policy)
        transform_list.append(autoaugment.AutoAugment(policy=policy))
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])
    
    # Random Erasing (after ToTensor)
    if aug_config.random_erasing:
        transform_list.append(
            transforms.RandomErasing(p=aug_config.random_erasing_prob, scale=(0.02, 0.2))
        )
    
    return transforms.Compose(transform_list)


def get_test_transform() -> transforms.Compose:
    """Build test transform (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])


def get_dataloaders(
    aug_config: AugmentationConfig,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 train and test dataloaders.
    
    Args:
        aug_config: Augmentation configuration
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Build transforms
    train_transform = get_train_transform(aug_config)
    test_transform = get_test_transform()
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"Data loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Augmentation: AutoAugment={aug_config.auto_augment}, "
          f"RandomErasing={aug_config.random_erasing}, "
          f"Mixup={aug_config.mixup}, CutMix={aug_config.cutmix}")
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    from config import AugmentationConfig
    
    aug_config = AugmentationConfig(
        auto_augment=True,
        random_erasing=True
    )
    
    train_loader, test_loader = get_dataloaders(aug_config, batch_size=128)
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
