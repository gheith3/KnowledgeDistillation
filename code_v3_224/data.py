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

# Path to pre-processed 224x224 data
PRELOADED_DATA_DIR = DATA_DIR / "cifar100_224"


class PreloadedCIFAR100(torch.utils.data.Dataset):
    """Dataset for pre-resized 224x224 CIFAR-100 images saved as .pt files."""
    
    def __init__(self, pt_path, transform=None):
        self.images, self.labels = torch.load(pt_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Images are already Tensors, apply transform if provided
        if self.transform:
            img = self.transform(img)
            
        return img, label


def get_train_transform_preloaded(aug_config: AugmentationConfig) -> transforms.Compose:
    """
    Build training transform for pre-loaded 224x224 Tensor images.
    Note: No Resize or ToTensor needed since data is already 224x224 Tensors.
    """
    transform_list = []
    
    # Augmentation (works on Tensors)
    if aug_config.random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    # AutoAugment expects PIL images, so we skip it for preloaded Tensors
    # If needed, convert: ToPILImage -> AutoAugment -> ToTensor
    # For now, we skip AutoAugment on preloaded data for speed
    
    # Normalize (works on Tensors)
    transform_list.append(transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD))
    
    # Random Erasing (works on Tensors)
    if aug_config.random_erasing:
        transform_list.append(
            transforms.RandomErasing(p=aug_config.random_erasing_prob, scale=(0.02, 0.2))
        )
    
    return transforms.Compose(transform_list)


def get_test_transform_preloaded() -> transforms.Compose:
    """Build test transform for pre-loaded 224x224 Tensor images."""
    return transforms.Compose([
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])


def get_train_transform(aug_config: AugmentationConfig, image_size: int = 224) -> transforms.Compose:
    """Build training transform with upscaling for 224x224 images (original CIFAR-100)."""
    transform_list = []
    
    # 1. UPSCALE: Resize raw 32x32 CIFAR images to target size (e.g., 224x224)
    transform_list.append(
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)
    )
    
    # 2. Augmentation (adapted for larger images)
    if aug_config.random_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    # AutoAugment (works better on 224x224)
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


def get_test_transform(image_size: int = 224) -> transforms.Compose:
    """Build test transform with upscaling (no augmentation)."""
    return transforms.Compose([
        # UPSCALE: Resize test images to match training size
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])


def get_dataloaders(
    aug_config: AugmentationConfig,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
    image_size: int = 224,
    use_preloaded: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 train and test dataloaders.
    
    Args:
        aug_config: Augmentation configuration
        batch_size: Batch size (default 32 for 224x224 images)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        image_size: Target image size (default 224 for EfficientNetV2)
        use_preloaded: If True, use pre-resized 224x224 .pt files (much faster)
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Check if preloaded data exists
    train_pt_path = PRELOADED_DATA_DIR / "train.pt"
    test_pt_path = PRELOADED_DATA_DIR / "test.pt"
    
    if use_preloaded and train_pt_path.exists() and test_pt_path.exists():
        # Use pre-loaded 224x224 Tensor data (FAST)
        print("Using pre-loaded 224x224 data from .pt files...")
        
        train_transform = get_train_transform_preloaded(aug_config)
        test_transform = get_test_transform_preloaded()
        
        train_dataset = PreloadedCIFAR100(str(train_pt_path), transform=train_transform)
        test_dataset = PreloadedCIFAR100(str(test_pt_path), transform=test_transform)
        
        # IMPORTANT: On Windows, num_workers must be 0 for preloaded data
        # Large tensors in memory cannot be shared between processes
        num_workers = 0
        print("  Note: Using num_workers=0 (required for preloaded data on Windows)")
        
        data_source = "preloaded"
    else:
        # Fallback: Load original CIFAR-100 and resize on-the-fly (SLOW)
        if use_preloaded:
            print("WARNING: Pre-loaded data not found. Run 'python prepare_data.py' first.")
            print("Falling back to on-the-fly resizing (slower)...")
        
        train_transform = get_train_transform(aug_config, image_size=image_size)
        test_transform = get_test_transform(image_size=image_size)
        
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
        
        data_source = "on-the-fly"
    
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
    
    print(f"Data loaded ({data_source}):")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Augmentation: AutoAugment={aug_config.auto_augment}, "
          f"RandomErasing={aug_config.random_erasing}, "
          f"Mixup={aug_config.mixup}, CutMix={aug_config.cutmix}")
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test data loading with 224x224 images
    from config import AugmentationConfig, BaseConfig
    
    base_config = BaseConfig()
    aug_config = AugmentationConfig(
        auto_augment=True,
        random_erasing=True
    )
    
    train_loader, test_loader = get_dataloaders(
        aug_config, 
        batch_size=base_config.batch_size,
        image_size=base_config.image_size
    )
    
    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")  # Should be [32, 3, 224, 224]
    print(f"Labels shape: {labels.shape}")
