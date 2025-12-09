"""
Model Definitions Module
========================
EfficientNetV2 Teacher and Student models for CIFAR-100.

Author: Gheith Alrawahi
Institution: Nankai University
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_v2_s, efficientnet_v2_l,
    EfficientNet_V2_S_Weights, EfficientNet_V2_L_Weights
)
from typing import Tuple

from config import MODELS_DIR
from utils import count_parameters, get_model_size_mb


def create_teacher_model(
    num_classes: int = 100,
    pretrained: bool = True,
    device: str = "cuda"
) -> nn.Module:
    """
    Create EfficientNetV2-L teacher model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        device: Device to load model on
    
    Returns:
        Teacher model
    """
    if pretrained:
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
    else:
        weights = None
    
    model = efficientnet_v2_l(weights=weights)
    
    # Replace classifier for CIFAR-100
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(device)
    
    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    print(f"Teacher Model: EfficientNetV2-L")
    print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Pretrained: {pretrained}")
    
    return model


def create_student_model(
    num_classes: int = 100,
    pretrained: bool = True,
    device: str = "cuda"
) -> nn.Module:
    """
    Create EfficientNetV2-S student model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        device: Device to load model on
    
    Returns:
        Student model
    """
    if pretrained:
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    else:
        weights = None
    
    model = efficientnet_v2_s(weights=weights)
    
    # Replace classifier for CIFAR-100
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(device)
    
    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    print(f"Student Model: EfficientNetV2-S")
    print(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Pretrained: {pretrained}")
    
    return model


def load_teacher_model(
    model_path: str,
    num_classes: int = 100,
    device: str = "cuda"
) -> nn.Module:
    """Load a trained teacher model from checkpoint."""
    model = create_teacher_model(num_classes, pretrained=False, device=device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Teacher model loaded from: {model_path}")
    
    return model


def load_student_model(
    model_path: str,
    num_classes: int = 100,
    device: str = "cuda"
) -> nn.Module:
    """Load a trained student model from checkpoint."""
    model = create_student_model(num_classes, pretrained=False, device=device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Student model loaded from: {model_path}")
    
    return model


def get_model_comparison() -> dict:
    """Get comparison statistics between teacher and student models."""
    # Create temporary models to get stats
    teacher = efficientnet_v2_l()
    student = efficientnet_v2_s()
    
    teacher_params = count_parameters(teacher)
    student_params = count_parameters(student)
    
    comparison = {
        "teacher": {
            "name": "EfficientNetV2-L",
            "parameters": teacher_params,
            "parameters_millions": teacher_params / 1e6
        },
        "student": {
            "name": "EfficientNetV2-S",
            "parameters": student_params,
            "parameters_millions": student_params / 1e6
        },
        "compression_ratio": teacher_params / student_params
    }
    
    return comparison


if __name__ == "__main__":
    # Test model creation
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    teacher = create_teacher_model(device=device)
    print()
    student = create_student_model(device=device)
    
    print("\n" + "=" * 60)
    comparison = get_model_comparison()
    print(f"Compression Ratio: {comparison['compression_ratio']:.2f}x")
