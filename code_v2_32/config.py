"""
Experiment Configuration Module
================================
Centralized configuration for all Knowledge Distillation experiments.

Author: Gheith Alrawahi
Institution: Nankai University
Thesis: Robust Knowledge Distillation for Compact Vision Models
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import torch

# =============================================================================
# Directory Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "figures"
DATA_DIR = PROJECT_ROOT / "data"

# Create directories
for dir_path in [RESULTS_DIR, MODELS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Base Configuration
# =============================================================================
@dataclass
class BaseConfig:
    """Base configuration shared across all experiments."""
    # Dataset
    dataset: str = "CIFAR-100"
    num_classes: int = 100
    train_samples: int = 50000
    test_samples: int = 10000
    
    # Models
    teacher_model: str = "EfficientNetV2-L"
    student_model: str = "EfficientNetV2-S"
    
    # Training
    num_epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.05
    patience: int = 30
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    
    # Checkpointing
    checkpoint_frequency: int = 10  # Save every N epochs
    keep_checkpoints: int = 5
    
    # Reproducibility
    seed: int = 42
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path):
        with open(path, 'r') as f:
            return cls(**json.load(f))


# =============================================================================
# Augmentation Configuration
# =============================================================================
@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # Basic augmentation
    random_crop: bool = True
    random_crop_padding: int = 4
    random_horizontal_flip: bool = True
    
    # Advanced augmentation
    auto_augment: bool = False
    auto_augment_policy: str = "CIFAR10"
    random_erasing: bool = False
    random_erasing_prob: float = 0.25
    
    # Mixing augmentation
    mixup: bool = True
    mixup_alpha: float = 0.8
    cutmix: bool = True
    cutmix_alpha: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Distillation Configuration
# =============================================================================
@dataclass
class DistillationConfig:
    """Knowledge Distillation configuration."""
    method: str = "standard_kd"  # "standard_kd" or "dkd"
    temperature: float = 4.0
    
    # Standard KD parameters
    alpha: float = 0.7  # Weight for soft loss (1-alpha for hard loss)
    
    # DKD parameters
    dkd_alpha: float = 1.0  # Weight for TCKD
    dkd_beta: float = 8.0   # Weight for NCKD
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Experiment Configurations
# =============================================================================
@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_id: str
    experiment_name: str
    description: str
    
    base: BaseConfig = field(default_factory=BaseConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    
    # Runtime fields (set when experiment starts)
    _run_id: str = field(default="", init=False, repr=False)
    _results_path: Path = field(default=None, init=False, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "run_id": self._run_id,
            "base": self.base.to_dict(),
            "augmentation": self.augmentation.to_dict(),
            "distillation": self.distillation.to_dict()
        }
    
    def get_results_dir(self, use_timestamp: bool = True) -> Path:
        """
        Get results directory for this experiment.
        
        Args:
            use_timestamp: If True, append timestamp to prevent overwriting.
                          If False, use experiment_id only (will overwrite).
        
        Returns:
            Path to results directory
        """
        from datetime import datetime
        
        if use_timestamp:
            # Generate unique run ID with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._run_id = f"{self.experiment_id}_{timestamp}"
        else:
            self._run_id = self.experiment_id
        
        self._results_path = RESULTS_DIR / self._run_id
        
        # Check if directory exists
        if self._results_path.exists():
            if use_timestamp:
                # This shouldn't happen with timestamp, but just in case
                import time
                time.sleep(1)
                return self.get_results_dir(use_timestamp=True)
            else:
                print(f" WARNING: Results directory already exists: {self._results_path}")
                print(f"   Previous results will be OVERWRITTEN!")
                response = input("   Continue? [y/N]: ").strip().lower()
                if response != 'y':
                    raise RuntimeError("Experiment cancelled to prevent overwriting.")
        
        self._results_path.mkdir(parents=True, exist_ok=True)
        return self._results_path
    
    def save(self, path: Optional[Path] = None):
        """Save configuration to JSON file."""
        if path is None:
            if self._results_path is None:
                path = RESULTS_DIR / self.experiment_id / "config.json"
            else:
                path = self._results_path / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved: {path}")


# =============================================================================
# Pre-defined Experiment Configurations
# =============================================================================

# Experiment v1: Baseline Standard KD
EXPERIMENT_V1 = ExperimentConfig(
    experiment_id="v1_baseline",
    experiment_name="Baseline Standard KD",
    description="Standard KD with basic Mixup/CutMix augmentation",
    base=BaseConfig(),
    augmentation=AugmentationConfig(
        auto_augment=False,
        random_erasing=False,
        mixup=True,
        cutmix=True
    ),
    distillation=DistillationConfig(
        method="standard_kd",
        alpha=0.7,
        temperature=4.0,
        label_smoothing=0.0  # No label smoothing in v1
    )
)

# Experiment v2: Enhanced Standard KD
EXPERIMENT_V2 = ExperimentConfig(
    experiment_id="v2_enhanced",
    experiment_name="Enhanced Standard KD",
    description="Standard KD with AutoAugment + RandomErasing + Label Smoothing + LR Warmup",
    base=BaseConfig(),
    augmentation=AugmentationConfig(
        auto_augment=True,
        random_erasing=True,
        mixup=True,
        cutmix=True
    ),
    distillation=DistillationConfig(
        method="standard_kd",
        alpha=0.7,
        temperature=4.0,
        label_smoothing=0.1
    )
)

# Experiment v3: DKD with beta=8.0
EXPERIMENT_V3 = ExperimentConfig(
    experiment_id="v3_dkd_beta8",
    experiment_name="DKD (β=8.0)",
    description="Decoupled KD with default beta=8.0 - Expected to show over-regularization",
    base=BaseConfig(num_epochs=300),  # Extended for teacher
    augmentation=AugmentationConfig(
        auto_augment=True,
        random_erasing=True,
        mixup=True,
        cutmix=True
    ),
    distillation=DistillationConfig(
        method="dkd",
        dkd_alpha=1.0,
        dkd_beta=8.0,
        temperature=4.0,
        label_smoothing=0.1
    )
)

# Experiment v3.1: DKD with tuned beta=2.0
EXPERIMENT_V3_1 = ExperimentConfig(
    experiment_id="v3_1_dkd_beta2",
    experiment_name="DKD (β=2.0) - Tuned",
    description="Decoupled KD with reduced beta=2.0 to fix over-regularization",
    base=BaseConfig(),
    augmentation=AugmentationConfig(
        auto_augment=True,
        random_erasing=True,
        mixup=True,
        cutmix=True
    ),
    distillation=DistillationConfig(
        method="dkd",
        dkd_alpha=1.0,
        dkd_beta=2.0,
        temperature=4.0,
        label_smoothing=0.1
    )
)

# Experiment v4: Best Teacher + Standard KD (Saturation Test)
EXPERIMENT_V4 = ExperimentConfig(
    experiment_id="v4_saturation",
    experiment_name="Saturation Test",
    description="Strong Teacher (v3) + Standard KD to test capacity saturation",
    base=BaseConfig(),
    augmentation=AugmentationConfig(
        auto_augment=True,
        random_erasing=True,
        mixup=True,
        cutmix=True
    ),
    distillation=DistillationConfig(
        method="standard_kd",
        alpha=0.7,
        temperature=4.0,
        label_smoothing=0.1
    )
)

# All experiments
ALL_EXPERIMENTS = {
    "v1": EXPERIMENT_V1,
    "v2": EXPERIMENT_V2,
    "v3": EXPERIMENT_V3,
    "v3.1": EXPERIMENT_V3_1,
    "v4": EXPERIMENT_V4
}


def get_experiment(name: str) -> ExperimentConfig:
    """Get experiment configuration by name."""
    if name not in ALL_EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}. Available: {list(ALL_EXPERIMENTS.keys())}")
    return ALL_EXPERIMENTS[name]


if __name__ == "__main__":
    # Print all experiment configurations
    print("=" * 60)
    print("EXPERIMENT CONFIGURATIONS")
    print("=" * 60)
    
    for name, config in ALL_EXPERIMENTS.items():
        print(f"\n{name}: {config.experiment_name}")
        print(f"  Method: {config.distillation.method}")
        print(f"  Augmentation: AutoAugment={config.augmentation.auto_augment}")
        if config.distillation.method == "dkd":
            print(f"  DKD Beta: {config.distillation.dkd_beta}")
        else:
            print(f"  KD Alpha: {config.distillation.alpha}")
