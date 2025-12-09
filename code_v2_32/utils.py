"""
Utility Functions Module
========================
Common utilities for training, evaluation, and logging.

Author: Gheith Alrawahi
Institution: Nankai University
"""

import os
import json
import glob
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


# =============================================================================
# Data Augmentation
# =============================================================================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, device: str = "cuda"):
    """Apply Mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def rand_bbox(size: Tuple, lam: float) -> Tuple[int, int, int, int]:
    """Generate random bounding box for CutMix."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, device: str = "cuda"):
    """Apply CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda based on actual box area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


# =============================================================================
# Loss Functions
# =============================================================================
def standard_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Standard Knowledge Distillation Loss.
    
    L = α * T² * KL(softmax(s/T) || softmax(t/T)) + (1-α) * CE(s, y)
    """
    # Soft targets (KL Divergence)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    # Hard targets with optional label smoothing
    hard_loss = F.cross_entropy(student_logits, labels, label_smoothing=label_smoothing)
    
    # Combined loss
    loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return loss


def _get_gt_mask(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Get mask for target (ground truth) class."""
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Get mask for non-target classes."""
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def _cat_mask(t: torch.Tensor, mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
    """Concatenate masked values."""
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    return torch.cat([t1, t2], dim=1)


def dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 8.0,
    temperature: float = 4.0
) -> torch.Tensor:
    """
    Decoupled Knowledge Distillation (DKD) Loss.
    
    Reference: Zhao et al. "Decoupled Knowledge Distillation", CVPR 2022
    Paper: https://arxiv.org/abs/2203.08679
    Official Code: https://github.com/megvii-research/mdistiller
    
    The key insight is to decouple the classical KD loss into two parts:
    
    L_DKD = α * L_TCKD + β * L_NCKD
    
    where:
    - L_TCKD (Target Class KD): Transfers knowledge about the target class probability
      This is the "binary" probability of target vs non-target classes.
      
    - L_NCKD (Non-Target Class KD): Transfers the "dark knowledge" - the relative
      probabilities among non-target classes. This is where the rich information lies.
    
    Temperature Scaling (T²):
    - Both TCKD and NCKD are scaled by T² to maintain gradient magnitude
    - This follows the original Hinton et al. (2015) formulation
    - When T > 1, softmax outputs become softer, gradients become smaller
    - T² compensates for this to keep gradient scale consistent
    
    Args:
        student_logits: Raw logits from student model [batch_size, num_classes]
        teacher_logits: Raw logits from teacher model [batch_size, num_classes]
        target: Ground truth labels [batch_size]
        alpha: Weight for TCKD loss (default: 1.0)
        beta: Weight for NCKD loss (default: 8.0, crucial hyperparameter)
        temperature: Softmax temperature (default: 4.0)
    
    Returns:
        Combined DKD loss (scalar tensor)
    
    Note on beta:
        - Higher beta emphasizes dark knowledge transfer
        - Original paper uses beta=8.0 for CIFAR-100
        - With strong augmentation, lower beta (2.0-4.0) may work better
    """
    gt_mask = _get_gt_mask(student_logits, target)
    other_mask = _get_other_mask(student_logits, target)
    
    # =========================================================================
    # TCKD: Target Class Knowledge Distillation
    # =========================================================================
    # Compute softmax probabilities with temperature scaling
    pred_student = F.softmax(student_logits / temperature, dim=1)
    pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
    
    # Extract target and non-target probability sums
    # pred_cat[:, 0] = p(target class)
    # pred_cat[:, 1] = sum of p(non-target classes) = 1 - p(target)
    pred_student_cat = _cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher_cat = _cat_mask(pred_teacher, gt_mask, other_mask)
    
    # KL divergence for binary distribution (target vs rest)
    # Note: We use log(p + eps) instead of log_softmax because pred_cat is already normalized
    log_pred_student_cat = torch.log(pred_student_cat + 1e-8)
    tckd_loss = F.kl_div(
        log_pred_student_cat, 
        pred_teacher_cat, 
        reduction='batchmean'
    ) * (temperature ** 2)  # T² scaling per Hinton et al.
    
    # =========================================================================
    # NCKD: Non-Target Class Knowledge Distillation
    # =========================================================================
    # Mask out target class by subtracting large value (effectively -inf after softmax)
    # This creates a distribution over non-target classes only
    mask_value = 1000.0  # Large enough to make target class probability ~0
    
    # Renormalized distribution over non-target classes
    pred_teacher_nckd = F.softmax(
        teacher_logits / temperature - mask_value * gt_mask.float(), 
        dim=1
    )
    log_pred_student_nckd = F.log_softmax(
        student_logits / temperature - mask_value * gt_mask.float(), 
        dim=1
    )
    
    # KL divergence for non-target class distribution
    nckd_loss = F.kl_div(
        log_pred_student_nckd, 
        pred_teacher_nckd, 
        reduction='batchmean'
    ) * (temperature ** 2)  # T² scaling per Hinton et al.
    
    # =========================================================================
    # Combined Loss
    # =========================================================================
    return alpha * tckd_loss + beta * nckd_loss


def kd_loss_with_mixup(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
    method: str = "standard_kd",
    temperature: float = 4.0,
    alpha: float = 0.7,
    dkd_alpha: float = 1.0,
    dkd_beta: float = 8.0,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Knowledge Distillation Loss for mixed samples (Mixup/CutMix compatible).
    """
    if method == "standard_kd":
        loss_a = standard_kd_loss(student_logits, teacher_logits, labels_a, 
                                   temperature, alpha, label_smoothing)
        loss_b = standard_kd_loss(student_logits, teacher_logits, labels_b,
                                   temperature, alpha, label_smoothing)
        loss = lam * loss_a + (1 - lam) * loss_b
    
    elif method == "dkd":
        dkd_a = dkd_loss(student_logits, teacher_logits, labels_a, dkd_alpha, dkd_beta, temperature)
        dkd_b = dkd_loss(student_logits, teacher_logits, labels_b, dkd_alpha, dkd_beta, temperature)
        soft_loss = lam * dkd_a + (1 - lam) * dkd_b
        
        hard_loss = lam * F.cross_entropy(student_logits, labels_a, label_smoothing=label_smoothing) + \
                    (1 - lam) * F.cross_entropy(student_logits, labels_b, label_smoothing=label_smoothing)
        
        loss = soft_loss + 0.1 * hard_loss
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # NaN safety
    if torch.isnan(loss):
        hard_loss = lam * F.cross_entropy(student_logits, labels_a, label_smoothing=label_smoothing) + \
                    (1 - lam) * F.cross_entropy(student_logits, labels_b, label_smoothing=label_smoothing)
        return hard_loss
    
    return loss


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(dataloader)
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "correct": correct,
        "total": total
    }


# =============================================================================
# Checkpointing
# =============================================================================
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_acc: float,
    history: Dict[str, List],
    save_path: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / "best_model.pth"
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    checkpoint_path: Path = None
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def cleanup_checkpoints(checkpoint_dir: Path, keep: int = 5):
    """Remove old checkpoints, keeping only the most recent ones."""
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    
    if len(checkpoints) > keep:
        for chk in checkpoints[:-keep]:
            os.remove(chk)
            print(f"  Removed old checkpoint: {chk.name}")


# =============================================================================
# Results Logging
# =============================================================================
class TrainingLogger:
    """
    Unified logger for both Teacher and Student training.
    Saves epoch results to CSV and JSON for later analysis and figure generation.
    """
    
    def __init__(self, name: str, results_dir: Path, model_type: str = "student"):
        """
        Initialize training logger.
        
        Args:
            name: Identifier for this training run (e.g., "teacher", "v1_baseline")
            results_dir: Base directory for results
            model_type: Either "teacher" or "student"
        """
        self.name = name
        self.model_type = model_type
        self.results_dir = results_dir / name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'timestamp': []
        }
        
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.start_time = None
        
        print(f"Logger initialized: {self.results_dir}")
    
    def start_training(self):
        """Mark training start time."""
        import time
        self.start_time = time.time()
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        learning_rate: float,
        auto_save: bool = True
    ):
        """Log metrics for one epoch and optionally save to disk."""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(round(train_loss, 6))
        self.history['val_loss'].append(round(val_loss, 6))
        self.history['val_accuracy'].append(round(val_accuracy, 4))
        self.history['learning_rate'].append(round(learning_rate, 8))
        self.history['timestamp'].append(datetime.now().isoformat())
        
        # Auto-save after every epoch (silent)
        if auto_save:
            self._save_history_silent()
        
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_epoch = epoch
            return True  # New best
        return False
    
    def _save_history_silent(self):
        """Save training history without printing (called after every epoch)."""
        df = pd.DataFrame(self.history)
        df.to_csv(self.results_dir / "training_history.csv", index=False)
        
        with open(self.results_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def save_history(self):
        """Save training history to CSV and JSON (with confirmation message)."""
        self._save_history_silent()
        print(f"History saved: {self.results_dir}")
    
    def save_checkpoint_history(self):
        """Save history during training (called at checkpoint intervals)."""
        self.save_history()
    
    def get_training_time(self) -> float:
        """Get training time in minutes."""
        import time
        if self.start_time:
            return (time.time() - self.start_time) / 60
        return 0.0
    
    def save_final_results(
        self,
        model_name: str,
        total_epochs: int,
        early_stopped: bool,
        config: Dict[str, Any] = None,
        teacher_accuracy: float = None,  # Only for student
    ):
        """Save final training results."""
        import time
        training_time = self.get_training_time()
        
        results = {
            "name": self.name,
            "model_type": self.model_type,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "results": {
                "final_accuracy": self.best_accuracy,
                "best_epoch": self.best_epoch,
                "total_epochs": total_epochs,
                "early_stopped": early_stopped,
                "training_time_minutes": round(training_time, 2)
            },
            "history_summary": {
                "final_train_loss": self.history['train_loss'][-1] if self.history['train_loss'] else None,
                "final_val_loss": self.history['val_loss'][-1] if self.history['val_loss'] else None,
                "best_val_accuracy": self.best_accuracy,
                "epochs_recorded": len(self.history['epoch'])
            }
        }
        
        # Add retention rate for student
        if self.model_type == "student" and teacher_accuracy:
            results["results"]["teacher_accuracy"] = teacher_accuracy
            results["results"]["retention_rate"] = round(
                (self.best_accuracy / teacher_accuracy) * 100, 2
            ) if teacher_accuracy > 0 else 0
        
        # Add config if provided
        if config:
            results["config"] = config
        
        with open(self.results_dir / "final_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE - {self.name}")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Best Accuracy: {self.best_accuracy:.2f}%")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Total Epochs: {total_epochs}")
        print(f"Early Stopped: {early_stopped}")
        print(f"Training Time: {training_time:.1f} minutes")
        if self.model_type == "student" and teacher_accuracy:
            print(f"Teacher Accuracy: {teacher_accuracy:.2f}%")
            print(f"Retention Rate: {results['results']['retention_rate']:.2f}%")
        print(f"Results saved: {self.results_dir}")
        
        return results


# Backward compatibility alias
class ExperimentLogger(TrainingLogger):
    """Alias for backward compatibility."""
    
    def __init__(self, experiment_id: str, results_dir: Path):
        super().__init__(experiment_id, results_dir, model_type="student")
        self.experiment_id = experiment_id
    
    def save_final_results(
        self,
        config: Dict[str, Any],
        teacher_accuracy: float,
        student_accuracy: float,
        total_epochs: int,
        early_stopped: bool,
        training_time: float
    ):
        """Save final experiment results (backward compatible)."""
        # Update best accuracy from parameter
        self.best_accuracy = student_accuracy
        
        results = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": {
                "teacher_accuracy": teacher_accuracy,
                "student_accuracy": student_accuracy,
                "retention_rate": (student_accuracy / teacher_accuracy) * 100 if teacher_accuracy > 0 else 0,
                "best_epoch": self.best_epoch,
                "total_epochs": total_epochs,
                "early_stopped": early_stopped,
                "training_time_minutes": training_time
            },
            "history_summary": {
                "final_train_loss": self.history['train_loss'][-1] if self.history['train_loss'] else None,
                "final_val_loss": self.history['val_loss'][-1] if self.history['val_loss'] else None,
                "best_val_accuracy": self.best_accuracy,
                "epochs_recorded": len(self.history['epoch'])
            }
        }
        
        with open(self.results_dir / "final_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS - {self.experiment_id}")
        print(f"{'='*60}")
        print(f"Teacher Accuracy: {teacher_accuracy:.2f}%")
        print(f"Student Accuracy: {student_accuracy:.2f}%")
        print(f"Retention Rate: {results['results']['retention_rate']:.2f}%")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Training Time: {training_time:.1f} minutes")
        print(f"Results saved: {self.results_dir / 'final_results.json'}")
        
        return results


# =============================================================================
# Model Utilities
# =============================================================================
def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    set_seed(42)
    print("Utilities loaded successfully!")
