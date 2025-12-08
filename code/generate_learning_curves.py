"""
Generate Learning Curves for Knowledge Distillation Report
Compares Version 1 (Baseline) vs Version 2 (Enhanced)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
CHECKPOINT_DIR = Path('./outputs/checkpoints')
OUTPUT_DIR = Path('../reports/kd_result_01')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load checkpoints with history
print("Loading training histories...")

# Version 1 (Baseline): distilled_student_kd
v1_checkpoint = torch.load(CHECKPOINT_DIR / 'distilled_student_kd_epoch80.pth', map_location='cpu')
v1_history = v1_checkpoint['history']

# Version 2 (Enhanced): distilled_student_enhanced  
v2_checkpoint = torch.load(CHECKPOINT_DIR / 'distilled_student_enhanced_epoch80.pth', map_location='cpu')
v2_history = v2_checkpoint['history']

print(f"Version 1: {len(v1_history['train_loss'])} epochs")
print(f"Version 2: {len(v2_history['train_loss'])} epochs")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Version 1 (Baseline)
ax1 = axes[0]
epochs_v1 = range(1, len(v1_history['train_loss']) + 1)
ax1.plot(epochs_v1, v1_history['train_loss'], 'b-', label='Training Loss', linewidth=1.5)
ax1.plot(epochs_v1, v1_history['val_loss'], 'r-', label='Validation Loss', linewidth=1.5)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Version 1 (Baseline)\nStandard KD + CutMix/Mixup', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, len(v1_history['train_loss']))

# Plot Version 2 (Enhanced)
ax2 = axes[1]
epochs_v2 = range(1, len(v2_history['train_loss']) + 1)
ax2.plot(epochs_v2, v2_history['train_loss'], 'b-', label='Training Loss', linewidth=1.5)
ax2.plot(epochs_v2, v2_history['val_loss'], 'r-', label='Validation Loss', linewidth=1.5)

# Mark warmup region
ax2.axvspan(1, 5, alpha=0.2, color='green', label='Warmup (5 epochs)')

ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Version 2 (Enhanced)\nAutoAugment + Label Smoothing + LR Warmup', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, len(v2_history['train_loss']))

# Adjust layout
plt.tight_layout()

# Save figure
output_path = OUTPUT_DIR / 'learning_curves.pdf'
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_path}")

# Also save as PNG for preview
output_png = OUTPUT_DIR / 'learning_curves.png'
plt.savefig(output_png, format='png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_png}")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

print("\nVersion 1 (Baseline):")
print(f"  Final Train Loss: {v1_history['train_loss'][-1]:.4f}")
print(f"  Final Val Loss:   {v1_history['val_loss'][-1]:.4f}")
print(f"  Best Val Acc:     {max(v1_history['val_accuracy']):.2f}%")
print(f"  Gap (Val-Train):  {v1_history['val_loss'][-1] - v1_history['train_loss'][-1]:.4f}")

print("\nVersion 2 (Enhanced):")
print(f"  Final Train Loss: {v2_history['train_loss'][-1]:.4f}")
print(f"  Final Val Loss:   {v2_history['val_loss'][-1]:.4f}")
print(f"  Best Val Acc:     {max(v2_history['val_accuracy']):.2f}%")
print(f"  Gap (Val-Train):  {v2_history['val_loss'][-1] - v2_history['train_loss'][-1]:.4f}")
