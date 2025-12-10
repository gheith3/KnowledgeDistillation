# Cross-Resolution Knowledge Distillation

Robust Model Compression Under Strong Data Augmentation for Compact Vision Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains the experimental code and results for a Master's thesis investigating **Knowledge Distillation (KD)** stability under strong data augmentation. We demonstrate a novel **Cross-Resolution Distillation** approach that achieves superior model compression results.

### Key Contributions

1. **Regularization-Distillation Conflict**: We identify that Decoupled KD (DKD) with high β values collapses under strong augmentation, while Standard KD remains robust.

2. **Cross-Resolution Distillation**: We achieve **77.93%** student accuracy by using a 64×64 Teacher with a 32×32 Student, providing a **1.74%** improvement over the baseline.

3. **5.6× Model Compression**: The Student (21M parameters) retains **92.35%** of the Teacher's accuracy (84.39%) while being 5.6× smaller.

## Results Summary

| Model                              | Resolution | Accuracy   | Parameters |
| ---------------------------------- | ---------- | ---------- | ---------- |
| Teacher (EfficientNetV2-L)         | 32×32      | 76.65%     | 118M       |
| Teacher (EfficientNetV2-L)         | 64×64      | 84.39%     | 118M       |
| Student (Standard KD, v2)          | 32×32      | 76.19%     | 21M        |
| **Student (Cross-Resolution, v4)** | **64→32**  | **77.93%** | **21M**    |
| Student (DKD, β=8.0)               | 32×32      | 66.85%     | 21M        |
| Student (DKD, β=2.0)               | 32×32      | 75.63%     | 21M        |

## Repository Structure

```text
KnowledgeDistillation/
├── code_v2_32/                 # Main experimental code (32×32 baseline)
│   ├── 01_run_experiment.ipynb # Training notebook
│   ├── 02_generate_figures.ipynb # Figure generation
│   ├── 03_compare_results.ipynb  # Results comparison
│   ├── config.py               # Configuration settings
│   ├── data.py                 # Data loading utilities
│   ├── models.py               # Model definitions
│   ├── utils.py                # Training utilities
│   ├── checkpoints/            # Saved model weights
│   ├── results/                # Experiment results (JSON, CSV)
│   └── figures/                # Generated figures (PDF, PNG)
├── reports/                    # LaTeX reports
│   └── supervisor_report_02/   # Progress report with results
├── thesis_proposal/            # Thesis proposal document
└── local/                      # Local notes and guidelines
```

## Methodology

### Models

- **Teacher**: EfficientNetV2-L (118M parameters)
- **Student**: EfficientNetV2-S (21M parameters)

### Dataset

- **CIFAR-100**: 100 classes, 50K training images, 10K test images

### Data Augmentation

- AutoAugment (CIFAR-100 policy)
- Random Erasing (p=0.25)
- Mixup (α=0.8)
- CutMix (α=1.0)

### Distillation Methods

1. **Standard KD** (Hinton et al., 2015): Temperature-scaled soft labels
2. **Decoupled KD** (Zhao et al., 2022): Separates target and non-target class knowledge

## Quick Start

### Requirements

```bash
pip install torch torchvision timm pandas matplotlib seaborn
```

### Training

```python
# Open and run the notebook
jupyter notebook code_v2_32/01_run_experiment.ipynb
```

### Generate Figures

```python
jupyter notebook code_v2_32/02_generate_figures.ipynb
```

## Key Findings

### 1. Standard KD is More Robust

Standard KD achieved **99.40%** teacher retention without hyperparameter tuning, while DKD collapsed under strong augmentation.

### 2. Cross-Resolution Distillation Works

Using a higher-resolution Teacher (64×64) with a lower-resolution Student (32×32) improves performance by **1.74%** with zero additional inference cost.

### 3. DKD Requires Careful Tuning

DKD with β=8.0 collapsed to 66.85%. Reducing β to 2.0 recovered performance to 75.63%, but still underperformed Standard KD.

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{alrawahi2025crossres,
  title={Cross-Resolution Knowledge Distillation: Robust Model Compression Under Strong Data Augmentation for Compact Vision Models},
  author={Alrawahi, Gheith},
  school={Nankai University},
  year={2025}
}
```

## Author

**Gheith Alrawahi**  
Master's Student, Software Engineering  
Nankai University  
Supervisor: Prof. Jing Wang

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
