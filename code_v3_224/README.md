# Knowledge Distillation Experiments - 224x224 Resolution

## Master's Thesis: Robust Knowledge Distillation for Compact Vision Models

**Author:** Gheith Alrawahi  
**Institution:** Nankai University  
**Supervisor:** Prof. Jing Wang  
**Date:** December 2025

---

## 📁 Directory Structure

```text
code/
├── config.py                  # Centralized configuration for all experiments
├── utils.py                   # Utility functions (losses, augmentation, logging)
├── data.py                    # Data loading with configurable augmentation
├── models.py                  # Model definitions (Teacher/Student)
├── 01_run_experiment.ipynb    # Main experiment notebook
├── 02_generate_figures.ipynb  # Generate publication figures
├── 03_compare_results.ipynb   # Compare all experiments
├── README.md                  # This file
│
├── results/                   # Experiment results (JSON, CSV)
│   ├── teacher/               # Teacher training results
│   │   ├── training_history.csv
│   │   ├── training_history.json
│   │   └── final_results.json
│   │
│   ├── v1_baseline_YYYYMMDD_HHMMSS/   # Student results (timestamped)
│   │   ├── config.json
│   │   ├── training_history.csv
│   │   ├── training_history.json
│   │   └── final_results.json
│   │
│   └── ...                    # Other experiments
│
├── models/                    # Trained model weights
│   ├── teacher_trained.pth    # Best teacher model
│   ├── v1_baseline_best.pth   # Best student models
│   └── v1_baseline_final.pth  # Final student models
│
├── checkpoints/               # Training checkpoints (every 10 epochs)
│   ├── teacher/
│   │   └── teacher_epoch_10.pth, teacher_epoch_20.pth, ...
│   └── v1_baseline/
│       └── checkpoint_epoch_10.pth, ...
│
├── figures/                   # Generated figures (PDF, PNG)
└── data/                      # CIFAR-100 dataset (auto-downloaded)
```

---

## 🧪 Experiments

| ID   | Name       | Method       | Key Feature                     | Expected Result         |
| :--- | :--------- | :----------- | :------------------------------ | :---------------------- |
| -    | Teacher    | -            | EfficientNetV2-L baseline       | ~90%+ (224x224)         |
| v1   | Baseline   | Standard KD  | Mixup + CutMix only             | ~85%+                   |
| v2   | Enhanced   | Standard KD  | + AutoAugment + Label Smoothing | ~87%+                   |
| v3   | DKD β=8.0  | Decoupled KD | Default DKD parameters          | ~85% (over-regularized) |
| v3.1 | DKD β=2.0  | Decoupled KD | Tuned beta parameter            | ~87%+                   |
| v4   | Saturation | Standard KD  | Strong teacher + Standard KD    | ~87%+ (saturation)      |

> **Note:** This version uses **224×224 upscaled images** (from CIFAR-100's native 32×32) for optimal EfficientNetV2 performance. Expected accuracy is significantly higher than 32×32 baseline.

---

## 🚀 Quick Start

### 1. Run an Experiment

Open `01_run_experiment.ipynb` and change the experiment name:

```python
EXPERIMENT_NAME = "v1"  # Options: "v1", "v2", "v3", "v3.1", "v4"
USE_TIMESTAMP = True    # Set False to overwrite previous results
```

Then run all cells. The notebook will:

1. Train Teacher (if not exists)
2. Train Student with Knowledge Distillation
3. Save all results automatically

### 2. Generate Figures

After running experiments, open `02_generate_figures.ipynb` to create publication-quality figures:

- Loss comparison curves
- Accuracy convergence plots
- Retention bar charts

### 3. Compare Results

Open `03_compare_results.ipynb` to generate:

- Comparison tables (CSV, LaTeX)
- Summary report (Markdown)
- Key findings analysis

---

## 📊 Output Files

### For Teacher

```text
results/teacher/
├── training_history.csv      # Epoch-by-epoch metrics (saved after EVERY epoch)
├── training_history.json     # Same data in JSON format
└── final_results.json        # Final results summary
```

### For Each Student Experiment

```text
results/v1_baseline_20251207_135825/
├── config.json               # Experiment configuration
├── training_history.csv      # Epoch-by-epoch metrics (saved after EVERY epoch)
├── training_history.json     # Same data in JSON format
└── final_results.json        # Final results summary
```

### training_history.csv Format

```csv
epoch,train_loss,val_loss,val_accuracy,learning_rate,timestamp
1,4.2476,3.002,25.51,0.0002,2025-12-07T14:00:00
2,3.6917,2.4599,36.64,0.0004,2025-12-07T14:02:00
...
```

> **Note:** `training_history.csv/json` is automatically saved after **every epoch** to prevent data loss if training is interrupted.

### final_results.json Structure

**Teacher:**

```json
{
  "name": "teacher",
  "model_type": "teacher",
  "model_name": "EfficientNetV2-L",
  "results": {
    "final_accuracy": 75.76,
    "best_epoch": 67,
    "total_epochs": 80,
    "early_stopped": true,
    "training_time_minutes": 120.5
  }
}
```

**Student:**

```json
{
  "name": "v1_baseline_20251207_135825",
  "model_type": "student",
  "model_name": "EfficientNetV2-S",
  "results": {
    "final_accuracy": 72.50,
    "best_epoch": 55,
    "total_epochs": 70,
    "early_stopped": true,
    "training_time_minutes": 90.2,
    "teacher_accuracy": 75.76,
    "retention_rate": 95.7
  },
  "config": { ... }
}
```

---

## ⚙️ Configuration

All hyperparameters are defined in `config.py`:

### Base Configuration

| Parameter              | Value | Description                                   |
| :--------------------- | :---- | :-------------------------------------------- |
| `image_size`           | 224   | Input image resolution (upscaled from 32×32)  |
| `num_epochs`           | 200   | Maximum training epochs                       |
| `batch_size`           | 32    | Batch size (reduced for 224×224 memory usage) |
| `learning_rate`        | 0.001 | Initial learning rate                         |
| `weight_decay`         | 0.05  | AdamW weight decay                            |
| `patience`             | 30    | Early stopping patience                       |
| `warmup_epochs`        | 5     | Learning rate warmup epochs                   |
| `checkpoint_frequency` | 10    | Save checkpoint every N epochs                |
| `keep_checkpoints`     | 5     | Keep last N checkpoints                       |

> **Memory Note:** 224×224 images are ~50x larger than 32×32. Batch size is reduced from 128 to 32 to avoid OOM errors. If you still encounter OOM, reduce to 16.

### Data Saving Strategy

| Event                       | What is Saved                  | Location                       |
| :-------------------------- | :----------------------------- | :----------------------------- |
| **Every epoch**             | `training_history.csv/json`    | `results/{experiment}/`        |
| **Every 10 epochs**         | Checkpoint (model + optimizer) | `checkpoints/{experiment}/`    |
| **On accuracy improvement** | Best model                     | `models/{experiment}_best.pth` |
| **End of training**         | Final results + model          | `results/` + `models/`         |

### Augmentation

| Parameter         | v1    | v2   | v3/v3.1 | v4   |
| :---------------- | :---- | :--- | :------ | :--- |
| `auto_augment`    | False | True | True    | True |
| `random_erasing`  | False | True | True    | True |
| `mixup`           | True  | True | True    | True |
| `cutmix`          | True  | True | True    | True |
| `label_smoothing` | 0.0   | 0.1  | 0.1     | 0.1  |

### Distillation

| Parameter     | Standard KD | DKD (v3) | DKD (v3.1) |
| :------------ | :---------- | :------- | :--------- |
| `method`      | standard_kd | dkd      | dkd        |
| `temperature` | 4.0         | 4.0      | 4.0        |
| `alpha`       | 0.7         | -        | -          |
| `dkd_alpha`   | -           | 1.0      | 1.0        |
| `dkd_beta`    | -           | 8.0      | 2.0        |

---

## 📈 Key Findings

1. **224×224 resolution significantly boosts performance** - Teacher accuracy expected >90%
2. **Standard KD outperforms DKD** under strong augmentation
3. **DKD is sensitive to beta parameter** - reducing from 8.0 to 2.0 improves performance
4. **Capacity saturation exists** - stronger teachers don't always improve students
5. **AutoAugment + Label Smoothing** significantly improve results
6. **BICUBIC interpolation** used for upscaling to preserve image quality

---

## 🔧 Requirements

```text
torch>=2.0
torchvision
numpy
pandas
matplotlib
seaborn
tqdm
tabulate
```

Install all requirements:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn tqdm tabulate
```

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@mastersthesis{alrawahi2025kd,
  author = {Gheith Alrawahi},
  title = {Robust Knowledge Distillation for Compact Vision Models},
  school = {Nankai University},
  year = {2025}
}
```
