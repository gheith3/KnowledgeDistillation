# Experiment Results Summary

**Generated:** 2025-12-10 16:45


## Overview

- **Total experiments:** 8
- **Teacher model (32x32):** EfficientNetV2-L (76.65%)
- **Teacher model (64x64):** EfficientNetV2-L (84.39%)
- **Student model:** EfficientNetV2-S
- **Dataset:** CIFAR-100
- **Training epochs:** 200 (with early stopping)


## Results Table

| ID             | Type    | Resolution   | Method      | AutoAug   |   Accuracy (%) | Teacher Gap (%)   |   Best Epoch |   Epochs |   Time (min) | Early Stop   | Retention (%)   |
|:---------------|:--------|:-------------|:------------|:----------|---------------:|:------------------|-------------:|---------:|-------------:|:-------------|:----------------|
| teacher        | Teacher | 32x32        | -           | N         |          76.65 | -                 |          199 |      200 |        389.7 | N            | -               |
| teacher_v4     | Teacher | 64x64        | -           | N         |          84.39 | -                 |          196 |      200 |       1048.9 | N            | -               |
| v1_baseline    | Student | 32x32        | standard_kd | N         |          76.12 | 0.53              |          200 |      200 |        300.2 | N            | 99.3            |
| v2_enhanced    | Student | 32x32        | standard_kd | Y         |          76.19 | 0.46              |          187 |      200 |        305.7 | N            | 99.4            |
| v3_dkd_beta8   | Student | 32x32        | dkd         | Y         |          66.85 | 9.80              |           54 |       84 |        124.9 | Y            | 87.2            |
| v3_1_dkd_beta2 | Student | 32x32        | dkd         | Y         |          75.63 | 1.02              |          184 |      200 |        295.3 | N            | 98.7            |
| v4_saturation  | Student | 64→32        | standard_kd | Y         |          77.93 | 6.46              |          180 |      200 |        336.5 | N            | 92.3            |



## Key Findings


### 1. Higher Resolution Improves Teacher

Training the teacher at 64×64 resolution instead of 32×32 improved accuracy 
from 76.65% to 84.39% (**+7.74%**). This demonstrates 
that higher resolution provides more visual information for the model to learn from.


### 2. Robustness of Standard KD

Standard KD proved exceptionally robust against strong augmentation. 
The baseline (v1) achieved 76.12%, and the enhanced version (v2) reached 76.19%. 
The marginal gain (+0.07%) despite adding AutoAugment indicates that 
**Standard KD is highly data-efficient** and stable.


### 3. Fragility of High-Beta DKD

Experiment v3 (DKD, β=8.0) collapsed to 66.85%, confirming the 
**Regularization-Distillation Conflict**. High reliance on dark knowledge interferes with 
strong augmentation noise. Reducing β to 2.0 (v3.1) recovered performance to 75.63%, 
validating that DKD requires sensitive tuning compared to Standard KD.


### 4. Capacity Saturation Analysis (Key Result)

The most significant finding is the comparison between v2 and v4:

| Metric | v2 (32×32 Teacher) | v4 (64×64 Teacher) |
|--------|-------------------|-------------------|
| Teacher Accuracy | 76.65% | 84.39% |
| Student Accuracy | 76.19% | 77.93% |
| Teacher Gap | 0.46% | 6.46% |

**Analysis:**
- Teacher improved by: **+7.74%**
- Student improved by: **+1.74%**
- Knowledge Transfer Efficiency: **22.5%**

The student captured only 22.5% of the teacher's improvement, demonstrating 
**partial capacity saturation** - the student benefits from a stronger teacher but with 
**diminishing returns**.


## Summary Statistics

| Metric | Value |
|--------|-------|
| Best Student Accuracy | 77.93% |
| Best Retention (vs 32×32) | 99.40% |
| Best Retention (vs 64×64) | 92.35% |
| Teacher Gap (v2) | 0.46% |
| Teacher Gap (v4) | 6.46% |
| Knowledge Transfer Efficiency | 22.5% |


## Experiment Details


### Teacher (32×32)
- **Accuracy:** 76.65%
- **Best Epoch:** 199
- **Training Time:** 389.7 minutes


### Teacher (64×64)
- **Accuracy:** 84.39%
- **Best Epoch:** 196
- **Training Time:** 1048.9 minutes


### v1_baseline
- **Accuracy:** 76.12%
- **Best Epoch:** 200
- **Training Time:** 300.2 minutes
- **Teacher Gap (vs 32×32):** 0.53%


### v2_enhanced
- **Accuracy:** 76.19%
- **Best Epoch:** 187
- **Training Time:** 305.7 minutes
- **Teacher Gap (vs 32×32):** 0.46%


### v3_dkd_beta8
- **Accuracy:** 66.85%
- **Best Epoch:** 54
- **Training Time:** 124.9 minutes
- **Teacher Gap (vs 32×32):** 9.80%


### v3_1_dkd_beta2
- **Accuracy:** 75.63%
- **Best Epoch:** 184
- **Training Time:** 295.3 minutes
- **Teacher Gap (vs 32×32):** 1.02%


### v4_saturation
- **Accuracy:** 77.93%
- **Best Epoch:** 180
- **Training Time:** 336.5 minutes
- **Teacher Gap (vs 64×64):** 6.46%
