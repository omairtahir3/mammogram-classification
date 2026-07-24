# Two-Stage Mammography Classification System

A deep learning pipeline for automated breast cancer detection and classification using the **VinDr-Mammography dataset**. This project implements a **Multi-View DenseNet121** architecture that simultaneously processes Craniocaudal (CC) and Mediolateral Oblique (MLO) views to improve diagnostic accuracy.

## Architecture Overview
The system employs a two-stage classification approach based on BI-RADS scores:
1. **Stage 1 (Screening):** Classifies scans as Negative (BI-RADS 1) vs. Positive (BI-RADS 2–6).
2. **Stage 2 (Diagnostic):** Further classifies positive scans into Benign (BI-RADS 2–3) vs. Malignant (BI-RADS 4–6).

The core model is a highly modified **DenseNet121** initialized with ImageNet weights, containing over 16 million trainable parameters. It utilizes a custom multi-view input pipeline to aggregate spatial features from both standard mammography viewpoints.

## Technical Stack
- **Frameworks:** PyTorch, TorchVision
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Hardware:** NVIDIA CUDA (Optimized for single-GPU training with mixed memory management)

## Dataset & Preprocessing
The model is trained on the [VinDr-Mammography dataset](https://physionet.org/content/vindr-mammo/1.0.0/).
- **Data Split:** 70% Training (Class-Balanced) | 15% Validation | 15% Test
- **Transformations:** 
  - Resized to `384x384` resolution
  - Tensor conversion and ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Data Balancing:** Applied rigorous undersampling to the training subset to achieve a perfect 50/50 class distribution, preventing model bias toward majority classes.

## Training Configuration
- **Optimizer:** Adam (`lr=1e-4`, `weight_decay=5e-6`)
- **Scheduler:** `ReduceLROnPlateau` (Monitors validation loss, `patience=5`)
- **Loss Function:** Binary Cross Entropy (`BCELoss`)
- **Early Stopping:** Triggered dynamically based on validation loss divergence (`patience=75`).

## Performance Metrics

### Stage 1: Negative vs. Positive
Evaluated on a natural (imbalanced) test distribution of 1,426 paired samples.
- **Accuracy:** 69.00%
- **Recall (Sensitivity):** 62.6%
- **Precision:** 51.2%
- **F1 Score:** 0.563

### Stage 2: Benign vs. Malignant
Evaluated on a natural (imbalanced) test distribution of 470 paired positive samples.
- **Accuracy:** 62.77%
- **Recall (Sensitivity):** 50.7%
- **Precision:** 41.2%
- **F1 Score:** 0.455

*Note: The model generates ROC curves, Confusion Matrices, and AUC scores automatically during the evaluation phase using Seaborn.*

## Usage & Reproducibility

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/omairtahir3/Mammography-Classification.git
cd Mammography-Classification
pip install torch torchvision pandas numpy scikit-learn seaborn matplotlib
