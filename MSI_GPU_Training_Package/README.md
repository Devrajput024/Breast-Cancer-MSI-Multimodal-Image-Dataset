# Multi-Modal Breast Cancer Classification Using ResNet-18

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20RTX%204050-76b900.svg)](https://www.nvidia.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for independent multi-modal image classification across three distinct medical imaging modalities using **PyTorch**, **torchvision**, and transfer learning with **ResNet-18**.

> **IMPORTANT CLINICAL DISCLAIMER**  
> This project is designed strictly for academic and educational research. The models, metrics, and scripts describe statistical pattern classification on benchmark image datasets and **DO NOT** constitute a medical diagnostic device, clinical decision-support system, or medical advice.

---

## 1. Project Overview & Architecture

This repository contains independent classification pipelines for three separate modalities from the **MultiModel Breast Cancer MSI Dataset**:
1. **Ultrasound Images MSI**: Binary classification between `benign` and `malignant` lesions.
2. **Histopathological MSI**: Binary classification between `benign` and `malignant` biopsy tissue specimens (BreaKHis format).
3. **Chest X-Ray MSI**: Binary classification between `Normal` and `Malignant` thoracic radiographs.

Each modality is trained independently using a dedicated **ResNet-18** backbone initialized with ImageNet pretrained weights. **Images from different modalities are deliberately NOT combined**, as there is no verified patient-level linkage across these three datasets.

---

## 2. Dataset Verification & Leakage Prevention

### Verified Dataset Image Counts

| Modality | Folder Name | Class 0 (Negative) | Class 1 (Positive) | Total Images | Resolution | Color Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ultrasound** | `Ultrasound Images_MSI` | `benign`: 406 | `malignant`: 400 | **806** | 224x224 (756), 227x227 (50) | RGB (PNG) |
| **Histopathology** | `Histopathological_MSI` | `benign`: 623 | `malignant`: 623 | **1,246** | 224x224 (1,246) | RGB (PNG) |
| **Chest X-Ray** | `Chest_XRay_MSI` | `Normal`: 500 | `Malignant`: 500 | **1,000** | 640x640 (1,000) | RGB (JPEG) |
| **Combined** | — | **1,529** | **1,523** | **3,052** | — | **100% Readable (0 Corrupt)** |

### Leakage-Safe Grouped Splitting (70% Train / 15% Val / 15% Test)
Standard random splitting causes severe data leakage when repeated views, patches, or augmented versions of the same original source appear in both training and test sets. To guarantee unbiased evaluation:

1. **Ultrasound Grouping**:
   - The 806 images derive from only **50 unique source images** (32 benign, 18 malignant) with multiple rotation (`-rotated1`, `-rotated2`) and sharpening (`-sharpened`) variants.
   - We extract the base source stem (e.g. `Multispectral_benign (1)`) and ensure that all augmented variants of an original image stay strictly within the same partition fold.
2. **Histopathology Patient/Slide Grouping**:
   - The 1,246 images follow the BreaKHis biopsy slide structure (e.g., `Multispectral_SOB_B_A-14-22549AB-200-001.png`), containing multiple fields of view per slide (51 unique slide groups).
   - We group by patient/slide prefix (`SOB_<B/M>_<Type>-<Year>-<SlideID>`), guaranteeing that crops from the same biopsy slide never appear across train and test sets.
3. **Chest X-Ray Instance Handling**:
   - Images are Roboflow exports with instance identifiers. Grouped strictly within class; cross-modality and patient linkage limitations are explicitly documented.

---

## 3. Hardware & Environment Configuration

- **Operating System**: Windows 11 (64-bit)
- **GPU Accelerator**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **CUDA Runtime**: CUDA 12.1 (Driver: 592.00, CUDA 13.1 capable)
- **Python Runtime**: Python 3.11.9 (64-bit) in `.venv311`

### Python Dependencies (`requirements.txt`)
```text
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.2.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=9.5.0
```

---

## 4. Execution Guide (Step-by-Step)

### Step 1: Activate Virtual Environment
Open PowerShell or Command Prompt in the project root:
```powershell
.venv311\Scripts\activate
```

Verify GPU availability:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

### Step 2: Run Dataset Integrity & Duplicate Validation
Inspects all 3,052 images, computes MD5/SHA256 and perceptual dHash, identifies duplicates, checks formats, and generates manifests:
```powershell
python MSI_GPU_Training_Package/validate_dataset.py --output dataset_manifests
```
**Outputs**:
- `dataset_manifests/complete_dataset_manifest.csv`
- `dataset_manifests/ultrasound_manifest.csv`
- `dataset_manifests/histopathology_manifest.csv`
- `dataset_manifests/chest_xray_manifest.csv`
- `dataset_manifests/dataset_validation_summary.json`

---

### Step 3: Train Models (Full Training or Smoke Test)

#### Quick Smoke Test (1 Epoch, Verifies Data Loading, GPU Forward Pass, Loss, Checkpointing & Plots):
```powershell
python MSI_GPU_Training_Package/train_breast_cancer.py --smoke-test --output smoke_test_outputs
```

#### Full Training (12 Epochs with 3 Warmup Epochs & Fine-Tuning):
```powershell
python MSI_GPU_Training_Package/train_breast_cancer.py --epochs 12 --freeze-epochs 3 --batch 16 --output outputs
```

**Training Workflow**:
- **Epochs 1–3**: ResNet-18 backbone is frozen; classification head warms up with `AdamW(lr=1e-3)`.
- **Epochs 4–12**: `layer4` is unfrozen and fine-tuned with `lr=1e-4`, while the head trains with `3e-4`.
- **Early Stopping**: Monitored on `val_macro_f1` (patience = 5 epochs).
- **Best Model Selection**: Checkpoint with highest validation Macro F1 score is saved.

---

### Step 4: Standalone Model Evaluation
Evaluate any trained model checkpoint against a split CSV:
```powershell
python MSI_GPU_Training_Package/evaluate_models.py --model outputs/ultrasound_best_resnet18.pt --splits outputs/ultrasound_splits.csv --split-name test --output evaluation_results
```

---

### Step 5: Single-Image Inference
Classify a new medical image using a trained model:
```powershell
python MSI_GPU_Training_Package/predict.py --model outputs/ultrasound_best_resnet18.pt --image "MultiModel Breast Cancer MSI Dataset/Ultrasound Images_MSI/malignant/Multispectral_malignant (1)-rotated1-rotated1.png"
```

For automated pipelines or JSON integration:
```powershell
python MSI_GPU_Training_Package/predict.py --model outputs/histopathology_best_resnet18.pt --image "MultiModel Breast Cancer MSI Dataset/Histopathological_MSI/benign/Multispectral_SOB_B_A-14-22549AB-200-001.png" --json
```

---

## 5. Experimental Results & Performance Summary

The final models were evaluated on strictly held-out test partitions (never seen during training or checkpoint selection).

| Modality | Best Epoch | Test Samples | Test Accuracy | Test Macro F1 | Malignant Recall (Sensitivity) | Specificity (Class 0) | Malignant Precision | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ultrasound** | **9** | 102 | **85.29%** | **0.8174** | **1.0000 (100%)** | **81.48%** | 58.33% | **1.0000** |
| **Histopathology** | **4** | 214 | **57.94%** | **0.5776** | 44.81% | **91.67%** | **93.24%** | **0.7184** |
| **Chest X-Ray** | **12** | 150 | **62.67%** | **0.6261** | **70.42%** | 55.70% | 58.82% | **0.6424** |

### Key Observations
1. **Ultrasound Classifier**:
   - Achieved **100% Malignant Sensitivity** (0 false negatives on the test set) and **1.000 ROC-AUC**, successfully learning distinctive morphological features of malignant vs benign ultrasound acoustic textures.
2. **Histopathology Classifier**:
   - Reached **91.7% Specificity** and **93.2% Malignant Precision**. Early stopping at Epoch 4 prevented the model from overfitting to slide-specific histological artifacts.
3. **Chest X-Ray Classifier**:
   - Improved from chance level (49.0% accuracy in earlier runs) to **62.7% accuracy** and **70.4% sensitivity** through two-stage backbone fine-tuning.

---

## 6. Directory Layout & Artifacts

```text
Breast Cancer Project/
├── .venv311/                            # Python 3.11 + PyTorch CUDA environment
├── MultiModel Breast Cancer MSI Dataset/ # Extracted medical image datasets
│   ├── Chest_XRay_MSI/                 # 1,000 images (Normal & Malignant)
│   ├── Histopathological_MSI/          # 1,246 images (benign & malignant)
│   └── Ultrasound Images_MSI/          # 806 images (benign & malignant)
├── dataset_manifests/                   # Manifests generated during Phase 2
│   ├── complete_dataset_manifest.csv
│   ├── ultrasound_manifest.csv
│   ├── histopathology_manifest.csv
│   ├── chest_xray_manifest.csv
│   └── dataset_validation_summary.json
├── outputs/                             # Final models, curves, and evaluation
│   ├── ultrasound_best_resnet18.pt
│   ├── histopathology_best_resnet18.pt
│   ├── chest_xray_best_resnet18.pt
│   ├── ultrasound_loss_curve.png
│   ├── ultrasound_metrics_curve.png
│   ├── ultrasound_confusion_matrix.png
│   ├── ultrasound_roc_curve.png
│   ├── histopathology_loss_curve.png
│   ├── histopathology_metrics_curve.png
│   ├── histopathology_confusion_matrix.png
│   ├── histopathology_roc_curve.png
│   ├── chest_xray_loss_curve.png
│   ├── chest_xray_metrics_curve.png
│   ├── chest_xray_confusion_matrix.png
│   ├── chest_xray_roc_curve.png
│   ├── comparison_summary.csv
│   └── *_test_metrics.json
└── MSI_GPU_Training_Package/            # Code package
    ├── train_breast_cancer.py          # Main training script
    ├── validate_dataset.py             # Validation and manifest generation
    ├── evaluate_models.py              # Standalone evaluation & plotting
    ├── predict.py                      # Single-image inference
    ├── requirements.txt                # Pip dependencies
    └── README.md                       # Complete documentation
```

---

## 7. Limitations & Open Challenges

1. **Absence of Cross-Modality Patient Linkage**:
   The three datasets originate from disparate collection protocols. They cannot be concatenated into an early-fusion or multi-modal co-attention model without true patient matching.
2. **Ultrasound Augmentation Redundancy**:
   Because the 806 ultrasound images originate from 50 primary lesions, the effective independent sample size is 50. Grouped splitting prevents leakage, but gathering larger prospective ultrasound cohorts remains necessary.
3. **Chest Radiography Task Specificity**:
   Chest radiographs predominantly capture lung, mediastinal, and thoracic pathology. Their clinical relevance to primary breast cancer detection is limited compared to dedicated mammography or ultrasound.
4. **Clinical Generalizability**:
   All models were trained on single-source benchmark distributions. Performance across different clinical imaging hardware, stain protocols, or patient demographics has not been validated.
