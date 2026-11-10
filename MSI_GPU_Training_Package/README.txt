MSI BREAST IMAGE CLASSIFICATION — WINDOWS + RTX 4050

Project layout:
- MultiModel Breast Cancer MSI Dataset/
  - Ultrasound Images_MSI/benign and malignant
  - Histopathological_MSI/benign and malignant
  - Chest_XRay_MSI/Normal and Malignant
- MSI_GPU_Training_Package/train_breast_cancer.py
- msi_results_full/ or experiment folders with saved metrics and checkpoints

SETUP
1) Use the extracted dataset directory already present in the workspace:
   C:\Users\asus\OneDrive\Desktop\Breast Cncer Project\MultiModel Breast Cancer MSI Dataset

2) Create and activate a Python 3.11 virtual environment:
   cd "C:\Users\asus\OneDrive\Desktop\Breast Cncer Project"
   py -3.11 -m venv .venv311
   .venv311\Scripts\activate
   python -m pip install --upgrade pip

3) Install the training dependencies:
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   python -m pip install pandas scikit-learn pillow matplotlib

4) Verify the GPU is detected:
   python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA NOT AVAILABLE')"

TRAINING
- Train all modalities (Ultrasound, Histopathology, Chest X-ray):
  python MSI_GPU_Training_Package\train_breast_cancer.py --data "C:\Users\asus\OneDrive\Desktop\Breast Cncer Project\MultiModel Breast Cancer MSI Dataset" --output "C:\Users\asus\OneDrive\Desktop\Breast Cncer Project\msi_results_full" --modalities ultrasound histopathology chest_xray --epochs 12 --batch 16 --workers 0

- Train a single modality with a quick validation check:
  python MSI_GPU_Training_Package\train_breast_cancer.py --data "C:\Users\asus\OneDrive\Desktop\Breast Cncer Project\MultiModel Breast Cancer MSI Dataset" --output "C:\Users\asus\OneDrive\Desktop\Breast Cncer Project\msi_results_validation" --modalities ultrasound --epochs 1 --batch 8 --workers 0

- Targeted chest X-ray tuning experiment:
  python MSI_GPU_Training_Package\train_breast_cancer.py --data "C:\Users\asus\OneDrive\Desktop\Breast Cncer Project\MultiModel Breast Cancer MSI Dataset" --output "C:\Users\asus\OneDrive\Desktop\Breast Cncer Project\msi_results_chest_exp1" --modalities chest_xray --epochs 12 --batch 8 --lr 3e-4 --freeze-epochs 2 --workers 0

INFERENCE / EVALUATION
- The script automatically saves:
  - comparison_summary.csv
  - *_manifest.csv
  - *_splits.csv
  - *_history.csv
  - *_test_metrics.json
  - *_best_resnet18.pt
  - *_confusion_matrix.png
  - *_loss_curve.png
  - *_metrics_curve.png
  - *_roc_curve.png

- To evaluate a saved checkpoint, use the same training script with the same data/output folder and the desired modality. The script always selects the best validation-F1 checkpoint before final test evaluation.

DATA AUDIT NOTES
- The ultrasound dataset contains many augmentation variants such as rotated, sharpened, and repeated transforms. The script therefore uses a source-group ID that preserves the original source identity while stripping augmentation suffixes from the end of the filename.
- The chest X-ray dataset in the workspace is labeled only as 'Normal' and 'Malignant' in two class folders, with no source README, patient metadata, acquisition metadata, or provenance record in the dataset directory itself.
- The filenames use synthetic-style patterns such as '..._png.rf.<hash>_1.jpg'; the saved dataset manifest shows 1000 JPEG images at 640x640 and a per-image source ID that is generated from the filename, not a verified clinical study ID.
- Visual inspection of representative chest X-ray files shows red/purple pseudo-color regions, visible text overlays, irregular heatmap-like patterns, and no clear clinical chest anatomy. This is strong evidence that the images are not standard clinical chest radiographs, but it is not proof of the original source or exact generation pipeline.
- The training code never uses the test set for tuning. All checkpoint selection is based on validation Macro F1.
- The saved baseline results in msi_results_full are the first full-run results from the project and were preserved as-is.

CHEST X-RAY PROVENANCE STATUS
- Confirmed fact: the workspace contains only the class folders and image files; there is no verified source document, DICOM metadata, or dataset card showing where these chest X-ray images came from.
- Confirmed fact: the filenames and visuals are consistent with synthetic or heavily transformed/generated images rather than a standard clinical radiography collection.
- Unverified: the exact original dataset source, labeling policy, acquisition device, and clinical meaning of 'Normal' vs 'Malignant' labels.
- Restriction: the project should treat chest X-ray results as experimental and not as evidence of clinical capability until the dataset provenance and labels are resolved.
- Recommended next step: replace the chest X-ray subset with a documented, clinically sourced dataset or clearly restrict the final report to the fact that this subset is a synthetic/transformed challenge set rather than a real diagnostic dataset.

BASELINE vs IMPROVED RESULTS
Baseline files are saved under msi_results_full and are consistent with the original results:
- Ultrasound: Accuracy 0.8614; Macro F1 0.8608; Malignant precision 0.8370; Malignant recall 0.9059; ROC-AUC 0.9089
- Histopathology: Accuracy 0.7600; Macro F1 0.7596; Malignant precision 0.8738; Malignant recall 0.6569; ROC-AUC 0.8594
- Chest X-ray: Accuracy 0.4900; Macro F1 0.4838; Malignant precision 0.4762; Malignant recall 0.6250; ROC-AUC 0.5468

Improved chest X-ray experiment (validation-selected checkpoint):
- msi_results_chest_exp1/comparison_summary.csv
- Accuracy 0.5267; Macro F1 0.5261; Sensitivity 0.5211; Specificity 0.5316; ROC-AUC 0.5243

The baseline and improved runs use valid held-out test sets; model selection is driven only by validation data.

RESEARCH NOTICE
This project is for education and research only. It is not a clinical diagnostic system and does not establish medical validity.
