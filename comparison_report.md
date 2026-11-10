# Chest X-ray comparison and limitation report

## Verified baseline results

The baseline metrics were verified from the saved files in the workspace:

- [msi_results_full/comparison_summary.csv](msi_results_full/comparison_summary.csv)
- [msi_results_full/ultrasound_test_metrics.json](msi_results_full/ultrasound_test_metrics.json)
- [msi_results_full/histopathology_test_metrics.json](msi_results_full/histopathology_test_metrics.json)
- [msi_results_full/chest_xray_test_metrics.json](msi_results_full/chest_xray_test_metrics.json)

### Baseline metrics

| Modality | Accuracy | Macro F1 | Malignant Precision | Malignant Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Ultrasound | 0.8614 | 0.8608 | 0.8370 | 0.9059 | 0.9089 |
| Histopathology | 0.7600 | 0.7596 | 0.8738 | 0.6569 | 0.8594 |
| Chest X-ray | 0.4900 | 0.4838 | 0.4762 | 0.6250 | 0.5468 |

## Verified chest X-ray experiment results

The saved chest X-ray experiment output is in:

- [msi_results_chest_exp1/comparison_summary.csv](msi_results_chest_exp1/comparison_summary.csv)
- [msi_results_chest_exp1/chest_xray_test_metrics.json](msi_results_chest_exp1/chest_xray_test_metrics.json)
- [msi_results_chest_exp1/chest_xray_history.csv](msi_results_chest_exp1/chest_xray_history.csv)
- [msi_results_chest_exp1/chest_xray_confusion_matrix.png](msi_results_chest_exp1/chest_xray_confusion_matrix.png)
- [msi_results_chest_exp1/chest_xray_roc_curve.png](msi_results_chest_exp1/chest_xray_roc_curve.png)

### Chest X-ray experiment metrics

- Test accuracy: 0.5267
- Macro F1: 0.5261
- Malignant sensitivity/recall: 0.5211
- Malignant precision: 0.5000
- Specificity: 0.5316
- ROC-AUC: 0.5243

## Dataset audit

The dataset root is [MultiModel Breast Cancer MSI Dataset](MultiModel%20Breast%20Cancer%20MSI%20Dataset). The chest X-ray subset is [MultiModel Breast Cancer MSI Dataset/Chest_XRay_MSI](MultiModel%20Breast%20Cancer%20MSI%20Dataset/Chest_XRay_MSI). The labels are `Normal` and `Malignant` in two class folders.

Important findings, with what is confirmed vs uncertain:

- Confirmed: the workspace contains only the class folders and image files; there is no source README, provenance record, or patient-level metadata for the chest X-ray set in the dataset directory.
- Confirmed: the filenames include `rf.`-style tokens such as `..._png.rf.<hash>_1.jpg`.
- Confirmed: representative images show pseudo-color heatmap-like regions, obvious text overlays, and no recognizable conventional chest anatomy.
- Confirmed: the chest X-ray subset behaves near chance (baseline accuracy ~0.49, ROC-AUC ~0.55; validation-only experiment ~0.53 macro F1), which is consistent with a weak or misaligned task.
- Not confirmed: the exact original dataset source, the acquisition pipeline, or whether the labels correspond to real clinical diagnoses.
- Therefore: the evidence supports the claim that this chest X-ray subset is not a clean, clinically sourced radiography dataset, but it does not prove the exact generator or the clinical meaning of the labels without source documentation.
- This is a likely source of weak learning and near-random discrimination, but the project should avoid claiming more than this until provenance is validated.

## Confusion matrix interpretation

From the saved confusion matrix in [msi_results_chest_exp1/chest_xray_test_metrics.json](msi_results_chest_exp1/chest_xray_test_metrics.json):

- True Normal / Predicted Normal: 42
- True Normal / Predicted Malignant: 37
- True Malignant / Predicted Normal: 34
- True Malignant / Predicted Malignant: 37

This means:

- 37 false positives: normal cases were classified as malignant.
- 34 false negatives: malignant cases were classified as normal.
- The model is close to random, with the class boundary not being stable across the held-out test set.

## Underfitting vs overfitting

From [msi_results_chest_exp1/chest_xray_history.csv](msi_results_chest_exp1/chest_xray_history.csv):

- Train loss drops from about 0.713 to about 0.545.
- Train accuracy rises from about 0.516 to about 0.719.
- Validation loss stays around 0.72 to 0.81 and does not improve consistently.
- Validation Macro F1 peaks at epoch 4 (0.5495) and then declines.

Interpretation:

- The model is not strongly learning a clean signal; it is partially learning but not generalizing well.
- The later epochs show mild overfitting after the validation peak, but the main issue is weak dataset signal and likely label/data mismatch rather than a simple optimization bug.
- This is best described as a poor-quality or misaligned task definition, not a clean underfitting-only issue.

## Limitations and recommended next action

- No further expensive training is recommended before validating or replacing the chest X-ray dataset source.
- The ultrasound and histopathology baselines should remain unchanged unless a specific issue is found in those datasets.
- The chest X-ray results should be treated as experimental and not as evidence of real clinical performance.
- The next recommended action is to audit the dataset provenance and class labels directly with the original source, and only then decide whether to relabel, filter, or replace the chest X-ray set before additional model training.
