# Ultrasound improvement experiment (read-only validation-driven run)

Goal: perform one conservative improvement using the existing saved split assignment, validation Macro F1 selection, and the same held-out test set used by the baseline.

Important constraints:
- Original baseline files, checkpoint, training script, and saved metrics are preserved.
- The existing split assignment from `msi_results_full/ultrasound_splits.csv` is used exactly as saved.
- The test set is not used for training, model selection, or threshold tuning.
- The model selected for the experiment is the checkpoint with the highest validation Macro F1.

Experiment details:
- Architecture: ResNet-18 pretrained on ImageNet
- Split: saved ultrasound split assignment (train/validation/test)
- Improvement: lower learning rate from 1e-3 to 5e-4 while keeping the same architecture and group-aware assignment
- Model selection: validation Macro F1
- Final evaluation: same held-out test set as baseline

Baseline metrics (preserved):
- accuracy: 0.8614
- macro_f1: 0.8608
- malignant_precision: 0.8370
- malignant_sensitivity_recall: 0.9059
- specificity_class0: 0.8148
- roc_auc: 0.9089

New metrics:
- accuracy: 0.8373
- macro_f1: 0.8352
- malignant_precision: 0.9677
- malignant_sensitivity_recall: 0.7059
- specificity_class0: 0.9753
- roc_auc: 0.9497

Comparison:
- accuracy delta: -0.0241
- macro_f1 delta: -0.0256
- malignant precision delta: 0.1308
- malignant recall delta: -0.2000
- specificity delta: 0.1605
- roc_auc delta: 0.0408

Decision:
The new model did not improve the fair held-out evaluation, so the original baseline model is retained.

The system is not presented as a clinically validated diagnostic tool.
