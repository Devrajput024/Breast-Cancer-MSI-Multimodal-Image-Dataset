# Ultrasound audit (read-only)

## Scope
This audit preserves the original files, checkpoints, and metrics and only inspects the saved evidence.

## Split logic and saved artefacts
- Training script: [MSI_GPU_Training_Package/train_breast_cancer.py](../../MSI_GPU_Training_Package/train_breast_cancer.py)
- Saved manifest: [dataset_manifests/ultrasound_manifest.csv](../../dataset_manifests/ultrasound_manifest.csv)
- Saved split assignment: [msi_results_full/ultrasound_splits.csv](../../msi_results_full/ultrasound_splits.csv)
- Saved checkpoint: [msi_results_full/ultrasound_best_resnet18.pt](../../msi_results_full/ultrasound_best_resnet18.pt)

The training code groups by a source prefix such as `Multispectral_benign (1)` and strips augmentation suffixes before splitting. The saved split CSV uses the same `group` hash created for these source groups.

## Duplicate evidence from actual files
- Total ultrasound images: 806
- Unique groups: 50
- Exact duplicate images: 53
- Exact duplicate groups: 18
- Perceptual duplicate images: 87
- Perceptual duplicate groups: 25
- Exact duplicate images by split: {'train': 35, 'test': 12, 'validation': 6}
- Perceptual duplicate images by split: {'train': 55, 'test': 21, 'validation': 11}
- Exact MD5 hashes that appear in multiple splits: 0
- Perceptual dHash values that appear in multiple splits: 0
- Source IDs that appear in multiple splits: 0
- Group IDs that appear in multiple splits: 0

### Interpretation
The file-level evidence shows the ultrasound dataset is built from many transformed variants of the same source stems. The saved duplicate audit matches the naming pattern in the manifest: files like `Multispectral_benign (1)-rotated1.png`, `-rotated2.png`, `-sharpened.png`, and repeated augmentation chains all share the same base source stem and same group hash.

This means the effective independent sample size is much smaller than the nominal 806 images, even if the group-aware split logic keeps each source stem within one fold.

## Held-out test set check
The saved split CSV is a 70/15/15 group-aware assignment. The final split counts are {'train': 512, 'test': 166, 'validation': 128}.

The key question is whether any exact or near-duplicate image, same source stem, or same augmented family crosses train/validation/test boundaries. The computed audit shows:
- exact MD5 duplicates across splits: 0
- perceptual duplicates across splits: 0
- source IDs across splits: 0
- group IDs across splits: 0

This indicates the preserved test set is held-out at the split-assignment level, but the dataset still contains many source-derived variants that reduce the effective independence of the evaluation sample.

## Checkpoint and preprocessing verification
The saved model checkpoint loads successfully and matches the expected architecture and class mapping.
- Architecture: ResNet-18
- class_map: {'benign': 0, 'malignant': 1}
- image_size: 224
- best validation macro F1 in checkpoint: None

The preprocessing in the training script is standard ImageNet normalization and a 224x224 resize, which is consistent with the checkpoint and saved dataset metadata.

## Baseline metrics reviewed
- Test accuracy: 0.8614457831325302
- Test macro F1: 0.8608347002004738
- ROC-AUC: 0.9089324618736383

These are strong numbers for the preserved baseline, but they should be interpreted as a source-aware, grouped evaluation result rather than a fully independent patient-level estimate.

## Minimum safe next step
1. Keep the preserved baseline unchanged.
2. Before any retraining, create a source-aware ultrasound evaluation plan that explicitly documents the duplicate and group structure.
3. Only then approve a small validation-driven experiment, with the test set left untouched and all decisions based on validation data only.

This is a read-only audit and no model training or threshold tuning was performed.
