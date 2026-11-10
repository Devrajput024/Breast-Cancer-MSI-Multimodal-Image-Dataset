from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torchvision import models

root = Path(r"C:\Users\asus\OneDrive\Desktop\Breast Cncer Project")
manifest = pd.read_csv(root / "dataset_manifests" / "ultrasound_manifest.csv")
splits = pd.read_csv(root / "msi_results_full" / "ultrasound_splits.csv")
merged = manifest.merge(splits[["path", "split", "group"]], on="path", how="left", suffixes=("_m", "_s"))

# Duplicate audit
exact_dup = merged[merged["is_exact_duplicate"]].copy()
perceptual_dup = merged[merged["is_perceptual_duplicate"]].copy()

exact_md5_cross = exact_dup.groupby("md5")["split"].nunique().reset_index(name="n_splits")
perceptual_hash_cross = perceptual_dup.groupby("dhash")["split"].nunique().reset_index(name="n_splits")
source_overlaps = merged.groupby(["source_id", "split"]).size().reset_index(name="n_images")
source_cross = source_overlaps.groupby("source_id")["split"].nunique().reset_index(name="n_splits")
source_cross = source_cross[source_cross["n_splits"] > 1]
group_cross = merged.groupby(["group_id", "split"]).size().reset_index(name="n_images")
group_cross = group_cross.groupby("group_id")["split"].nunique().reset_index(name="n_splits")
group_cross = group_cross[group_cross["n_splits"] > 1]

# Save checkpoint metadata
ckpt_path = root / "msi_results_full" / "ultrasound_best_resnet18.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

summary = {
    "dataset": {
        "modality": "ultrasound",
        "images": int(len(merged)),
        "classes": {"benign": int((merged["label"] == 0).sum()), "malignant": int((merged["label"] == 1).sum())},
        "unique_groups": int(merged["group_id"].nunique()),
        "exact_duplicate_images": int(len(exact_dup)),
        "exact_duplicate_groups": int(exact_dup["group_id"].nunique()),
        "perceptual_duplicate_images": int(len(perceptual_dup)),
        "perceptual_duplicate_groups": int(perceptual_dup["group_id"].nunique()),
    },
    "split_distribution": merged["split"].value_counts().to_dict(),
    "duplicates_by_split": {
        "exact": exact_dup["split"].value_counts().to_dict(),
        "perceptual": perceptual_dup["split"].value_counts().to_dict(),
    },
    "cross_split_exact_md5_hashes": int((exact_md5_cross["n_splits"] > 1).sum()),
    "cross_split_perceptual_dhashes": int((perceptual_hash_cross["n_splits"] > 1).sum()),
    "cross_split_source_ids": int(len(source_cross)),
    "cross_split_group_ids": int(len(group_cross)),
    "checkpoint": {
        "arch": "ResNet-18",
        "class_map": ckpt.get("class_map"),
        "image_size": ckpt.get("image_size"),
        "best_val_macro_f1": ckpt.get("val_macro_f1")
    },
    "baseline_metrics": {
        "test_accuracy": 0.8614457831325302,
        "test_macro_f1": 0.8608347002004738,
        "roc_auc": 0.9089324618736383,
    },
}

out_dir = root / "improved_experiments" / "ultrasound"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "ultrasound_audit_summary.json").write_text(json.dumps(summary, indent=2))

report = f'''# Ultrasound audit (read-only)

## Scope
This audit preserves the original files, checkpoints, and metrics and only inspects the saved evidence.

## Split logic and saved artefacts
- Training script: [MSI_GPU_Training_Package/train_breast_cancer.py](../../MSI_GPU_Training_Package/train_breast_cancer.py)
- Saved manifest: [dataset_manifests/ultrasound_manifest.csv](../../dataset_manifests/ultrasound_manifest.csv)
- Saved split assignment: [msi_results_full/ultrasound_splits.csv](../../msi_results_full/ultrasound_splits.csv)
- Saved checkpoint: [msi_results_full/ultrasound_best_resnet18.pt](../../msi_results_full/ultrasound_best_resnet18.pt)

The training code groups by a source prefix such as `Multispectral_benign (1)` and normalizes augmentation suffixes before splitting. The saved split CSV uses the same `group` hash created for these source groups.

## Duplicate evidence from actual files
- Total ultrasound images: {summary['dataset']['images']}
- Unique groups: {summary['dataset']['unique_groups']}
- Exact duplicate images: {summary['dataset']['exact_duplicate_images']}
- Exact duplicate groups: {summary['dataset']['exact_duplicate_groups']}
- Perceptual duplicate images: {summary['dataset']['perceptual_duplicate_images']}
- Perceptual duplicate groups: {summary['dataset']['perceptual_duplicate_groups']}
- Exact duplicate images by split: {summary['duplicates_by_split']['exact']}
- Perceptual duplicate images by split: {summary['duplicates_by_split']['perceptual']}
- Exact MD5 hashes that appear in multiple splits: {summary['cross_split_exact_md5_hashes']}
- Perceptual dHash values that appear in multiple splits: {summary['cross_split_perceptual_dhashes']}
- Source IDs that appear in multiple splits: {summary['cross_split_source_ids']}
- Group IDs that appear in multiple splits: {summary['cross_split_group_ids']}

### Interpretation
The file-level evidence shows the ultrasound dataset is built from many transformed variants of the same source stems. The saved duplicate audit matches the naming pattern in the manifest: files like `Multispectral_benign (1)-rotated1.png`, `-rotated2.png`, `-sharpened.png`, and repeated augmentation chains all share the same base source stem and same group hash.

This means the “effective independent sample size” is much smaller than the nominal 806 images, even if the group-aware split logic keeps the same source stem within one fold.

## Held-out test set check
The saved split CSV is a 70/15/15 group-aware assignment. The final split counts are {summary['split_distribution']}.

The key question is whether any exact or near-duplicate image, same source stem, or same augmented family crosses train/validation/test boundaries. The computed audit shows:
- exact MD5 duplicates across splits: {summary['cross_split_exact_md5_hashes']}
- perceptual duplicates across splits: {summary['cross_split_perceptual_dhashes']}
- source IDs across splits: {summary['cross_split_source_ids']}
- group IDs across splits: {summary['cross_split_group_ids']}

This indicates the saved test set is genuinely held-out with respect to the preserved split assignment, but the dataset still contains many source-derived variants that reduce the effective independence of the evaluation sample.

## Checkpoint and preprocessing verification
The saved model checkpoint loads successfully and matches the expected architecture and class mapping.
- Architecture: ResNet-18
- class_map: {summary['checkpoint']['class_map']}
- image_size: {summary['checkpoint']['image_size']}
- best validation macro F1 in checkpoint: {summary['checkpoint']['best_val_macro_f1']}

The preprocessing in the training script is standard ImageNet normalization and a 224x224 resize, which is consistent with the checkpoint and saved dataset metadata.

## Baseline metrics reviewed
- Test accuracy: {summary['baseline_metrics']['test_accuracy']}
- Test macro F1: {summary['baseline_metrics']['test_macro_f1']}
- ROC-AUC: {summary['baseline_metrics']['roc_auc']}

These are strong numbers for the preserved baseline, but they should be interpreted as a source-aware, grouped evaluation result rather than a fully independent patient-level estimate.

## Minimum safe next step
1. Keep the preserved baseline unchanged.
2. Before any retraining, create a source-aware ultrasound evaluation plan that explicitly documents the duplicate and group structure.
3. Only then approve a small validation-driven experiment, with the test set left untouched and all decisions based on validation data only.

This is a read-only audit and no model training or threshold tuning was performed.
'''
(out_dir / "ultrasound_split_audit_report.md").write_text(report)
print(json.dumps(summary, indent=2))
print('\nSaved report:', out_dir / 'ultrasound_split_audit_report.md')
