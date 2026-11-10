# Audit Review: Split Integrity and Interpretability

## Scope

This review is audit-only and does not change model architecture, retrain any model, regenerate splits, or alter baseline outputs.

The review was based on the preserved project artifacts already in the workspace, including:

- [dataset_manifests/dataset_validation_summary.json](../dataset_manifests/dataset_validation_summary.json)
- [dataset_manifests/ultrasound_manifest.csv](../dataset_manifests/ultrasound_manifest.csv)
- [dataset_manifests/histopathology_manifest.csv](../dataset_manifests/histopathology_manifest.csv)
- [msi_results_full/ultrasound_splits.csv](../msi_results_full/ultrasound_splits.csv)
- [msi_results_full/histopathology_splits.csv](../msi_results_full/histopathology_splits.csv)
- [msi_feature_audit/audit_summary.json](../msi_feature_audit/audit_summary.json)
- [msi_feature_audit/ultrasound/gradcam/gradcam_summary.csv](../msi_feature_audit/ultrasound/gradcam/gradcam_summary.csv)
- [msi_feature_audit/histopathology/gradcam/gradcam_summary.csv](../msi_feature_audit/histopathology/gradcam/gradcam_summary.csv)
- [msi_feature_audit/ultrasound/masking_summary.csv](../msi_feature_audit/ultrasound/masking_summary.csv)
- [msi_feature_audit/histopathology/masking_summary.csv](../msi_feature_audit/histopathology/masking_summary.csv)

## 1. Split integrity status

### Ultrasound

Status: Uncertain but likely acceptable with explicit caveat.

Evidence:
- The validation summary reports 806 ultrasound images, 50 unique source groups, 53 exact duplicates, and 87 perceptual duplicates.
- The manifest contains a source_id field derived from the base source stem such as Multispectral_benign (1), and the code in [MSI_GPU_Training_Package/train_breast_cancer.py](../MSI_GPU_Training_Package/train_breast_cancer.py) groups ultrasound variants by source stem before splitting.
- This indicates that the intended leakage-control rule is source-group aware rather than image-instance random splitting.
- The exact duplicate and perceptual duplicate rows belong to the same source_id and are likely transformed variants of the same source lesion, which is consistent with a source-aware split strategy.

Important limitation:
- The saved manifest gives a strong source-based grouping signal, but it does not include a verified patient-level identifier or a provenance record proving that each source stem corresponds to a unique patient or lesion episode.
- Because the original dataset provenance is not fully documented, the split is best described as source-aware and likely leakage-resistant, but not fully clinically validated.

Conclusion:
- The design is consistent with the project’s split strategy and likely avoids direct source leakage across splits.
- However, the evidence is not strong enough to call the ultrasound split definitively proven leakage-free without a direct cross-split duplicate-overlap check on the saved split assignments.

### Histopathology

Status: Pass with caveat.

Evidence:
- The validation summary reports 1246 histopathology images, 51 unique groups, zero exact duplicates, and zero perceptual duplicates.
- The manifest shows group_id and source_id values derived from the slide identifier prefix such as Multispectral_SOB_B_A-14-22549AB.
- This is the expected BreaKHis-style slide structure, and the code groups by a slide-like prefix, which is a strong grouping signal for leakage prevention.
- The saved split files and manifest structure are consistent with grouped split logic rather than random instance splitting.

Important limitation:
- The project still relies on a derived grouping heuristic from filename structure, not a separate patient record or explicit slide metadata file.
- This means the grouping is likely reliable, but it is not a fully externally validated clinical grouping workflow.

Conclusion:
- The histopathology split integrity is stronger than ultrasound because the slide-level grouping is well aligned with the image naming convention and the duplicate audit is clean.
- The status is pass with a metadata caveat, not a fully proven patient-level guarantee.

### Chest X-ray

Status: Not part of the current audit-only validation phase for the breast-cancer comparison, but it remains exploratory and unreliable.

Evidence:
- The chest X-ray subset contains 1000 images and 1000 unique groups in the validation summary.
- There are 113 perceptual duplicates, even though exact duplicates are zero.
- The project materials already flag the chest X-ray subset as potentially synthetic or transformed and caution that the provenance is unverified.

Conclusion:
- This subset should remain isolated as exploratory and should not be treated as equivalent in reliability to the ultrasound or histopathology baseline models.

## 2. Duplicate leakage findings

### Ultrasound duplicate issue

The ultrasound dataset contains:
- 53 exact duplicates
- 87 perceptual duplicates

The manifest shows these are not random copies scattered without context. They cluster around the same source stem, such as Multispectral_benign (1) and related rotated/sharpened variants. This pattern is exactly what the source-aware split logic is meant to control.

The project’s code in [MSI_GPU_Training_Package/train_breast_cancer.py](../MSI_GPU_Training_Package/train_breast_cancer.py) derives a group_id from the source stem before splitting. That is a reasonable strategy for a dataset whose images are explicitly transformed variations of a small number of source lesions.

However:
- the presence of many transformed variants means the effective sample count is far smaller than the nominal 806 images;
- the dataset is still not a clean collection of independent lesion instances;
- it should be treated as a grouped source-derived dataset rather than a fully independent patient sample set.

### Histopathology findings

The histopathology dataset has:
- 51 unique groups
- zero exact duplicates
- zero perceptual duplicates

This is a much cleaner profile. The naming pattern suggests slide-level grouping, which is sensible for BreaKHis-style data. There is no evidence here of duplicate leakage across splits from the saved manifest summaries.

## 3. Baseline review findings

### Ultrasound baseline

The preserved baseline in [msi_results_full/comparison_summary.csv](../msi_results_full/comparison_summary.csv) reports:
- Accuracy: 0.8614
- Macro F1: 0.8608
- Malignant sensitivity: 0.9059
- ROC-AUC: 0.9089

This is a strong result relative to the other modalities and is consistent with the model learning a clear pattern.

Key caveat:
- the effective independent sample size is likely smaller than the nominal count due to augmented copies and source-derived variants;
- therefore the reported performance is promising but not a fully independent patient-level estimate.

### Histopathology baseline

The preserved baseline reports:
- Accuracy: 0.7600
- Macro F1: 0.7596
- Malignant precision: 0.8738
- ROC-AUC: 0.8594

This is a moderate-to-strong result for a slide-based histopathology task and appears more stable than the chest X-ray subset.

The group structure and lack of duplicate artifacts make this model comparatively more defensible than the chest X-ray model.

## 4. Interpretability review from saved Grad-CAM and masking outputs

The saved Grad-CAM and masking evidence under [msi_feature_audit](../msi_feature_audit) is exploratory only. The audit summary explicitly warns that Grad-CAM is a saliency visualization aid and not proof of medically correct feature usage.

### Ultrasound

The ultrasound Grad-CAM outputs appear to highlight bright, structured regions in the image rather than stable anatomy-only regions. A few examples show strong activation concentrated in the background or border-like structures, and the masking summary includes border and center perturbation checks.

The masking summary shows that some predictions are sensitive to border masking or center masking. This is not proof of medically valid reasoning, but it does suggest that some decisions may be influenced by image structure outside the core lesion area.

Interpretation:
- There is plausible evidence of background or artifact sensitivity.
- This does not justify a claim that the model is following correct medical reasoning.

### Histopathology

The histopathology Grad-CAM outputs show a more tissue-like pattern than the ultrasound case, but the preserved notes still caution against over-interpretation. The visualizations are not a medical proof of diagnostic correctness.

The masking results suggest some probability changes when central or border regions are masked, but this is still only a sensitivity test. It is a useful diagnostic signal for model behavior, not causality proof.

Interpretation:
- The histopathology model appears more likely to focus on tissue structure than on obvious background artifacts.
- However, the audit remains qualitative and does not establish clinical validity.

## 5. Recommended fixes, in priority order

1. Keep the current split logic for the audit stage, but document the source-group caveat explicitly.
   - Ultrasound should be labeled as source-aware and likely leakage-resistant rather than fully patient-level validated.
   - Histopathology should remain the more stable dataset in this comparison.

2. Do not treat the chest X-ray subset as clinically reliable.
   - Keep it isolated from the main comparison narrative.
   - Mark its poor performance and uncertain provenance as exploratory only.

3. If a future improvement stage is approved, prefer the following cleanup order:
   - tighten ultrasound source-group validation
   - confirm there are no split-crossing variants after any final grouping decision
   - review histopathology group assignments against slide IDs
   - do not modify model architecture before the split and duplicate review is completed

4. For any later experiment stage, keep the test set untouched until the split and leakage review is closed.

## 6. Final conclusion for the audit-only phase

This audit supports the following summary:

- Ultrasound: Uncertain but likely acceptable with a source-group caveat. The dataset contains many transformed variants and should be treated as source-derived rather than fully independent patient data.
- Histopathology: Pass with caveat. The slide-based grouping is coherent and the duplicate audit is clean.
- Chest X-ray: Not part of the reliable primary comparison; keep separate as exploratory and not clinically valid.

The available evidence does not justify retraining or architecture changes at this point. The next step, if approved, should be a very small validation-only experiment only for the reliable modalities, after the split review has been fully documented.
