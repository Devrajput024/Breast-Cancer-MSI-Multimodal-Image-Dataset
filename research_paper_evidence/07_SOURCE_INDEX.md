# Source index

- `README.md`: app scope and research-use disclaimer.
- `comparison_report.md`: baseline and chest X-ray narrative; cross-check against JSON.
- `MSI_GPU_Training_Package/train_breast_cancer.py`: labels, groups, splits, transforms/model/training.
- `MSI_GPU_Training_Package/evaluate_models.py`: evaluation pipeline.
- `MSI_GPU_Training_Package/predict.py`: standalone inference.
- `msi_results_full/*_test_metrics.json`, `*_splits.csv`: baseline results/splits.
- `improved_experiments/ultrasound/metrics.json`: additional ultrasound run.
- `improved_experiments/histopathology_experiment_001/`: separate later run/config/results.
- `dataset_manifests/dataset_validation_summary.json`, modality manifests: counts and duplicates.
- `modality_detector/evaluation.json`, `external_evaluation.json`: detector performance and external evaluation.
- `app/main.py`, `app/modality_detector.py`: inference/routing.
- `audit_review/report.md`, `split_integrity_summary.csv`; `msi_feature_audit/`: audit and interpretability.
- `chest_xray_experiments/winner_heldout_test_metrics.json`, `msi_results_chest_exp1/`: exploratory results.
- `tests/test_backend_modality_guard.py`, `smoke_test_outputs/`: software checks.

`source_extracts/` contains selected small code and result files with original directory recorded by its subfolder or filename. Large checkpoints and datasets are excluded.
