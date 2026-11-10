# Results analysis

Metric values are fractions in `03_VERIFIED_RESULTS.csv`. Baseline ultrasound test metrics are 0.8614 accuracy, 0.8608 macro F1, 0.9059 malignant recall and 0.9089 ROC-AUC (n=166). Additional ultrasound is 0.8373 accuracy, 0.8352 macro F1, 0.7059 recall, 0.9677 precision, 0.9753 specificity and 0.9497 AUC (n=166). It trades lower accuracy/F1/sensitivity for higher precision, specificity and AUC; do not describe it as universally better. Metrics record equal n and split metadata, but confirm split-file identity and protocol before inference of fair comparison.

Baseline histopathology: n=250, accuracy 0.7600, macro F1 0.7596, malignant precision 0.8738, recall 0.6569, specificity 0.8850, AUC 0.8594. Later experiment test: n=214, accuracy 0.6589, macro F1 0.6516, precision 0.9451, recall 0.5584, specificity 0.9167, AUC 0.8190. Different split; do not directly rank. Its reported best validation Macro F1 0.9076 is a validation result, not test performance.

Modality detector external evaluation reports ultrasound 711/779 top-1 (91.3%), 490/779 accepted (62.9%), histopathology 920/1000 top-1 (92.0%), 697/1000 accepted (69.7%) at threshold 0.7413. Accuracy and acceptance coverage are distinct. External records show errors; this is not universal rejection assurance.

Chest X-ray baseline saved metrics are near chance (accuracy .49, AUC .5468); later held-out winner artifact reports .57 accuracy and .6034 AUC, which differs from earlier preliminary figures. Both are exploratory; source audit notes questionable provenance/appearance. Keep outside primary breast-modality claims.
