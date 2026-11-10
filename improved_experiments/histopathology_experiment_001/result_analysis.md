# Histopathology experiment read-only result analysis

## Scope

This review is read-only. It inspects the newly generated artifact set for the approved histopathology experiment without retraining, changing thresholds, modifying splits, or starting ultrasound work.

Artifacts inspected:

- [config.json](config.json)
- [history.csv](history.csv)
- [metrics.json](metrics.json)
- [confusion_matrix.png](confusion_matrix.png)
- [roc_curve.png](roc_curve.png)
- [best_checkpoint.pt](best_checkpoint.pt)
- [../../msi_results_full/histopathology_history.csv](../../msi_results_full/histopathology_history.csv)
- [../../msi_results_full/histopathology_test_metrics.json](../../msi_results_full/histopathology_test_metrics.json)
- [../../MSI_GPU_Training_Package/train_breast_cancer.py](../../MSI_GPU_Training_Package/train_breast_cancer.py)

## 1. Verified configuration and checkpoint metadata

The new run configuration in [config.json](config.json) states:

- modality: histopathology
- dataset root: c:\Users\asus\OneDrive\Desktop\Breast Cncer Project\MultiModel Breast Cancer MSI Dataset
- architecture: ResNet-18 pretrained on ImageNet
- learning rate: 0.0005
- freeze_epochs: 2
- epochs: 8
- patience: 3
- batch size: 16
- seed: 42
- validation metric: macro_f1
- checkpoint selection: best validation macro F1
- test set used for final evaluation only: true

The checkpoint metadata in [best_checkpoint.pt](best_checkpoint.pt) confirms:

- epoch = 7
- val_macro_f1 = 0.9075966615568046
- val_accuracy = 0.9078341013824884
- val_roc_auc = 0.9703324808184144
- modality = histopathology
- class_map = {benign: 0, malignant: 1}
- pos_name = malignant
- neg_name = benign

This matches the class mapping used by the project code in [../../MSI_GPU_Training_Package/train_breast_cancer.py](../../MSI_GPU_Training_Package/train_breast_cancer.py): benign = 0 and malignant = 1.

## 2. Dataset, class mapping, and evaluation definitions

The project code uses a single dataset root and the same class map for histopathology in both the baseline and the new run:

- benign label = 0
- malignant label = 1
- positive class = malignant
- probability threshold for classification = 0.5
- final evaluation is performed after checkpoint selection by validation macro F1

The code path that enforces this is visible in [../../MSI_GPU_Training_Package/train_breast_cancer.py](../../MSI_GPU_Training_Package/train_breast_cancer.py):

- MODALITIES['histopathology']['classes'] = {'benign': 0, 'malignant': 1}
- evaluate_model uses torch.softmax(logits, dim=1)[:, 1] and compares probability >= 0.5
- best checkpoint is saved only when validation macro F1 improves, and the test set is evaluated only after the loop ends

This means the class mapping and evaluation definitions are consistent with the baseline pipeline. However, the saved split files are not identical between the original baseline and the new run:

- Baseline counts: train = 796, val = 200, test = 250
- New run counts: train = 815, val = 217, test = 214

Therefore, the new experiment did not reuse the exact same saved split assignments as the original baseline run. It used the same dataset root and class definitions, but it is not a bit-for-bit identical split replication of the baseline.

## 3. Curve analysis: overfitting and epoch selection

The validation curve in [history.csv](history.csv) shows the following:

- Epoch 1: val_macro_f1 = 0.6244
- Epoch 2: 0.7889
- Epoch 3: 0.8433
- Epoch 4: 0.8802
- Epoch 5: 0.8893
- Epoch 6: 0.8611
- Epoch 7: 0.9076
- Epoch 8: 0.8523

The training metrics continue to improve through the end of training, but validation macro F1 peaks at epoch 7 and declines at epoch 8. Validation loss also rises from 0.2209 at epoch 7 to 0.3381 at epoch 8, which is a classic sign of mild overfitting after the validation optimum.

This makes epoch 7 the correct checkpoint according to the project’s selection rule: save the checkpoint with the best validation macro F1, not the final epoch.

## 4. Confusion matrix and per-class precision/recall

The confusion matrix in [metrics.json](metrics.json) is:

- true benign / predicted benign = 55
- true benign / predicted malignant = 5
- true malignant / predicted benign = 68
- true malignant / predicted malignant = 86

This yields:

- benign recall = 55 / (55 + 5) = 0.9167
- benign precision = 55 / (55 + 68) = 0.4472
- malignant recall = 86 / (86 + 68) = 0.5584
- malignant precision = 86 / (86 + 5) = 0.9451

The model is therefore strong at recognizing benign cases once they are predicted as benign, but it misses a large share of true malignant cases. The low malignant sensitivity is the main weakness. In practical terms, the classifier is conservative about predicting malignant, favoring high precision at the cost of lower recall.

This pattern is consistent with the fixed probability threshold of 0.5 and the project’s evaluation rule, which does not tune threshold on the test set.

## 5. Positive-class definition and threshold verification

The code specifies:

- positive class = malignant
- label 1
- predicted probability = softmax(logits)[:, 1]
- threshold = 0.5

The checkpoint metadata also confirms:

- pos_name = malignant
- pos_label = 1

The same threshold is used for all predictions in the training/evaluation code. There is no threshold optimization on the test set. This is consistent with the requirement not to tune the test set for model choice or decision threshold.

## 6. Validation-test gap and possible explanations

The new experiment reports:

- validation macro F1 = 0.9076 at epoch 7
- test macro F1 = 0.6516

The gap is substantial. The most likely explanations, based on the verified artifacts, are:

1. Different split composition from the original baseline
   - The original baseline and this new run do not share the same split counts or exact saved split assignment files.
   - This means the validation and test cases may not align in the same way, so a direct comparison of absolute numbers should be interpreted carefully.

2. Fixed threshold and class imbalance
   - The model prioritizes malignant precision and specific benign recall, but it misses many malignant examples at the default 0.5 threshold.
   - This is visible in the confusion matrix and the reported sensitivity value of 0.5584.

3. No evidence of direct source leakage from the histopathology audit
   - The audit in [../../audit_review/report.md](../../audit_review/report.md) reports clean histopathology duplicate checks and slide-like grouping behavior.
   - This reduces the likelihood that the gap is caused by obvious duplicate leakage in histopathology.

4. Validation and test were not used interchangeably
   - The training loop only saves the best validation checkpoint and evaluates the test set after training ends.
   - The test set was not used to select epochs, hyperparameters, or checkpoint decisions.

The available evidence does not support a claim that the gap is caused by source grouping or duplicates in the histopathology set. It is more likely a consequence of a different split composition plus the unoptimized 0.5 decision threshold and the model’s recall tradeoff.

## 7. Test-set independence and selection integrity

The script confirms the intended model-selection workflow:

- save the best checkpoint when validation macro F1 improves
- stop based on validation metric under patience
- only evaluate the test set after the best checkpoint is selected

This means the test set was not used for epoch selection, hyperparameter choice, or checkpoint selection. The test set was kept untouched during model selection, as required.

## 8. Limitations and recommended next steps

### Verified limitations

- The saved split files for the original baseline and the new experiment are not identical, so they are not the same split assignment set.
- The new experiment shows strong validation performance but weaker test-set F1, which is a real outcome that should not be hidden.
- The model is thresholded at 0.5 and not tuned on the validation or test set for this read-only pass.
- The audit supports histopathology as the most stable modality, but it does not prove perfect clinical validity.

### Recommended next steps if approval is granted later

1. Keep the histopathology experiment as a single validation-driven improvement, not as a broad retraining wave.
2. Evaluate a small validation-only threshold sweep, if needed, but do not tune on the test set.
3. Compare calibration and sensitivity/precision tradeoffs before any further architecture changes.
4. Do not start ultrasound training until the next approval step is explicitly provided.

## 9. Bottom line

The new histopathology experiment achieved a verified best validation macro F1 of 0.9076 at epoch 7 and a corresponding validation AUC of 0.9703. The validation performance is clearly better than the original baseline, but the test-set Macro F1 of 0.6516 is materially lower, and the low malignant sensitivity is the main operational weakness.

The strongest evidence-based interpretation is:

- the model improved on validation under the project’s selection rule,
- the new run is not a direct re-use of the original baseline split assignments,
- the test set remained untouched during model selection,
- the low sensitivity is primarily explained by the fixed threshold and the class-distribution tradeoff rather than by a clear evidence of data leakage in the histopathology pipeline.

This is a valid read-only analysis of the verified artifacts, and no further training or threshold optimization was performed as part of this review.
