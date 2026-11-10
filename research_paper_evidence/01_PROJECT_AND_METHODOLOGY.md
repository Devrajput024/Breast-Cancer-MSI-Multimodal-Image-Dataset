# Project and methodology

## Verified scope and architecture
The workspace implements separate binary classifiers for ultrasound (benign/malignant) and histopathology (benign/malignant), plus exploratory chest X-ray (Normal/Malignant). `MSI_GPU_Training_Package/train_breast_cancer.py` specifies independent ResNet-18 pipelines, ImageNet normalization, and grouped splits; README describes ImageNet pretrained weights and two-stage frozen-head/layer4 fine-tuning. Saved output JSONs establish these as recorded results, but exact configs vary by experiment. Class 1 is malignant in breast tasks; evaluation JSONs label malignant positive.

## Data flow
Image folders/manifests -> derived source/slide groups -> train/validation/test partitions -> independent model training and best validation Macro-F1 checkpoint -> held-out evaluation. App validates file decoding/dimensions, checks modality with a detector, routes to ultrasound or histopathology classifier, and returns results/errors.

## Protocol details and gaps
The saved baseline counts are 806 ultrasound (512/128/166) and 1,246 histopathology (796/200/250). The later histopathology experiment has a separate 815/217/214 partition. Training code contains transform/split/optimizer logic; copied extracts are the precise record. README documents 12 epochs, 3 frozen warmup epochs, AdamW and early stopping, but these values should only be attributed to the documented run when matching its saved config. Input resize, augmentation, batch size, random seed, and exact pretrained-weight enum must be checked per run; where not identified in a specific output, report ?not found in inspected files.?

Evaluation reports accuracy, macro F1, malignant precision/recall, class-0 specificity and ROC-AUC. Thresholds are not consistently recorded in classifier result files. App uses an independent modality detector and does not offer chest X-ray inference. Grad-CAM and masking are exploratory sensitivity/visualization methods, not causal or clinical evidence. No cross-modality patient linkage is documented.
