# CheXpert Replacement Dataset Plan

Status: non-destructive planning only. No dataset download, no file deletion, no retraining, and no overwrite of current project outputs.

## 1. Verified source and dataset facts

The replacement candidate is CheXpert from the official Stanford source:

- https://stanfordmlgroup.github.io/competitions/chexpert/

From the official dataset description and paper summary:

- CheXpert is a large chest X-ray dataset collected from Stanford Hospital and associated radiology reports.
- It contains 224,316 chest radiographs from 65,240 patients.
- The dataset is intended for automated chest radiograph interpretation, not for a binary “Normal vs Malignant” task.
- Labels are extracted from radiology reports using the CheXpert labeler.
- The official label schema is multi-label: 14 observations, each represented in report-derived form as: positive, negative, uncertain, or unmentioned/blank.
- The dataset design explicitly reports uncertainty labels and uses patient-level structure; the official source also emphasizes test-set evaluation with radiologist annotations.
- The official page states that the dataset is available via the Stanford AIMI / ReDivis hosting path, and the test-set labels are also linked from the official GitHub repository for the test-set labels.

Important: I have not downloaded the full dataset or modified the current project files. The goal here is to verify access and task fit before any large download or model migration.

## 2. Access requirements and manual verification needed

The official site points to the dataset hosted on the Stanford AIMI / ReDivis distribution path and to the test-set label repository.

Before downloading, the user must complete the following manual steps on the official channels:

1. Open the CheXpert official page:
   - https://stanfordmlgroup.github.io/competitions/chexpert/
2. Follow the official Stanford AIMI / ReDivis link on that page to access the dataset.
3. Review and accept the applicable dataset access conditions, usage terms, and any required registration or approval steps.
4. Confirm the exact archive name, folder structure, and any split files available on the official site before download.
5. If the dataset is not publicly downloadable without registration, complete the required account/approval process before proceeding.

I do not proceed with any download or storage change until the user confirms the required access step has been completed.

## 3. Label definitions and why this is not a direct Normal-vs-Malignant replacement

CheXpert labels are not designed as a binary cancer task. The official documentation defines a multi-label observation schema with labels such as:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Pleural Effusion
- and other findings relevant to chest radiography

The dataset uses structured report-derived labels, with per-observation logic such as:

- 1 = positive
- 0 = negative
- -1 = uncertain
- blank or unmentioned = no explicit mention

This is fundamentally different from the project’s current chest X-ray label scheme, which is:

- Normal
- Malignant

Therefore, CheXpert is not a drop-in replacement for the current chest X-ray experiment. The task must be redesigned to match the dataset’s documented labeling rules.

## 4. Proposed binary educational task (defensible, but narrower than the original)

A defensible educational binary task can be created, but it must be explicitly defined and not mislabeled as a breast cancer task.

### Recommended binary framing (for education and reproducible experiments)

Use a single pathology label as the positive class, for example:

- Pleural Effusion present vs absent
- Atelectasis present vs absent
- Cardiomegaly present vs absent

This is a clean educational setting because the label exists in the official schema and is clinically interpretable.

### Why this is acceptable

- The official dataset documentation explicitly describes these findings and their rules.
- The task is single-label and binary after mapping from the original observation label.
- It can be implemented using patient-level grouping and validation-driven selection.

### Example mapping

For a selected observation such as Pleural Effusion:

- positive label = 1
- negative label = 0
- uncertain = -1

Handling uncertain labels must be decided before training:

- Option A: exclude uncertain cases from the binary task
- Option B: map uncertain to positive (U-Ones style)
- Option C: map uncertain to negative (U-Zeroes style)
- Option D: keep uncertain as a separate class in a tertiary formulation

For a classroom project, the best safe default is:

- include only definite positive and definite negative examples for the initial binary task,
- investigate uncertainty handling separately,
- and report the exact rule in the final experiment config.

This avoids silently inventing a label policy that is not documented in the official dataset.

## 5. Patient-level grouping and split strategy

CheXpert is patient-based and the official docs describe image counts and patient records. The split strategy must respect patient independence.

### Recommended split design

- Use patient ID or study ID as the grouping key.
- Keep all images from the same patient in a single split.
- Use a fixed random seed.
- Keep the final test set untouched until the end.
- Use only training data to compute any preprocessing statistics.

### Split logic

- Train / validation / test = 70 / 15 / 15 by patient or study group
- or 80 / 10 / 10 if the dataset is large enough and you need more validation stability
- Use deterministic, documented grouping and a fixed seed

### Limitation

Because the dataset metadata and the exact split file format are not yet downloaded or inspected locally, patient-level grouping must be verified on the official downloaded data before training begins.

## 6. Preprocessing and evaluation plan

### Preprocessing

- Load DICOM or converted images according to the official dataset package structure.
- Standardize image resolution and grayscale conversion if required by the dataset package.
- Use only training-data statistics for normalization.
- Apply augmentation only to training images; validation/test images remain unaugmented.
- Keep the project’s existing ResNet-18 transfer-learning workflow unless a new task requires a different objective.

### Evaluation metrics

For the binary educational task, use:

- Accuracy
- Macro precision, recall, and F1
- Positive-class precision and recall
- Specificity
- Sensitivity
- ROC-AUC
- Confusion matrix
- Training and validation loss curves

The positive class must be clearly defined in the JSON/CSV outputs. For example:

- positive = Pleural Effusion present
- negative = Pleural Effusion absent

The project should never report a chest X-ray model as a medical diagnostic system or as a substitute for the current breast cancer task.

## 7. Code changes required later (after approval)

The current project makes assumptions that are specific to the old chest X-ray subset and the binary Normal/Malignant task. Those assumptions must be decoupled before migration.

### Likely code changes

1. Add a separate dataset configuration block for CheXpert.
2. Replace the chest_xray class mapping with a task-specific mapping, e.g.:
   - class 0 = absent / negative
   - class 1 = present / positive
3. Replace the current chest_xray manifest builder with a CheXpert manifest parser that respects:
   - patient ID or study ID
   - image path
   - official label value
   - uncertainty handling rule
4. Update split logic to use patient-level grouping.
5. Update training objective to binary cross-entropy for the selected binary task.
6. Update evaluation metrics to report the chosen positive class explicitly.
7. Update documentation to clearly state:
   - old chest X-ray results remain experimental and unverified
   - new CheXpert task is a different task with different labels
8. Save all CheXpert experiments in a separate folder so the old experimental chest X-ray outputs remain preserved.

## 8. Proposed migration folder layout

Create a new, separate experiment folder at the project root, for example:

- replacement_xray_plan/
- improved_experiments/
- chexpert_experiments/

The existing outputs and metrics remain in their original folders and are not to be replaced or overwritten.

## 9. Decisions still requiring approval

Before any download or training begins, I need your approval on these items:

1. Which official CheXpert access route do you want to use?
   - direct Stanford AIMI / ReDivis page
2. Which binary educational task do you want to define?
   - Pleural Effusion vs absent
   - Atelectasis vs absent
   - Cardiomegaly vs absent
   - another derived binary task
3. How do we handle uncertain labels?
   - exclude uncertain labels
   - map uncertain to positive
   - map uncertain to negative
4. Do you want a single binary task, or a multi-label educational variant using several official CheXpert findings?

## 10. exact next step after your approval

After approval, the next safe step will be:

1. open the official Stanford AIMI / ReDivis link,
2. complete the required account/terms step,
3. verify the dataset folder structure and label files,
4. create a documented manifest in a new experimental folder,
5. define the exact binary label mapping,
6. only then run a small validation-only experiment.

No download, no deletion, and no model training will begin until this manual access verification is finished.
