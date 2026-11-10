# Histopathology split comparison report

## Executive summary

I reviewed the preserved original baseline split file and the approved histopathology experiment split file without retraining or changing any artifacts.

The evidence shows that the two runs do not use the same split assignments:

- Baseline split counts: train = 796, validation = 200, test = 250
- New experiment split counts: train = 815, validation = 217, test = 214
- Both files contain the same set of 1,246 image paths, but the assignment of those paths to train/validation/test differs substantially.

This means the new experiment cannot be treated as a directly comparable rerun of the exact original baseline split. Any performance comparison should therefore be interpreted as a comparison across different held-out sets, not as a matched re-evaluation of the same data split.

## Verified evidence

### 1) The same paths exist, but split membership differs

The preserved CSVs both contain 1,246 unique image paths.

Per-split overlap between the baseline and the new run was:

- Train overlap: 513 of 796 baseline train images
- Validation overlap: 39 of 200 baseline validation images
- Test overlap: 39 of 250 baseline test images

This is a strong sign that the split assignments were regenerated differently, even though the same dataset and image list were used.

### 2) Group structure is not consistent across the two split files

The baseline split file reports 1,246 unique group IDs, meaning nearly every row has a unique group value.

The new experiment split file reports only 51 unique groups, with a maximum group size of 62 images.

This is not a minor difference: it indicates the group field in the new run reflects a repeated group identity (such as a slide/patient group), while the baseline file effectively did not use repeated group IDs in the same way.

The project code defines the grouping behavior in [MSI_GPU_Training_Package/train_breast_cancer.py](../../MSI_GPU_Training_Package/train_breast_cancer.py):

- `extract_group_id` hashes a class + source identifier
- `breakhis_slide` is the histopathology strategy
- the split is generated with `GroupShuffleSplit` on the group column

That means the group membership is part of the actual split-generation logic, and if the saved files do not share consistent group semantics, they are not the same split.

### 3) The preserved baseline split is not reconstructable from the current run artifacts alone

The current saved artifacts do not include enough information to recover the exact original baseline split assignment with a high-confidence reproduction.

The reason is straightforward:

- the original baseline CSV and the new run CSV have different group distributions,
- the new run group semantics are not identical to the old ones,
- the project does not retain a single canonical manifest and seed log proving the original split assembly was exactly replicated.

Because of that, the exact original baseline split cannot be stated as identical or reconstructed from the present files alone.

## Why this matters for comparison

The new histopathology experiment was validated using the project’s selection rule (best validation macro F1), and the validation result was strong. However, because the held-out sets differ between the baseline and this run, the comparison is not a fair like-for-like test of the same evaluation set.

In practical terms:

- the validation improvement may still reflect genuine model improvement,
- but it cannot be treated as a direct head-to-head comparison on the original baseline splits,
- and the test-set metrics should not be interpreted as a matched re-test of the exact same data partition.

## Recommended reproducible evaluation plan

If a future experiment is approved, the safest reproducible standard is:

1. Save the exact dataset manifest used for the run, not just the model outputs.
2. Save the split CSV and the exact `group` generation code/path configuration with the run.
3. Record the dataset hash or manifest checksum before training.
4. Keep the same random seed, same group strategy, and same split logic across all experiments meant to be compared.
5. Perform all model selection and threshold tuning with validation only; never use the test set to choose the model or threshold.

This makes later model comparisons honest and reproducible.

## Final conclusion

The strongest evidence-based conclusion is:

- the original baseline and the approved histopathology experiment were not evaluated on the same saved split assignments,
- therefore, the two runs are not directly comparable as matched experiments,
- and the original baseline split cannot be confidently reconstructed from the preserved artifacts alone.

This is a read-only finding, and no retraining or split modification was performed while preparing it.
