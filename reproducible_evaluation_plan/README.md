# Reproducible histopathology evaluation plan

This directory is a read-only evaluation setup for the histopathology dataset. It does not overwrite original manifests or training outputs.

## Grouping evidence

The preserved manifest at `dataset_manifests/histopathology_manifest.csv` contains:
- 1,246 total images
- 623 benign / 623 malignant
- 51 unique `source_id` values
- 51 unique `group_id` values

The `source_id` values are filename-derived prefixes such as `Multispectral_SOB_B_A-14-22549AB` and are the strongest available slide/specimen grouping signal in the preserved metadata. This supports a conservative slide-level split strategy, but it is not a separate explicit patient-ID file; we therefore do not claim patient-level independence beyond the evidence in the manifest.

## Proposed strategy

- Group key: `source_id`
- Split method: per-class `GroupShuffleSplit`
- Random seed: 42 for the 15% test split
- Validation split: seed 43 for the internal 70/15/15 partitioning
- Objective: ensure that no slide/specimen prefix crosses train, validation, or test folds

## Resulting counts

- Train: 845 total (449 benign, 396 malignant)
- Validation: 237 total (98 benign, 139 malignant)
- Test: 164 total (76 benign, 88 malignant)

## Leakage checks

- `source_id` overlaps between splits: 0
- No related source appears in multiple folds
- Exact duplicate path rows are not present in the preserved manifest

## File list

- `histopathology_proposed_split_manifest.csv`
- `split_config.json`
- `manifest_checksum.sha256`
- `README.md`

## Manifest checksum

The training data manifest used for this split was hashed with SHA256 and saved to `manifest_checksum.sha256`.
