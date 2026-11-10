"""Evaluate a saved modality detector on independent public breast-image sets.

Expected external files in %TEMP%/breast_modality_external_eval:
- BUSI.parquet (CC BY 4.0)
- breast-histology-test.parquet (CC0)

This script does not train or alter either detector or cancer checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.modality_detector import load_modality_detector

EVAL_DIR = Path(os.environ["TEMP"]) / "breast_modality_external_eval"
MAX_HISTO_IMAGES = 1000


def get_image_bytes(value: object) -> bytes:
    if isinstance(value, dict):
        return value["bytes"]
    if hasattr(value, "getvalue"):
        return value.getvalue()
    raise TypeError(f"Unexpected parquet image value: {type(value)!r}")


def evaluate_parquet(path: Path, expected: str, detector, *, limit: int | None = None) -> dict:
    frame = pd.read_parquet(path)
    if limit is not None and len(frame) > limit:
        frame = frame.sample(n=limit, random_state=2026).reset_index(drop=True)

    project_hashes = set(
        pd.read_csv(ROOT / "dataset_manifests" / "complete_dataset_manifest.csv")["sha256"].dropna().astype(str)
    )
    unique_images: dict[str, Image.Image] = {}
    unique_row_indices: list[int] = []
    overlap_hashes: set[str] = set()
    duplicate_rows = 0
    for row_index, record in enumerate(frame.image):
        raw = get_image_bytes(record)
        digest = hashlib.sha256(raw).hexdigest()
        if digest in project_hashes:
            overlap_hashes.add(digest)
            continue
        if digest in unique_images:
            duplicate_rows += 1
            continue
        with Image.open(BytesIO(raw)) as image:
            unique_images[digest] = image.convert("RGB").copy()
        unique_row_indices.append(row_index)

    items = list(unique_images.values())
    predictions = []
    for start in range(0, len(items), 32):
        predictions.extend(detector.classify_many(items[start:start + 32]))

    accepted = sum(result[0] == expected for result in predictions)
    confident_wrong_class = sum(result[0] is not None and result[0] != expected for result in predictions)
    cross_supported_false_accepts = sum(
        expected in {"ultrasound", "histopathology"}
        and result[0] in {"ultrasound", "histopathology"}
        and result[0] != expected
        for result in predictions
    )
    rejected_as_unavailable_xray = sum(
        result[0] == "chest_xray" for result in predictions
    )
    rejected = sum(result[0] is None for result in predictions)
    top1_correct = sum(result[2] == expected for result in predictions)
    raw_classes: dict[str, int] = {}
    for _accepted, _confidence, raw_class in predictions:
        raw_classes[raw_class] = raw_classes.get(raw_class, 0) + 1
    n = len(predictions)
    confidently_misclassified = [
        {
            "sampled_row_index": unique_row_indices[index],
            "predicted_modality": result[2],
            "confidence": result[1],
        }
        for index, result in enumerate(predictions)
        if result[2] not in (expected, "other") and result[1] >= detector.threshold
    ]
    threshold_tradeoff = []
    for threshold in (0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999):
        wrong_confident = sum(
            result[2] not in (expected, "other") and result[1] >= threshold
            for result in predictions
        )
        cross_supported_confident = sum(
            expected in {"ultrasound", "histopathology"}
            and result[2] in {"ultrasound", "histopathology"}
            and result[2] != expected
            and result[1] >= threshold
            for result in predictions
        )
        expected_confident = sum(
            result[2] == expected and result[1] >= threshold
            for result in predictions
        )
        threshold_tradeoff.append({
            "threshold": threshold,
            "correct_modality_accepted": expected_confident,
            "wrong_modality_confident": wrong_confident,
            "cross_supported_modality_confident": cross_supported_confident,
        })
    return {
        "source_file": path.name,
        "expected_modality": expected,
        "rows_sampled": int(len(frame)),
        "exact_duplicate_rows_removed": duplicate_rows,
        "images_overlapping_project_manifest": len(overlap_hashes),
        "unique_independent_images_evaluated": n,
        "top1_modality_accuracy": top1_correct / n if n else None,
        "accepted_as_expected_at_detector_threshold": accepted,
        "confident_wrong_class_predictions": confident_wrong_class,
        "cross_supported_modality_false_accepts": cross_supported_false_accepts,
        "rejected_as_chest_xray_unavailable": rejected_as_unavailable_xray,
        "rejected_or_classified_other": rejected,
        "raw_detector_classes": raw_classes,
        "confident_wrong_modality_examples": confidently_misclassified,
        "threshold_tradeoff": threshold_tradeoff,
    }


def evaluate_project_holdout(detector) -> dict:
    rows = []
    for modality in ("ultrasound", "histopathology", "chest_xray"):
        frame = pd.read_csv(ROOT / "msi_results_full" / f"{modality}_splits.csv")
        rows.extend((modality, Path(path)) for path in frame.loc[frame.split == "test", "path"])
    predictions = []
    for start in range(0, len(rows), 32):
        batch_rows = rows[start:start + 32]
        images = []
        for _expected, path in batch_rows:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        predictions.extend(detector.classify_many(images))
    thresholds = []
    for threshold in (detector.threshold, 0.85, 0.90, 0.95, 0.98):
        correct = sum(
            raw == expected and confidence >= threshold
            for (expected, _path), (_accepted, confidence, raw) in zip(rows, predictions)
        )
        wrong = sum(
            raw != expected and raw != "other" and confidence >= threshold
            for (expected, _path), (_accepted, confidence, raw) in zip(rows, predictions)
        )
        thresholds.append({"threshold": threshold, "correct_modality_accepted": correct, "wrong_modality_confident": wrong})
    return {"images": len(rows), "threshold_tradeoff": thresholds}


def main() -> None:
    detector = load_modality_detector(torch.device("cpu"))
    report = {
        "detector_threshold": detector.threshold,
        "evaluation_type": "external datasets not used in detector training, validation, or test splits",
        "project_split_holdout_reference": evaluate_project_holdout(detector),
        "results": [
            evaluate_parquet(EVAL_DIR / "BUSI.parquet", "ultrasound", detector),
            evaluate_parquet(
                EVAL_DIR / "breast-histology-test.parquet",
                "histopathology",
                detector,
                limit=MAX_HISTO_IMAGES,
            ),
        ],
        "limitations": [
            "BUSI includes known source quality caveats and does not represent all ultrasound devices.",
            "The histology test split contains 50x50 patches, not whole-slide pathology images.",
            "This evaluation tests modality recognition only, not cancer classifier accuracy.",
        ],
    }
    output = ROOT / "modality_detector" / "external_evaluation.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
