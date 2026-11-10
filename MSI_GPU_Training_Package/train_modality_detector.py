"""Train and evaluate a separate three-way image modality verifier.

Uses the preserved group-based split manifests and does not read or modify any
cancer-model checkpoint. Run from the project root with the project Python env.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "modality_detector"
KNOWN_CLASSES = ("ultrasound", "histopathology", "chest_xray")
CLASSES = KNOWN_CLASSES + ("other",)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ManifestImages(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASS_TO_INDEX[row.modality]


def extract_features(backbone: torch.nn.Module, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(ManifestImages(frame), batch_size=64, shuffle=False, num_workers=0)
    vectors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for images, targets in loader:
            features = backbone(images.to(DEVICE)).cpu().numpy()
            vectors.append(features)
            labels.append(targets.numpy())
    return np.concatenate(vectors), np.concatenate(labels)


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    # Read each modality manifest, then combine while retaining the frozen splits.
    parts = [pd.read_csv(ROOT / "msi_results_full" / f"{name}_splits.csv") for name in KNOWN_CLASSES]
    data = pd.concat(parts, ignore_index=True)
    if set(data.modality.unique()) != set(KNOWN_CLASSES):
        raise RuntimeError("Unexpected modalities in split manifests")
    if not data.path.map(lambda p: Path(p).is_file()).all():
        missing = data.loc[~data.path.map(lambda p: Path(p).is_file()), "path"].head().tolist()
        raise FileNotFoundError(f"Manifest image path missing: {missing}")

    weights = models.ResNet18_Weights.DEFAULT
    full_model = models.resnet18(weights=weights)
    backbone = torch.nn.Sequential(*list(full_model.children())[:-1], torch.nn.Flatten()).to(DEVICE).eval()

    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    # A small, explicitly labeled unrelated class uses non-medical project plots.
    # Keep the three folders disjoint across train/validation/test.
    chart_folders = {
        "train": [ROOT / "outputs"],
        "validation": [ROOT / "chest_xray_experiments"],
        "test": [ROOT / "smoke_test_outputs"],
    }
    seen_hashes: set[str] = set()
    other_parts: dict[str, pd.DataFrame] = {}
    for split, folders in chart_folders.items():
        records = []
        for folder in folders:
            for path in folder.glob("*.png"):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                records.append({"path": str(path), "modality": "other"})
        other_parts[split] = pd.DataFrame(records)

    for split in ("train", "validation", "test"):
        known_frame = data[data.split == split]
        frame = pd.concat([known_frame, other_parts[split]], ignore_index=True).reset_index(drop=True)
        features[split] = extract_features(backbone, frame)
        print(f"Extracted {split}: {len(frame)} images", flush=True)

    x_train, y_train = features["train"]
    x_validation, y_validation = features["validation"]
    x_test, y_test = features["test"]
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
    classifier.fit(x_train, y_train)

    val_probabilities = classifier.predict_proba(x_validation)
    # Choose the lowest single threshold that accepts no wrong modality in validation.
    wrong_scores = val_probabilities.copy()
    wrong_scores[np.arange(len(y_validation)), y_validation] = 0.0
    threshold = float(wrong_scores.max() + 1e-6)
    threshold = min(threshold, 0.999999)

    test_probabilities = classifier.predict_proba(x_test)
    test_predictions = test_probabilities.argmax(axis=1)
    test_wrong_scores = test_probabilities.copy()
    test_wrong_scores[np.arange(len(y_test)), y_test] = 0.0
    accepted_correct = (test_probabilities[np.arange(len(y_test)), y_test] >= threshold)
    false_accept = (test_wrong_scores.max(axis=1) >= threshold)
    test_report = {
        "classes": list(CLASSES),
        "threshold_selected_on_validation": threshold,
        "validation_images": int(len(y_validation)),
        "validation_wrong_modality_accepts_at_threshold": int((wrong_scores.max(axis=1) >= threshold).sum()),
        "test_images": int(len(y_test)),
        "test_accuracy_top1": float(accuracy_score(y_test, test_predictions)),
        "test_confusion_matrix": confusion_matrix(y_test, test_predictions, labels=range(4)).tolist(),
        "test_classification_report": classification_report(
            y_test, test_predictions, labels=range(4), target_names=CLASSES, output_dict=True, zero_division=0
        ),
        "test_known_modality_acceptance_rate": float(accepted_correct[y_test < 3].mean()),
        "test_other_rejection_count": int(((test_probabilities[:, 3] >= threshold) & (test_predictions == 3)).sum()),
        "test_other_rejection_rate": float((test_predictions[y_test == 3] == 3).mean()),
        "test_correct_modality_acceptance_count": int(accepted_correct.sum()),
        "test_correct_modality_acceptance_rate": float(accepted_correct.mean()),
        "test_wrong_modality_acceptance_count": int(false_accept.sum()),
        "test_wrong_modality_acceptance_rate": float(false_accept.mean()),
        "unrelated_image_evaluation": {
            "test_type": "held-out non-medical charts from project experiment outputs",
            "test_count": int((y_test == CLASS_TO_INDEX["other"]).sum()),
            "limitations": "This small chart set does not represent all unrelated images or establish general open-set rejection.",
        },
    }

    # Restore the torchvision model and install the independent trained modality head.
    detector = models.resnet18(weights=weights)
    detector.fc = torch.nn.Linear(detector.fc.in_features, len(CLASSES))
    with torch.no_grad():
        detector.fc.weight.copy_(torch.from_numpy(classifier.coef_.astype(np.float32)))
        detector.fc.bias.copy_(torch.from_numpy(classifier.intercept_.astype(np.float32)))
    OUT.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": detector.cpu().state_dict(),
        "classes": CLASSES,
        "acceptance_threshold": threshold,
        "threshold_selection": "maximum validation probability assigned to any wrong modality, plus 1e-6",
        "backbone_weights": str(weights),
    }, OUT / "modality_resnet18.pt")
    (OUT / "evaluation.json").write_text(json.dumps(test_report, indent=2))
    print(json.dumps(test_report, indent=2))


if __name__ == "__main__":
    main()
