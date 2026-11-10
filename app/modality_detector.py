from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
DETECTOR_PATH = ROOT / "modality_detector" / "modality_resnet18.pt"
MODALITY_NAMES = ("ultrasound", "histopathology", "chest_xray", "other")


class ModalityDetector:
    """Separate modality classifier; its output is never a cancer class."""

    def __init__(self, checkpoint: dict[str, Any], device: torch.device):
        self.device = device
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(MODALITY_NAMES))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device).eval()
        self.threshold = float(checkpoint["acceptance_threshold"])
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def classify(self, image: Image.Image) -> tuple[str | None, float]:
        accepted, confidence, _raw_class = self.classify_many([image])[0]
        return accepted, confidence

    @torch.inference_mode()
    def classify_many(self, images: list[Image.Image]) -> list[tuple[str | None, float, str]]:
        if not images:
            return []
        tensors = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        probabilities = torch.softmax(self.model(tensors), dim=1).cpu()
        results = []
        for row in probabilities:
            score, index = row.max(dim=0)
            confidence = float(score)
            raw_class = MODALITY_NAMES[int(index)]
            accepted = raw_class if confidence >= self.threshold and raw_class != "other" else None
            results.append((accepted, confidence, raw_class))
        return results


_CACHE: dict[str, ModalityDetector] = {}


def load_modality_detector(device: torch.device) -> ModalityDetector:
    key = str(device)
    if key not in _CACHE:
        if not DETECTOR_PATH.exists():
            raise FileNotFoundError(f"Modality detector checkpoint missing: {DETECTOR_PATH}")
        checkpoint = torch.load(DETECTOR_PATH, map_location=device, weights_only=False)
        if tuple(checkpoint.get("classes", ())) != MODALITY_NAMES:
            raise ValueError("Modality detector class metadata is invalid")
        _CACHE[key] = ModalityDetector(checkpoint, device)
    return _CACHE[key]
