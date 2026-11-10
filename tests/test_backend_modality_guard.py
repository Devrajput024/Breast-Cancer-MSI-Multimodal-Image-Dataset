from __future__ import annotations

import io
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.modality_detector import load_modality_detector
from MSI_GPU_Training_Package.train_breast_cancer import MODALITIES

client = TestClient(app)
UNVERIFIED = "Image modality could not be verified. Please upload a suitable ultrasound or histopathology image."
MISMATCH = "Selected modality does not match the uploaded image. Please upload a matching image."


def held_out_image(modality: str) -> Image.Image:
    frame = pd.read_csv(f"msi_results_full/{modality}_splits.csv")
    path = frame.loc[frame.split == "test", "path"].iloc[0]
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FixedCancerModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, tensor):
        self.calls += 1
        return torch.tensor([[0.0, 2.0]])


def fake_checkpoint():
    return {
        "class_map": {"benign": 0, "malignant": 1},
        "image_size": 224,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }


@pytest.mark.parametrize("modality", ["ultrasound", "histopathology"])
def test_valid_held_out_image_reaches_selected_cancer_classifier(modality: str):
    model = FixedCancerModel()
    with patch("app.main.load_model", return_value=(model, fake_checkpoint())):
        response = client.post(
            f"/api/predict?modality={modality}",
            files={"file": (f"{modality}.png", image_bytes(held_out_image(modality)), "image/png")},
        )
    assert response.status_code == 200, response.text
    assert response.json()["success"] is True
    assert response.json()["predicted_class"] == "malignant"
    assert model.calls == 1


@pytest.mark.parametrize(
    ("image_modality", "selected_modality"),
    [("ultrasound", "histopathology"), ("histopathology", "ultrasound")],
)
def test_cross_modality_upload_is_rejected_before_cancer_model(image_modality: str, selected_modality: str):
    with patch("app.main.load_model") as cancer_model:
        response = client.post(
            f"/api/predict?modality={selected_modality}",
            files={"file": ("image.png", image_bytes(held_out_image(image_modality)), "image/png")},
        )
    assert response.status_code == 400
    assert response.json()["error"] == MISMATCH
    cancer_model.assert_not_called()


@pytest.mark.parametrize("selected_modality", ["ultrasound", "histopathology"])
def test_chest_xray_is_rejected_before_cancer_model(selected_modality: str):
    with patch("app.main.load_model") as cancer_model:
        response = client.post(
            f"/api/predict?modality={selected_modality}",
            files={"file": ("chest.png", image_bytes(held_out_image("chest_xray")), "image/png")},
        )
    assert response.status_code == 400
    assert response.json()["error"] == MISMATCH
    cancer_model.assert_not_called()


@pytest.mark.parametrize("selected_modality", ["ultrasound", "histopathology"])
def test_unrelated_project_chart_is_rejected_before_cancer_model(selected_modality: str):
    chart_path = "smoke_test_outputs/ultrasound_roc_curve.png"
    with patch("app.main.load_model") as cancer_model:
        with open(chart_path, "rb") as chart:
            response = client.post(
                f"/api/predict?modality={selected_modality}",
                files={"file": ("chart.png", chart, "image/png")},
            )
    assert response.status_code == 400
    assert response.json()["error"] == UNVERIFIED
    cancer_model.assert_not_called()


def test_detector_evaluation_and_checkpoint_include_other_class():
    detector = load_modality_detector(torch.device("cpu"))
    assert detector.threshold > 0.0
    assert detector.classify(held_out_image("ultrasound"))[0] == "ultrasound"
    assert detector.classify(held_out_image("histopathology"))[0] == "histopathology"


@pytest.mark.parametrize("modality", ["ultrasound", "histopathology"])
def test_checkpoint_class_mapping_matches_training_configuration(modality: str):
    expected = MODALITIES[modality]["classes"]
    assert expected == {"benign": 0, "malignant": 1}
    checkpoint = torch.load(
        f"msi_results_full/{modality}_best_resnet18.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["modality"] == modality
    assert checkpoint["class_map"] == expected
    assert {index: name for name, index in checkpoint["class_map"].items()} == {0: "benign", 1: "malignant"}


def test_corrupt_image_is_rejected_before_model_call():
    with patch("app.main.load_model") as cancer_model:
        response = client.post(
            "/api/predict?modality=ultrasound",
            files={"file": ("broken.png", b"not an image", "image/png")},
        )
    assert response.status_code == 400
    cancer_model.assert_not_called()


def test_unsupported_upload_is_rejected_before_model_call():
    with patch("app.main.load_model") as cancer_model:
        response = client.post(
            "/api/predict?modality=histopathology",
            files={"file": ("scan.pdf", b"pdf", "application/pdf")},
        )
    assert response.status_code == 400
    assert "Unsupported image format" in response.json()["error"]
    cancer_model.assert_not_called()
