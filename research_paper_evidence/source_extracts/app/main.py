from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFile
from torch import nn
from torchvision import models
from torchvision import transforms
from app.modality_detector import load_modality_detector

ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT
STATIC_DIR = ROOT / "app" / "static"
TEMPLATE_DIR = ROOT / "app" / "templates"
MODEL_DIR = ROOT / "msi_results_full"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = {
    "ultrasound": MODEL_DIR / "ultrasound_best_resnet18.pt",
    "histopathology": MODEL_DIR / "histopathology_best_resnet18.pt",
}

MODEL_METRICS = {
    "ultrasound": json.loads((MODEL_DIR / "ultrasound_test_metrics.json").read_text()),
    "histopathology": json.loads((MODEL_DIR / "histopathology_test_metrics.json").read_text()),
}

MODEL_INFO = {
    "ultrasound": {
        "name": "Ultrasound",
        "status": "available",
        "description": "Binary classification model trained on the preserved ultrasound baseline checkpoint.",
        "note": "This is the original baseline model used for the preserved benchmark evaluation.",
        "research_status": "Research-use only; no clinical validation.",
        "metrics": MODEL_METRICS["ultrasound"],
    },
    "histopathology": {
        "name": "Histopathology",
        "status": "available",
        "description": "Binary histopathology classifier using the preserved baseline checkpoint.",
        "note": "This model is suitable for exploratory research use only and should be interpreted cautiously.",
        "research_status": "Research-use only; not clinically validated.",
        "metrics": MODEL_METRICS["histopathology"],
    },
    "chest_xray": {
        "name": "Chest X-ray",
        "status": "unavailable",
        "description": "Excluded from reliable inference workflows due to weak preserved performance and limited evidence.",
        "note": "Not used in the interface because the preserved baseline is not reliable enough for a production-style prediction workflow.",
        "research_status": "Excluded from reliable inference. Historical experimental status only.",
        "metrics": None,
    },
}

MODEL_CACHE: dict[str, tuple[nn.Module, dict[str, Any]]] = {}

REJECTION_MESSAGE = "Image modality could not be verified. Please upload a suitable ultrasound or histopathology image."
MODALITY_MISMATCH_MESSAGE = "Selected modality does not match the uploaded image. Please upload a matching image."
REJECTION_CODE = "MODALITY_REJECTION"


def validate_image_modality(image: Image.Image, modality: str) -> tuple[bool, str]:
    """Verify modality with the independent detector before cancer inference."""
    if modality not in {"ultrasound", "histopathology"}:
        return False, REJECTION_MESSAGE
    if image.width < 64 or image.height < 64:
        return False, REJECTION_MESSAGE
    try:
        detected_modality, _confidence = load_modality_detector(DEVICE).classify(image)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Modality verification is unavailable: {exc}") from exc
    if detected_modality is None:
        return False, REJECTION_MESSAGE
    if detected_modality != modality:
        return False, MODALITY_MISMATCH_MESSAGE
    return True, "ok"


app = FastAPI(
    title="Breast Cancer Image Classification",
    description="Research dashboard for ultrasound and histopathology image classification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def load_model(modality: str) -> tuple[nn.Module, dict[str, Any]]:
    if modality not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail=f"Unsupported modality '{modality}'. Supported modalities are: ultrasound, histopathology.")

    if modality in MODEL_CACHE:
        return MODEL_CACHE[modality]

    checkpoint_path = MODEL_PATHS[modality]
    if not checkpoint_path.exists():
        raise HTTPException(status_code=500, detail=f"Checkpoint missing for {modality}: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    expected_class_map = {"benign": 0, "malignant": 1}
    if checkpoint.get("modality") != modality or checkpoint.get("class_map") != expected_class_map:
        raise HTTPException(status_code=500, detail=f"Checkpoint metadata is invalid for {modality}; expected class mapping {expected_class_map}.")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    MODEL_CACHE[modality] = (model, checkpoint)
    return MODEL_CACHE[modality]


def build_transform(checkpoint: dict[str, Any]) -> Any:
    image_size = int(checkpoint.get("image_size", 224))
    normalization = checkpoint.get("normalization", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    mean = normalization.get("mean", [0.485, 0.456, 0.406])
    std = normalization.get("std", [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


@app.get("/")
def home() -> FileResponse:
    return FileResponse(TEMPLATE_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str | list[str]]:
    return {"status": "ok", "supported_modalities": ["ultrasound", "histopathology"]}


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    return {
        "research_disclaimer": "This application is for research and educational use only. It does not constitute clinical diagnosis or medical advice.",
        "modalities": [
            {
                "key": "ultrasound",
                "name": MODEL_INFO["ultrasound"]["name"],
                "status": MODEL_INFO["ultrasound"]["status"],
                "description": MODEL_INFO["ultrasound"]["description"],
                "note": MODEL_INFO["ultrasound"]["note"],
                "metrics": MODEL_INFO["ultrasound"]["metrics"],
            },
            {
                "key": "histopathology",
                "name": MODEL_INFO["histopathology"]["name"],
                "status": MODEL_INFO["histopathology"]["status"],
                "description": MODEL_INFO["histopathology"]["description"],
                "note": MODEL_INFO["histopathology"]["note"],
                "metrics": MODEL_INFO["histopathology"]["metrics"],
            },
            {
                "key": "chest_xray",
                "name": MODEL_INFO["chest_xray"]["name"],
                "status": MODEL_INFO["chest_xray"]["status"],
                "description": MODEL_INFO["chest_xray"]["description"],
                "note": MODEL_INFO["chest_xray"]["note"],
                "metrics": None,
            },
        ],
    }


@app.post("/api/predict")
async def predict(modality: str, file: UploadFile = File(...)) -> JSONResponse:
    if modality not in {"ultrasound", "histopathology"}:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Chest X-ray is not available for reliable inference in this interface. Please use ultrasound or histopathology.",
            },
        )

    if not file.filename:
        return JSONResponse(status_code=400, content={"success": False, "error": "No file was provided."})

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": f"Unsupported image format '{suffix}'. Supported formats are: {sorted(SUPPORTED_EXTENSIONS)}.",
            },
        )

    image_bytes = await file.read()
    if not image_bytes:
        return JSONResponse(status_code=400, content={"success": False, "error": "Uploaded image is empty."})

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb_image = img.convert("RGB")
    except Exception as exc:
        return JSONResponse(status_code=400, content={"success": False, "error": f"Corrupt or unreadable image: {exc}"})

    is_valid_modality, modality_message = validate_image_modality(rgb_image, modality)
    if not is_valid_modality:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": modality_message,
                "code": REJECTION_CODE,
            },
        )

    try:
        model, checkpoint = load_model(modality)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return JSONResponse(status_code=500, content={"success": False, "error": f"Failed to load model: {exc}"})

    transform = build_transform(checkpoint)

    try:
        tensor = transform(rgb_image).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Inference failed: {exc}"})

    class_map = checkpoint["class_map"]
    label_lookup = {int(v): str(k) for k, v in class_map.items()}
    predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    predicted_class = label_lookup.get(predicted_index, str(predicted_index))
    confidence = float(probabilities[predicted_index])

    class_probabilities = {
        label_lookup.get(i, f"class_{i}"): float(value)
        for i, value in enumerate(probabilities)
    }

    return JSONResponse(
        content={
            "success": True,
            "modality": modality,
            "predicted_class": predicted_class,
            "predicted_label": predicted_index,
            "model_confidence": round(confidence, 6),
            "model_confidence_percentage": round(confidence * 100, 2),
            "class_probabilities": class_probabilities,
            "checkpoint_epoch": checkpoint.get("epoch", "N/A"),
            "validation_macro_f1": checkpoint.get("val_macro_f1", "N/A"),
            "disclaimer": "This result represents model output only and is not a clinical diagnosis.",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
