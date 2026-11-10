"""Single-image inference script for MSI Breast Cancer ResNet-18 models.

Loads a saved model checkpoint, applies exact preprocessing, and outputs
predicted dataset class, confidence, and class probabilities with robust error handling.

Educational and research use only; not for clinical diagnosis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_model_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Load model architecture and weights from checkpoint file."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint file does not exist: {checkpoint_path}')

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, ckpt


def build_transform(ckpt: dict) -> transforms.Compose:
    """Reconstruct exact preprocessing transform used during model evaluation."""
    img_size = ckpt.get('image_size', 224)
    norm = ckpt.get('normalization', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm['mean'], std=norm['std']),
    ])


def predict_image(image_path: Path, model: nn.Module, ckpt: dict, device: torch.device) -> dict:
    """Perform inference on a single image and return detailed predictions."""
    if not image_path.exists():
        return {
            'success': False,
            'error': f'Image file not found: {image_path}',
            'image_path': str(image_path),
        }

    if image_path.suffix.lower() not in SUPPORTED_EXTS:
        return {
            'success': False,
            'error': f'Unsupported image format "{image_path.suffix}". Supported formats: {sorted(SUPPORTED_EXTS)}',
            'image_path': str(image_path),
        }

    try:
        with Image.open(image_path) as im:
            im.verify()
        with Image.open(image_path) as im:
            img_rgb = im.convert('RGB')
            orig_size = im.size
            orig_mode = im.mode
    except Exception as exc:
        return {
            'success': False,
            'error': f'Corrupt or unreadable image file: {exc}',
            'image_path': str(image_path),
        }

    transform = build_transform(ckpt)
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # Reconstruct class mapping {label: class_name}
    class_map = ckpt.get('class_map', {})
    inv_class_map = {v: k for k, v in class_map.items()} if class_map else {0: 'class_0', 1: 'malignant'}

    pred_label = int(probs.argmax())
    pred_class = inv_class_map.get(pred_label, f'class_{pred_label}')
    confidence = float(probs[pred_label])

    prob_dict = {inv_class_map.get(i, f'class_{i}'): float(p) for i, p in enumerate(probs)}

    return {
        'success': True,
        'image_path': str(image_path.resolve()),
        'image_name': image_path.name,
        'original_dimensions': {'width': orig_size[0], 'height': orig_size[1]},
        'original_mode': orig_mode,
        'modality': ckpt.get('modality', 'unknown'),
        'predicted_class': pred_class,
        'predicted_label': pred_label,
        'confidence': confidence,
        'confidence_percentage': f'{confidence * 100:.2f}%',
        'class_probabilities': prob_dict,
        'best_checkpoint_epoch': ckpt.get('epoch', 'N/A'),
        'validation_macro_f1': ckpt.get('val_macro_f1', 'N/A'),
        'disclaimer': (
            'NOTICE: This result represents statistical classification by an experimental deep-learning '
            'model trained on benchmark datasets. It DOES NOT constitute medical advice or clinical diagnosis.'
        ),
    }


def main():
    parser = argparse.ArgumentParser(description='Inference with MSI Breast Cancer ResNet-18 Models')
    parser.add_argument('--model', required=True, help='Path to saved model checkpoint (.pt)')
    parser.add_argument('--image', required=True, help='Path to test image file')
    parser.add_argument('--json', action='store_true', help='Output purely in JSON format')
    parser.add_argument('--cpu', action='store_true', help='Force CPU execution even if CUDA is available')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')

    ckpt_path = Path(args.model).resolve()
    img_path = Path(args.image).resolve()

    try:
        model, ckpt = load_model_checkpoint(ckpt_path, device)
    except Exception as exc:
        err = {'success': False, 'error': f'Failed to load model checkpoint: {exc}'}
        if args.json:
            print(json.dumps(err, indent=2))
        else:
            print(f'ERROR: {err["error"]}', file=sys.stderr)
        sys.exit(1)

    result = predict_image(img_path, model, ckpt, device)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print('=' * 70)
        print('MSI BREAST IMAGE CLASSIFIER -- INFERENCE RESULT')
        print('=' * 70)
        if not result['success']:
            print(f'FAILED: {result["error"]}')
            sys.exit(1)

        print(f"Modality Model:       {result['modality'].upper()} (ResNet-18)")
        print(f"Image File:           {result['image_name']}")
        print(f"Original Resolution:  {result['original_dimensions']['width']}x{result['original_dimensions']['height']} ({result['original_mode']})")
        print(f"Predicted Class:      {result['predicted_class']}")
        print(f"Confidence:           {result['confidence_percentage']}")
        print(f"\nClass Probabilities:")
        for cls_name, p in result['class_probabilities'].items():
            bar = '#' * int(p * 30)
            print(f"  - {cls_name:<12}: {p * 100:6.2f}%  |{bar:<30}|")
        print('=' * 70)
        print(result['disclaimer'])
        print('=' * 70)


if __name__ == '__main__':
    main()
