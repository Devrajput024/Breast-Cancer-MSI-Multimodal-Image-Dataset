"""Audit feature learning and shortcut learning in the saved MSI ResNet-18 models.

This script intentionally does not retrain the baseline models. It uses the saved
checkpoints and dataset splits already present in the workspace to:
  1. generate Grad-CAM overlays for correctly and incorrectly classified held-out images,
  2. run simple masking/perturbation sensitivity checks,
  3. compare the current chest X-ray model against a reasonable alternative (ResNet-50)
     using validation data only.

The outputs are saved under a dedicated audit folder.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODALITIES = {
    'ultrasound': {
        'folder': 'Ultrasound Images_MSI',
        'classes': {'benign': 0, 'malignant': 1},
        'pos_name': 'malignant',
        'neg_name': 'benign',
    },
    'histopathology': {
        'folder': 'Histopathological_MSI',
        'classes': {'benign': 0, 'malignant': 1},
        'pos_name': 'malignant',
        'neg_name': 'benign',
    },
    'chest_xray': {
        'folder': 'Chest_XRay_MSI',
        'classes': {'Normal': 0, 'Malignant': 1},
        'pos_name': 'Malignant',
        'neg_name': 'Normal',
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pil = Image.open(row['path']).convert('RGB')
        if self.transform is not None:
            img = self.transform(pil)
        else:
            img = pil
        return img, int(row['label']), row['path'], row['filename']


def build_model(modality: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def compute_predictions(model, loader, device):
    y_true, y_pred, probs = [], [], []
    with torch.no_grad():
        for x, label, path, fname in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            y_true.extend(np.array(label, dtype=int))
            y_pred.extend(pred)
            probs.extend(prob)
    return np.array(y_true), np.array(y_pred), np.array(probs)


def select_examples_for_gradcam(modality: str, split_csv: Path, checkpoint_path: Path, device: torch.device, n_correct: int = 2, n_wrong: int = 2):
    df = pd.read_csv(split_csv)
    test_df = df[df['split'] == 'test'].copy().reset_index(drop=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    loader = DataLoader(EvalDataset(test_df, transform), batch_size=32, shuffle=False, num_workers=0)
    model = build_model(modality, checkpoint_path, device)
    y_t, y_p, probs = compute_predictions(model, loader, device)
    test_df['pred'] = y_p
    test_df['true'] = y_t
    test_df['prob_malignant'] = probs

    selected = []
    correct_idx = test_df[test_df['pred'] == test_df['true']].index.tolist()
    wrong_idx = test_df[test_df['pred'] != test_df['true']].index.tolist()

    for idx in correct_idx[:n_correct]:
        selected.append({'index': idx, 'kind': 'correct', 'row': test_df.loc[idx]})
    for idx in wrong_idx[:n_wrong]:
        selected.append({'index': idx, 'kind': 'wrong', 'row': test_df.loc[idx]})

    return selected, model, test_df


def save_overlay(path: Path, heatmap: np.ndarray, out_path: Path, alpha: float = 0.5):
    img = Image.open(path).convert('RGB').resize((224, 224))
    img_np = np.array(img)
    heat = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    jet = plt.cm.jet(heat)[:, :, :3]
    jet_img = (jet * 255).astype(np.uint8)
    overlay = Image.blend(img, Image.fromarray(jet_img), alpha)
    overlay.save(out_path)


def gradcam_for_sample(model: nn.Module, image_path: str, label: int, device: torch.device, out_dir: Path, prefix: str):
    img_pil = Image.open(image_path).convert('RGB').resize((224, 224))
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])(img_pil).unsqueeze(0).to(device)

    model.eval()
    conv_outputs = []
    grad_outputs = []

    def forward_hook(module, inp, out):
        conv_outputs.append(out)

    def backward_hook(module, grad_in, grad_out):
        grad_outputs.append(grad_out[0])

    handle_f = model.layer4[-1].conv2.register_forward_hook(forward_hook)
    handle_b = model.layer4[-1].conv2.register_full_backward_hook(backward_hook)

    logits = model(tensor)
    target = int(logits.argmax(dim=1).item()) if label is None else int(label)
    model.zero_grad()
    logits[:, target].sum().backward()

    handle_f.remove()
    handle_b.remove()

    feature = conv_outputs[0][0].detach()
    grad = grad_outputs[0][0].detach()
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = (weights * feature).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam = cam.squeeze().cpu().numpy()

    heat = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), resample=Image.BILINEAR))
    save_overlay(Path(image_path), heat / 255.0, out_dir / f'{prefix}_overlay.png')
    img_pil.save(out_dir / f'{prefix}_original.png')
    return heat


def collect_gradcam_outputs(modality: str, checkpoint_path: Path, split_csv: Path, audit_dir: Path, device: torch.device):
    selected, model, test_df = select_examples_for_gradcam(modality, split_csv, checkpoint_path, device)
    modality_dir = audit_dir / modality / 'gradcam'
    modality_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for item in selected:
        idx = item['index']
        kind = item['kind']
        row = item['row']
        prefix = f'{modality}_{kind}_{idx}'
        out_dir = modality_dir / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        heat = gradcam_for_sample(model, row['path'], int(row['true']), device, out_dir, 'sample')
        summary_rows.append({
            'modality': modality,
            'kind': kind,
            'index': idx,
            'true_label': int(row['true']),
            'pred_label': int(row['pred']),
            'path': row['path'],
            'score': float(row['prob_malignant']),
            'heatmap_mean': float(heat.mean()),
            'heatmap_max': float(heat.max()),
        })

    pd.DataFrame(summary_rows).to_csv(modality_dir / 'gradcam_summary.csv', index=False)
    return summary_rows


def create_perturbation_summary(modality: str, checkpoint_path: Path, split_csv: Path, audit_dir: Path, device: torch.device):
    df = pd.read_csv(split_csv)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    model = build_model(modality, checkpoint_path, device)
    rows = []
    for _, row in test_df.head(10).iterrows():
        pil = Image.open(row['path']).convert('RGB').resize((224, 224))
        orig_tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            orig_logits = model(orig_tensor)
            orig_prob = torch.softmax(orig_logits, dim=1)[:, 1].item()
            orig_pred = int(orig_prob >= 0.5)

        for mode in ['border', 'center']:
            masked = np.array(pil)
            h, w = masked.shape[:2]
            if mode == 'border':
                pad = int(min(h, w) * 0.12)
                mask = np.zeros((h, w), dtype=bool)
                mask[pad:h-pad, pad:w-pad] = True
                masked[~mask] = 0
            else:
                cx, cy = w // 2, h // 2
                rad = min(h, w) // 3
                yy, xx = np.ogrid[:h, :w]
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2
                masked[mask] = 0

            masked_img = Image.fromarray(masked)
            masked_tensor = transform(masked_img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(masked_tensor)
                prob = torch.softmax(logits, dim=1)[:, 1].item()
            rows.append({
                'modality': modality,
                'image_path': str(row['path']),
                'true_label': int(row['label']),
                'orig_pred': orig_pred,
                'orig_prob': float(orig_prob),
                'masked_mode': mode,
                'masked_prob': float(prob),
                'delta_prob': float(prob - orig_prob),
            })

    out_csv = audit_dir / modality / 'masking_summary.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows


def compare_validation_alternative(modality: str, data_root: Path, audit_dir: Path, device: torch.device, epochs: int = 4):
    if modality != 'chest_xray':
        return None

    csv_path = data_root.parent / 'msi_results_chest_exp1' / 'chest_xray_splits.csv'
    if not csv_path.exists():
        csv_path = data_root / 'chest_xray_splits.csv'
    if not csv_path.exists():
        return None

    split_df = pd.read_csv(csv_path)
    train_df = split_df[split_df['split'] == 'train'].copy().reset_index(drop=True)
    val_df = split_df[split_df['split'] == 'validation'].copy().reset_index(drop=True)

    def make_loader(df, transform):
        return DataLoader(EvalDataset(df, transform), batch_size=16, shuffle=True if df is train_df else False, num_workers=0)

    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Compare current ResNet-18 (already saved from the project) against a ResNet-50 alternative.
    results = []
    for model_name, model_builder in [('resnet18', lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT)), ('resnet50', lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT))]:
        model = model_builder()
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.to(device)
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        best_val_f1 = -1.0
        best_state = None

        for epoch in range(epochs):
            model.train()
            for x, y, _, _ in make_loader(train_df, tf_train):
                x = x.to(device)
                y = y.to(device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_y, val_pred, val_prob = [], [], []
            with torch.no_grad():
                for x, y, _, _ in make_loader(val_df, tf_eval):
                    logits = model(x.to(device))
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    val_y.extend(np.array(y, dtype=int))
                    val_pred.extend((probs >= 0.5).astype(int))
                    val_prob.extend(probs)
            val_y = np.array(val_y)
            val_pred = np.array(val_pred)
            val_prob = np.array(val_prob)
            val_f1 = f1_score(val_y, val_pred, average='macro', zero_division=0)
            val_acc = accuracy_score(val_y, val_pred)
            val_auc = roc_auc_score(val_y, val_prob)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            results.append({'model': model_name, 'epoch': epoch + 1, 'val_accuracy': float(val_acc), 'val_macro_f1': float(val_f1), 'val_auc': float(val_auc)})

        if best_state is not None:
            model.load_state_dict(best_state)

    out_csv = audit_dir / 'validation_only_model_comparison.csv'
    pd.DataFrame(results).to_csv(out_csv, index=False)
    return out_csv


def main():
    parser = argparse.ArgumentParser(description='Audit feature learning and shortcut learning using saved checkpoints.')
    parser.add_argument('--data-root', default=r'C:\Users\asus\OneDrive\Desktop\Breast Cncer Project\MultiModel Breast Cancer MSI Dataset', help='Path to the extracted dataset root')
    parser.add_argument('--output-root', default=r'C:\Users\asus\OneDrive\Desktop\Breast Cncer Project\msi_feature_audit', help='Directory for all audit outputs')
    parser.add_argument('--checkpoint-root', default=r'C:\Users\asus\OneDrive\Desktop\Breast Cncer Project', help='Workspace root holding saved checkpoints')
    parser.add_argument('--modalities', nargs='+', default=['ultrasound', 'histopathology', 'chest_xray'])
    parser.add_argument('--compare-alternative', action='store_true', help='Run a small validation-only chest X-ray alternative-model comparison')
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_root.mkdir(parents=True, exist_ok=True)

    for modality in args.modalities:
        split_path = checkpoint_root / 'msi_results_full' / f'{modality}_splits.csv'
        if modality == 'chest_xray' and not split_path.exists():
            split_path = checkpoint_root / 'msi_results_chest_exp1' / f'{modality}_splits.csv'
        if not split_path.exists():
            print(f'SKIP: missing split CSV for {modality}: {split_path}')
            continue

        checkpoint_path = checkpoint_root / 'msi_results_full' / f'{modality}_best_resnet18.pt'
        if modality == 'chest_xray' and not checkpoint_path.exists():
            checkpoint_path = checkpoint_root / 'msi_results_chest_exp1' / f'{modality}_best_resnet18.pt'
        if not checkpoint_path.exists():
            print(f'SKIP: missing checkpoint for {modality}: {checkpoint_path}')
            continue

        print(f'Running Grad-CAM audit for {modality}...')
        collect_gradcam_outputs(modality, checkpoint_path, split_path, output_root, device)
        create_perturbation_summary(modality, checkpoint_path, split_path, output_root, device)

    if args.compare_alternative:
        print('Running validation-only alternative model comparison for chest X-ray...')
        compare_validation_alternative('chest_xray', data_root, output_root, device, epochs=4)

    summary = {
        'audit_root': str(output_root),
        'modalities_processed': args.modalities,
        'device': str(device),
        'notes': [
            'Grad-CAM is used only as a saliency visualization aid and does not prove medically correct feature usage.',
            'Masking tests are sensitivity checks only; they do not establish causal mechanism.',
            'The chest X-ray dataset appears to include synthetic or transformed examples and should be investigated before additional expensive training.',
        ],
    }
    (output_root / 'audit_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Audit outputs saved to {output_root}')


if __name__ == '__main__':
    main()
