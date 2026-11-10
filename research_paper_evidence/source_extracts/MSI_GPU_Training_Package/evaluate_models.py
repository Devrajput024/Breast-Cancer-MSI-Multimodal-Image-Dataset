"""Independent evaluation and reporting script for trained MSI ResNet-18 models.

Evaluates saved checkpoints on dataset splits, computes comprehensive classification
metrics (Accuracy, Macro F1, Sensitivity, Specificity, ROC-AUC), and produces publication-ready
confusion matrices and ROC curves.

Educational and research use only; not for clinical diagnosis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BreastCancerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        with Image.open(row['path']) as im:
            img = im.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row['label'])


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, ckpt


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    criterion = nn.CrossEntropyLoss()
    ys, preds, probs = [], [], []
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = torch.as_tensor(y, device=device, dtype=torch.long)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            count += len(y)

            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            ys.extend(y.cpu().numpy())
            probs.extend(prob)
            preds.extend((prob >= 0.5).astype(int))

    return np.array(ys), np.array(preds), np.array(probs), total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description='Evaluate MSI Breast Cancer Classifier')
    parser.add_argument('--model', required=True, help='Path to checkpoint file (.pt)')
    parser.add_argument('--splits', required=True, help='Path to splits CSV file containing image paths and labels')
    parser.add_argument('--split-name', default='test', choices=['train', 'validation', 'test', 'all'], help='Which split partition to evaluate')
    parser.add_argument('--output', default='evaluation_results', help='Output directory for evaluation metrics and plots')
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating on device: {device}')

    ckpt_path = Path(args.model).resolve()
    splits_path = Path(args.splits).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(ckpt_path, device)
    modality = ckpt.get('modality', 'unknown')
    img_size = ckpt.get('image_size', 224)
    norm = ckpt.get('normalization', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})

    df = pd.read_csv(splits_path)
    if 'split' in df.columns and args.split_name != 'all':
        df = df[df['split'] == args.split_name]
    print(f'Evaluating on {len(df)} samples from split "{args.split_name}".')

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm['mean'], std=norm['std']),
    ])

    loader = DataLoader(BreastCancerDataset(df, eval_tf), batch_size=args.batch, shuffle=False)
    y_true, y_pred, y_prob, loss = evaluate(model, loader, device)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / max(tn + fp, 1))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    pos_name = ckpt.get('pos_name', 'Malignant')
    neg_name = ckpt.get('neg_name', 'Benign/Normal')
    class_names = [neg_name, pos_name]

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{modality.upper()} — Confusion Matrix ({args.split_name.title()} Set)', fontsize=12, fontweight='bold', pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            plt.text(j, i, f'{count}\n({pct:.1f}%)', ha='center', va='center',
                     color='white' if count > thresh else 'black', fontweight='bold')
    plt.ylabel('True Class', fontsize=11)
    plt.xlabel('Predicted Class', fontsize=11)
    plt.tight_layout()
    cm_out = out_dir / f'{modality}_{args.split_name}_confusion_matrix.png'
    plt.savefig(cm_out, dpi=200)
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Chance (AUC = 0.50)')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title(f'{modality.upper()} — ROC Curve ({args.split_name.title()} Set)', fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    roc_out = out_dir / f'{modality}_{args.split_name}_roc_curve.png'
    plt.savefig(roc_out, dpi=200)
    plt.close()

    results = {
        'modality': modality,
        'split': args.split_name,
        'n_samples': len(df),
        'loss': float(loss),
        'accuracy': acc,
        'macro_f1': macro_f1,
        'malignant_sensitivity_recall': recall,
        'malignant_precision': precision,
        'specificity': specificity,
        'roc_auc': auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'confusion_matrix_plot': str(cm_out.resolve()),
        'roc_curve_plot': str(roc_out.resolve()),
    }

    metrics_out = out_dir / f'{modality}_{args.split_name}_evaluation.json'
    metrics_out.write_text(json.dumps(results, indent=2))
    print('=' * 70)
    print(f'EVALUATION RESULTS: {modality.upper()} ({args.split_name.upper()} SET)')
    print('=' * 70)
    print(f'Accuracy:             {acc:.4f} ({acc*100:.2f}%)')
    print(f'Macro F1:             {macro_f1:.4f}')
    print(f'Sensitivity (Recall): {recall:.4f}')
    print(f'Specificity:          {specificity:.4f}')
    print(f'Precision:            {precision:.4f}')
    print(f'ROC-AUC Score:        {auc:.4f}')
    print(f'Results saved to:     {metrics_out}')
    print('=' * 70)


if __name__ == '__main__':
    main()
