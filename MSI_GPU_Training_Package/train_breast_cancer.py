"""Train separate ResNet-18 image classifiers for MSI Breast Cancer modalities.

Modalities:
1. Ultrasound (Ultrasound Images_MSI): benign (0) vs malignant (1)
2. Histopathology (Histopathological_MSI): benign (0) vs malignant (1)
3. Chest X-Ray (Chest_XRay_MSI): Normal (0) vs Malignant (1)

Includes:
- Group-aware 70/15/15 train/val/test splitting to prevent data leakage
- Transfer learning with ImageNet-pretrained ResNet-18
- Two-stage training: frozen backbone head warmup -> layer4 fine-tuning
- Full metric logging (Loss, Accuracy, Precision, Recall, Specificity, Macro F1, ROC-AUC)
- Automatic generation of training loss curves, metric curves, ROC curves, and confusion matrix plots
- Comprehensive test set evaluation and comparison summary export

Educational and research use only; not for clinical diagnosis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
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
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

MODALITIES = {
    'ultrasound': {
        'folder': 'Ultrasound Images_MSI',
        'classes': {'benign': 0, 'malignant': 1},
        'pos_label': 1,
        'pos_name': 'malignant',
        'neg_name': 'benign',
        'group_strategy': 'ultrasound_source',
    },
    'histopathology': {
        'folder': 'Histopathological_MSI',
        'classes': {'benign': 0, 'malignant': 1},
        'pos_label': 1,
        'pos_name': 'malignant',
        'neg_name': 'benign',
        'group_strategy': 'breakhis_slide',
    },
    'chest_xray': {
        'folder': 'Chest_XRay_MSI',
        'classes': {'Normal': 0, 'Malignant': 1},
        'pos_label': 1,
        'pos_name': 'Malignant',
        'neg_name': 'Normal',
        'group_strategy': 'chest_xray_instance',
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def extract_group_id(path: Path, strategy: str, class_name: str) -> str:
    """Derive leakage-safe group identifier for grouped train/val/test splits."""
    stem = path.stem
    if strategy == 'ultrasound_source':
        m = re.match(r'^(Multispectral_(?:benign|malignant)\s*\(\d+\))', stem, re.IGNORECASE)
        source = m.group(1).strip() if m else stem
    elif strategy == 'breakhis_slide':
        m = re.match(r'^(Multispectral_SOB_[A-Za-z]+_[A-Za-z0-9]+-\d+-[A-Za-z0-9]+)', stem)
        source = m.group(1) if m else stem
    else:
        source = stem

    key = f'{class_name.lower()}|{source.lower()}'
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def resolve_class_dir(base: Path, target_name: str) -> Path:
    """Find matching class subdirectory case-insensitively."""
    exact = base / target_name
    if exact.is_dir():
        return exact
    if base.is_dir():
        for child in base.iterdir():
            if child.is_dir() and child.name.lower() == target_name.lower():
                return child
    return exact


def build_manifest(root: Path, modality: str) -> pd.DataFrame:
    """Scan and validate all image files for a given modality."""
    cfg = MODALITIES[modality]
    base = root / cfg['folder']
    if not base.is_dir():
        raise FileNotFoundError(f'Missing dataset folder: {base}')

    rows = []
    for cls_name, label in cfg['classes'].items():
        cdir = resolve_class_dir(base, cls_name)
        if not cdir.is_dir():
            raise FileNotFoundError(f'Missing class folder: {cdir}')
        for p in cdir.rglob('*'):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                try:
                    with Image.open(p) as im:
                        im.verify()
                except Exception:
                    print(f'  [WARNING] Unreadable image skipped: {p}')
                    continue
                gid = extract_group_id(p, cfg['group_strategy'], cls_name)
                rows.append({
                    'modality': modality,
                    'path': str(p.resolve()),
                    'filename': p.name,
                    'class_name': cls_name,
                    'label': label,
                    'group': gid,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f'No images found in {base}')
    if df['label'].nunique() != 2:
        raise RuntimeError(f'Expected 2 classes; found: {df["class_name"].value_counts().to_dict()}')

    # Verify no group spans across multiple classes
    bad_groups = df.groupby('group')['label'].nunique()
    if (bad_groups > 1).any():
        raise RuntimeError(f'Data leakage error: some groups span multiple labels in {modality}!')

    return df


def split_grouped(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Perform group-aware 70% Train, 15% Validation, 15% Test split."""
    # Target: 15% test, and 15% validation from total (0.15 / 0.85 ≈ 0.1765 of train_val)
    best_split = None
    best_loss = float('inf')

    for s in range(seed, seed + 300):
        try:
            # First split: 15% test
            gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=s)
            trval_idx, test_idx = next(gss1.split(df, df['label'], df['group']))
            trval = df.iloc[trval_idx].copy()
            test = df.iloc[test_idx].copy()

            if test['label'].nunique() < 2 or trval['label'].nunique() < 2:
                continue

            # Second split: ~17.65% of trval to make ~15% validation of total
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=s + 500)
            tr_idx, val_idx = next(gss2.split(trval, trval['label'], trval['group']))
            train = trval.iloc[tr_idx].copy()
            val = trval.iloc[val_idx].copy()

            if train['label'].nunique() < 2 or val['label'].nunique() < 2:
                continue

            total = len(df)
            p_train, p_val, p_test = len(train) / total, len(val) / total, len(test) / total
            # Loss deviation from 0.70 / 0.15 / 0.15
            loss = abs(p_train - 0.70) + abs(p_val - 0.15) + abs(p_test - 0.15)
            if loss < best_loss:
                best_loss = loss
                best_split = (train, val, test)

            # Accept if split is within 3% of target
            if abs(p_test - 0.15) <= 0.03 and abs(p_val - 0.15) <= 0.03:
                break
        except Exception:
            continue

    if best_split is not None:
        train, val, test = best_split
        note = 'Group-aware 70/15/15 stratified split using source/patient groups'
        return train, val, test, note

    raise RuntimeError('Could not find a valid group-aware split with both classes in all sets.')


class BreastCancerDataset(Dataset):
    """PyTorch Dataset loading images on the fly with transforms."""

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


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute predictions, probabilities, and evaluation loss."""
    model.eval()
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
            y_arr = y.cpu().numpy()
            ys.extend(y_arr)
            probs.extend(prob)
            preds.extend((prob >= 0.5).astype(int))

    eval_loss = total_loss / max(count, 1)
    return np.array(ys), np.array(preds), np.array(probs), eval_loss


def plot_training_curves(history: list[dict], out_dir: Path, modality: str):
    """Plot and save Loss and Accuracy/F1 curves."""
    df_h = pd.DataFrame(history)

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(df_h['epoch'], df_h['train_loss'], label='Train Loss', marker='o', color='#1f77b4', linewidth=2)
    plt.plot(df_h['epoch'], df_h['val_loss'], label='Val Loss', marker='s', color='#ff7f0e', linewidth=2)
    plt.title(f'{modality.upper()} — Training & Validation Loss', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_path = out_dir / f'{modality}_loss_curve.png'
    plt.savefig(loss_path, dpi=200)
    plt.close()

    # 2. Metric Curves
    plt.figure(figsize=(8, 5))
    plt.plot(df_h['epoch'], df_h['train_accuracy'], label='Train Acc', marker='o', color='#2ca02c', linewidth=2)
    plt.plot(df_h['epoch'], df_h['val_accuracy'], label='Val Acc', marker='s', color='#1f77b4', linewidth=2)
    plt.plot(df_h['epoch'], df_h['val_macro_f1'], label='Val Macro F1', marker='^', color='#d62728', linewidth=2)
    plt.title(f'{modality.upper()} — Training & Validation Metrics', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    metric_path = out_dir / f'{modality}_metrics_curve.png'
    plt.savefig(metric_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path, modality: str):
    """Plot formatted confusion matrix with raw counts and normalized percentages."""
    plt.figure(figsize=(6, 5))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{modality.upper()} — Test Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            plt.text(
                j,
                i,
                f'{count}\n({pct:.1f}%)',
                horizontalalignment='center',
                verticalalignment='center',
                color='white' if count > thresh else 'black',
                fontsize=11,
                fontweight='bold',
            )

    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, roc_auc: float, out_path: Path, modality: str):
    """Plot ROC curve with AUC score and chance diagonal."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title(f'{modality.upper()} — ROC Curve', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_modality(
    root: Path,
    out_dir: Path,
    modality: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Train, validate, checkpoint, and test a ResNet-18 model for one modality."""
    cfg = MODALITIES[modality]
    print(f'\n{"=" * 40} {modality.upper()} {"=" * 40}')

    # 1. Manifest & Split
    df = build_manifest(root, modality)
    print(f'Total Images: {len(df)} | Class Distribution: {df["class_name"].value_counts().to_dict()}')
    df.to_csv(out_dir / f'{modality}_manifest.csv', index=False)

    train_df, val_df, test_df, split_note = split_grouped(df, seed=args.seed)
    split_summary = pd.concat([
        train_df.assign(split='train'),
        val_df.assign(split='validation'),
        test_df.assign(split='test'),
    ])
    split_summary.to_csv(out_dir / f'{modality}_splits.csv', index=False)
    print(f'Splits: Train={len(train_df)} ({len(train_df)/len(df):.1%}), '
          f'Val={len(val_df)} ({len(val_df)/len(df):.1%}), '
          f'Test={len(test_df)} ({len(test_df)/len(df):.1%})')
    print(f'Split Method: {split_note}')

    # 2. Transforms & DataLoaders
    train_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    pin_mem = (device.type == 'cuda')
    train_loader = DataLoader(BreastCancerDataset(train_df, train_transform), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=pin_mem)
    val_loader = DataLoader(BreastCancerDataset(val_df, eval_transform), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin_mem)
    test_loader = DataLoader(BreastCancerDataset(test_df, eval_transform), batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=pin_mem)

    # 3. Model Setup
    print('Initializing ResNet-18 with ImageNet pretrained weights...')
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        print('Successfully loaded ImageNet weights.')
    except Exception as exc:
        print(f'Warning: Could not fetch online weights ({exc}), initializing randomly.')
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    model = model.to(device)

    # Stage 1: Freeze backbone, train only classification head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_epoch = 0
    best_ckpt_path = out_dir / f'{modality}_best_resnet18.pt'
    history = []
    patience_counter = 0

    max_epochs = 1 if args.smoke_test else args.epochs

    print(f'\nStarting Training ({max_epochs} epochs, freeze_epochs={args.freeze_epochs})...')
    for epoch in range(1, max_epochs + 1):
        # Stage 2: Unfreeze deeper layers (layer4) after freeze_epochs
        if not args.smoke_test and epoch == args.freeze_epochs + 1:
            print(f'>>> Epoch {epoch}: Unfreezing ResNet-18 layer4 for fine-tuning.')
            for param in model.layer4.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {'params': model.layer4.parameters(), 'lr': args.lr / 10},
                {'params': model.fc.parameters(), 'lr': args.lr / 3},
            ], weight_decay=1e-4)

        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=pin_mem)
            y = torch.as_tensor(y, device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(y)
            train_total += len(y)
            train_correct += (logits.argmax(dim=1) == y).sum().item()

            if args.smoke_test:
                break

        # Validation
        val_y, val_preds, val_probs, val_loss = evaluate_model(model, val_loader, device)
        val_acc = float(accuracy_score(val_y, val_preds))
        val_f1 = float(f1_score(val_y, val_preds, average='macro', zero_division=0))
        val_recall = float(recall_score(val_y, val_preds, zero_division=0))
        val_precision = float(precision_score(val_y, val_preds, zero_division=0))
        try:
            val_auc = float(roc_auc_score(val_y, val_probs))
        except Exception:
            val_auc = 0.5

        epoch_stats = {
            'epoch': epoch,
            'train_loss': train_loss_total / max(train_total, 1),
            'train_accuracy': train_correct / max(train_total, 1),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_macro_f1': val_f1,
            'val_malignant_recall': val_recall,
            'val_malignant_precision': val_precision,
            'val_roc_auc': val_auc,
        }
        history.append(epoch_stats)
        print(f"Epoch {epoch:2d}/{max_epochs} | "
              f"Train Loss: {epoch_stats['train_loss']:.4f}, Acc: {epoch_stats['train_accuracy']:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # Checkpoint selection
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_macro_f1': val_f1,
                'val_accuracy': val_acc,
                'val_roc_auc': val_auc,
                'modality': modality,
                'image_size': args.size,
                'class_map': cfg['classes'],
                'pos_label': cfg['pos_label'],
                'pos_name': cfg['pos_name'],
                'neg_name': cfg['neg_name'],
                'normalization': {'mean': IMAGENET_MEAN, 'std': IMAGENET_STD},
            }, best_ckpt_path)
            print(f'  --> Saved best checkpoint: {best_ckpt_path.name} (Val F1: {val_f1:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience and not args.smoke_test:
                print(f'Early stopping triggered after {args.patience} epochs without validation F1 improvement.')
                break

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = out_dir / f'{modality}_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f'Saved training history: {history_path}')

    # Plot loss and metric curves
    plot_training_curves(history, out_dir, modality)
    print(f'Generated loss and metric curves in {out_dir}')

    # 4. Final Evaluation on Held-Out Test Set using Best Checkpoint
    print(f'\nEvaluating best model (from Epoch {best_epoch}) on held-out test set...')
    best_checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_y, test_preds, test_probs, test_loss = evaluate_model(model, test_loader, device)
    cm = confusion_matrix(test_y, test_preds, labels=[0, 1])

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / max(tn + fp, 1))
    sensitivity = float(tp / max(tp + fn, 1)) # Malignant recall
    precision = float(tp / max(tp + fp, 1))
    test_acc = float(accuracy_score(test_y, test_preds))
    test_f1 = float(f1_score(test_y, test_preds, average='macro', zero_division=0))
    try:
        test_auc = float(roc_auc_score(test_y, test_probs))
    except Exception:
        test_auc = 0.5

    class_names = [cfg['neg_name'], cfg['pos_name']]
    clf_report = classification_report(test_y, test_preds, target_names=class_names, output_dict=True, zero_division=0)

    # Plot Confusion Matrix & ROC Curve
    cm_path = out_dir / f'{modality}_confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, cm_path, modality)

    roc_path = out_dir / f'{modality}_roc_curve.png'
    plot_roc_curve(test_y, test_probs, test_auc, roc_path, modality)

    metrics = {
        'modality': modality,
        'best_epoch': best_epoch,
        'n_images_total': int(len(df)),
        'n_train': int(len(train_df)),
        'n_validation': int(len(val_df)),
        'n_test': int(len(test_df)),
        'split_method': split_note,
        'positive_class': cfg['pos_name'],
        'positive_label': cfg['pos_label'],
        'test_loss': float(test_loss),
        'test_accuracy': test_acc,
        'test_macro_f1': test_f1,
        'malignant_sensitivity_recall': sensitivity,
        'malignant_precision': precision,
        'specificity_class0': specificity,
        'roc_auc': test_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': clf_report,
        'checkpoint_path': str(best_ckpt_path.resolve()),
        'disclaimer': 'Research and education only. Does not establish clinical validity.',
    }

    metrics_path = out_dir / f'{modality}_test_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f'Saved test metrics: {metrics_path}')
    print(f'Test Accuracy: {test_acc:.4f} | Macro F1: {test_f1:.4f} | Sensitivity: {sensitivity:.4f} | Specificity: {specificity:.4f} | ROC-AUC: {test_auc:.4f}')

    return metrics


def main():
    default_root = Path(__file__).resolve().parents[1] / 'MultiModel Breast Cancer MSI Dataset'
    parser = argparse.ArgumentParser(description='Train ResNet-18 on MSI Breast Cancer Modalities')
    parser.add_argument('--data', default=str(default_root), help='Path to extracted MSI dataset directory')
    parser.add_argument('--output', default='msi_training_outputs', help='Directory to save checkpoints, plots, and metrics')
    parser.add_argument('--modalities', nargs='+', choices=list(MODALITIES), default=list(MODALITIES))
    parser.add_argument('--epochs', type=int, default=12, help='Total training epochs')
    parser.add_argument('--freeze-epochs', type=int, default=3, help='Epochs to keep backbone frozen before fine-tuning')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--size', type=int, default=224, help='Image resolution (height/width)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate for classification head')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--smoke-test', action='store_true', help='Run 1 quick epoch to verify end-to-end execution')

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 80)
    print('MULTI-MODAL BREAST CANCER RESNET-18 CLASSIFIER')
    print(f'Compute Device: {device}')
    if device.type == 'cuda':
        print(f'GPU Device Name: {torch.cuda.get_device_name(0)}')
        print(f'Allocated Memory: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB')
    print(f'Data Directory: {Path(args.data).resolve()}')
    print(f'Output Directory: {Path(args.output).resolve()}')
    print(f'Modalities to Train: {args.modalities}')
    print(f'Smoke Test Mode: {args.smoke_test}')
    print('=' * 80)

    data_root = Path(args.data).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for m in args.modalities:
        res = train_modality(data_root, out_dir, m, args, device)
        results.append(res)

    # Save consolidated comparison summary
    summary_rows = []
    for r in results:
        summary_rows.append({
            'modality': r['modality'],
            'best_epoch': r['best_epoch'],
            'n_total': r['n_images_total'],
            'n_train': r['n_train'],
            'n_val': r['n_validation'],
            'n_test': r['n_test'],
            'test_accuracy': r['test_accuracy'],
            'test_macro_f1': r['test_macro_f1'],
            'malignant_sensitivity_recall': r['malignant_sensitivity_recall'],
            'malignant_precision': r['malignant_precision'],
            'specificity_class0': r['specificity_class0'],
            'roc_auc': r['roc_auc'],
            'split_method': r['split_method'],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / 'comparison_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print('\n' + '=' * 80)
    print('TRAINING RUN COMPLETE — COMPARISON SUMMARY')
    print('=' * 80)
    print(summary_df.to_string(index=False))
    print(f'\nAll results, checkpoints, plots, and manifests saved to: {out_dir}')


if __name__ == '__main__':
    main()
