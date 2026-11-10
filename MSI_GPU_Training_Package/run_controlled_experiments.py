"""Controlled Experiment Runner for Chest X-Ray ResNet-18 Optimization.

Runs multiple systematic, controlled experiments on Chest X-Ray MSI:
- Experiment 1: Baseline Frozen Head (Reproduction & Diagnostic)
- Experiment 2: Two-Stage Fine-Tuning (Layer 4 Unfrozen + StepLR)
- Experiment 3: Deep Fine-Tuning (Layer 3 + Layer 4 + CosineAnnealingLR + Radiographic Augmentations)
- Experiment 4: Label Smoothing (0.05) + Weight Decay Regularization + CosineAnnealingLR

Strictly selects the best model based ONLY on validation set Macro F1 / loss.
Evaluates the selected winning model once on the held-out test set.
Exports comprehensive logs, curves, and comparison tables.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class XRayDataset(Dataset):
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


def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, count = 0.0, 0
    ys, preds, probs = [], [], []

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

    y_arr = np.array(ys)
    p_arr = np.array(preds)
    prob_arr = np.array(probs)

    acc = float(accuracy_score(y_arr, p_arr))
    macro_f1 = float(f1_score(y_arr, p_arr, average='macro', zero_division=0))
    rec = float(recall_score(y_arr, p_arr, zero_division=0)) # Malignant recall
    prec = float(precision_score(y_arr, p_arr, zero_division=0))
    cm = confusion_matrix(y_arr, p_arr, labels=[0, 1])
    spec = float(cm[0, 0] / max(cm[0].sum(), 1))
    try:
        auc = float(roc_auc_score(y_arr, prob_arr))
    except Exception:
        auc = 0.5

    loss_avg = total_loss / max(count, 1)
    return loss_avg, acc, macro_f1, rec, prec, spec, y_arr, p_arr, prob_arr


def run_experiment(
    exp_name: str,
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
    device: torch.device,
) -> dict:
    print(f'\n{"=" * 70}')
    print(f'STARTING EXPERIMENT: {exp_name}')
    print(f'Configuration: {json.dumps(config, indent=2)}')
    print(f'{"=" * 70}')

    exp_dir = out_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Transforms
    if config.get('augmentation') == 'enhanced_radiographic':
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.96, 1.04)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(XRayDataset(train_df, train_tf), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(XRayDataset(val_df, eval_tf), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(XRayDataset(test_df, eval_tf), batch_size=batch_size, shuffle=False)

    # 2. Model Setup
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if config.get('pretrained', True) else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Freeze stages
    freeze_epochs = config.get('freeze_epochs', 2)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    base_lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Loss function with optional label smoothing
    label_smoothing = config.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    epochs = config.get('epochs', 10)
    scheduler_type = config.get('scheduler', 'none')
    scheduler = None

    history = []
    best_val_f1 = -1.0
    best_val_epoch = 0
    best_ckpt_path = exp_dir / 'best_model.pt'

    for epoch in range(1, epochs + 1):
        # Stage transition: unfreeze layers
        if epoch == freeze_epochs + 1 and config.get('unfreeze_layers'):
            layers_to_unfreeze = config['unfreeze_layers']
            print(f'>>> Epoch {epoch}: Unfreezing {layers_to_unfreeze}')
            param_groups = [{'params': model.fc.parameters(), 'lr': base_lr / 3}]
            if 'layer4' in layers_to_unfreeze:
                for p in model.layer4.parameters():
                    p.requires_grad = True
                param_groups.append({'params': model.layer4.parameters(), 'lr': base_lr / 10})
            if 'layer3' in layers_to_unfreeze:
                for p in model.layer3.parameters():
                    p.requires_grad = True
                param_groups.append({'params': model.layer3.parameters(), 'lr': base_lr / 25})

            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

            if scheduler_type == 'cosine':
                remaining_epochs = epochs - freeze_epochs
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)

        # Train loop
        model.train()
        train_loss_sum, train_correct, train_n = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = torch.as_tensor(y, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(y)
            train_n += len(y)
            train_correct += (logits.argmax(1) == y).sum().item()

        if scheduler is not None:
            scheduler.step()

        # Validation evaluation
        val_loss, val_acc, val_f1, val_rec, val_prec, val_spec, _, _, val_probs = evaluate_split(model, val_loader, device)
        try:
            val_auc = float(roc_auc_score(val_df['label'], val_probs))
        except Exception:
            val_auc = 0.5

        stat = {
            'epoch': epoch,
            'train_loss': train_loss_sum / max(train_n, 1),
            'train_accuracy': train_correct / max(train_n, 1),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_macro_f1': val_f1,
            'val_sensitivity': val_rec,
            'val_specificity': val_spec,
            'val_roc_auc': val_auc,
        }
        history.append(stat)
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {stat['train_loss']:.4f}, Acc: {stat['train_accuracy']:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        # Checkpoint selection strictly on validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'exp_name': exp_name,
                'epoch': epoch,
                'val_macro_f1': val_f1,
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'val_roc_auc': val_auc,
            }, best_ckpt_path)
            print(f'  --> New best validation F1 ({val_f1:.4f}) saved at Epoch {epoch}')

    # Save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(exp_dir / 'history.csv', index=False)

    # Plot loss and metric curves
    plt.figure(figsize=(7, 4.5))
    plt.plot(hist_df['epoch'], hist_df['train_loss'], label='Train Loss', marker='o')
    plt.plot(hist_df['epoch'], hist_df['val_loss'], label='Val Loss', marker='s')
    plt.title(f'{exp_name} — Loss Curve', fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / 'loss_curve.png', dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist_df['epoch'], hist_df['train_accuracy'], label='Train Acc', marker='o')
    plt.plot(hist_df['epoch'], hist_df['val_accuracy'], label='Val Acc', marker='s')
    plt.plot(hist_df['epoch'], hist_df['val_macro_f1'], label='Val F1', marker='^')
    plt.title(f'{exp_name} — Accuracy & F1', fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.grid(True, alpha=0.4); plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / 'metrics_curve.png', dpi=180)
    plt.close()

    # Load best checkpoint for validation summary
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    return {
        'exp_name': exp_name,
        'config': config,
        'best_epoch': best_val_epoch,
        'best_val_loss': best_ckpt['val_loss'],
        'best_val_accuracy': best_ckpt['val_accuracy'],
        'best_val_macro_f1': best_ckpt['val_macro_f1'],
        'best_val_roc_auc': best_ckpt['val_roc_auc'],
        'checkpoint_path': str(best_ckpt_path.resolve()),
        'history_csv': str((exp_dir / 'history.csv').resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='chest_xray_experiments')
    parser.add_argument('--splits', default='msi_results_full/chest_xray_splits.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 80)
    print('CHEST X-RAY SYSTEMATIC CONTROLLED EXPERIMENT SUITE')
    print(f'Device: {device} ({torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"})')
    print(f'Splits CSV: {args.splits}')
    print(f'Output Directory: {args.output}')
    print('=' * 80)

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_splits = pd.read_csv(args.splits)
    train_df = df_splits[df_splits['split'] == 'train']
    val_df = df_splits[df_splits['split'] == 'validation']
    test_df = df_splits[df_splits['split'] == 'test']

    print(f'Split counts: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}')

    # Define Experiment Configurations
    experiments = {
        'Exp1_Baseline_FrozenHead': {
            'epochs': 5,
            'freeze_epochs': 5,
            'unfreeze_layers': [],
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 16,
            'augmentation': 'standard',
            'scheduler': 'none',
            'label_smoothing': 0.0,
            'pretrained': True,
            'description': 'Reproduction of baseline: ResNet-18 frozen backbone, only linear head trained.',
        },
        'Exp2_Layer4_FineTuning': {
            'epochs': 10,
            'freeze_epochs': 2,
            'unfreeze_layers': ['layer4'],
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 16,
            'augmentation': 'standard',
            'scheduler': 'none',
            'label_smoothing': 0.0,
            'pretrained': True,
            'description': 'Warmup head 2 epochs, unfreeze layer4 with lr=1e-4, constant lr.',
        },
        'Exp3_Layer3_Layer4_CosineLR_EnhancedAug': {
            'epochs': 12,
            'freeze_epochs': 2,
            'unfreeze_layers': ['layer3', 'layer4'],
            'lr': 1e-3,
            'weight_decay': 5e-4,
            'batch_size': 16,
            'augmentation': 'enhanced_radiographic',
            'scheduler': 'cosine',
            'label_smoothing': 0.0,
            'pretrained': True,
            'description': 'Multi-layer fine-tuning (layers 3 & 4), CosineAnnealingLR, affine/contrast augmentations.',
        },
        'Exp4_LabelSmoothing_Regularized': {
            'epochs': 12,
            'freeze_epochs': 2,
            'unfreeze_layers': ['layer4'],
            'lr': 8e-4,
            'weight_decay': 1e-3,
            'batch_size': 16,
            'augmentation': 'enhanced_radiographic',
            'scheduler': 'cosine',
            'label_smoothing': 0.05,
            'pretrained': True,
            'description': 'Layer4 fine-tuning with label smoothing 0.05 and increased weight decay regularization.',
        },
    }

    results = []
    for exp_name, cfg in experiments.items():
        res = run_experiment(exp_name, cfg, train_df, val_df, test_df, out_dir, device)
        results.append(res)

    # Compare strictly by Validation Macro F1 and Validation Loss
    res_df = pd.DataFrame(results)
    res_df_sorted = res_df.sort_values(by='best_val_macro_f1', ascending=False)
    comparison_csv = out_dir / 'chest_xray_experiment_comparison.csv'
    res_df_sorted.to_csv(comparison_csv, index=False)

    print('\n' + '=' * 80)
    print('CHEST X-RAY EXPERIMENT COMPARISON (VALIDATION-BASED SELECTION)')
    print('=' * 80)
    display_cols = ['exp_name', 'best_epoch', 'best_val_loss', 'best_val_accuracy', 'best_val_macro_f1', 'best_val_roc_auc']
    print(res_df_sorted[display_cols].to_string(index=False))

    # Pick the winning model based ONLY on validation performance
    winner = res_df_sorted.iloc[0]
    print(f'\n>>> SELECTED WINNING MODEL BASED ON VALIDATION: {winner["exp_name"]}')
    print(f'    Best Validation Macro F1: {winner["best_val_macro_f1"]:.4f} (at Epoch {winner["best_epoch"]})')
    print(f'    Best Validation Accuracy: {winner["best_val_accuracy"]:.4f}')
    print(f'    Best Validation Loss:     {winner["best_val_loss"]:.4f}')

    # Now, and ONLY now, evaluate the selected winning model on the held-out test set
    print(f'\nEvaluating chosen model ({winner["exp_name"]}) ONCE on held-out test set...')
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_loader = DataLoader(XRayDataset(test_df, eval_tf), batch_size=16, shuffle=False)

    winning_ckpt = torch.load(winner['checkpoint_path'], map_location=device, weights_only=False)
    win_model = models.resnet18(weights=None)
    win_model.fc = nn.Linear(win_model.fc.in_features, 2)
    win_model.load_state_dict(winning_ckpt['model_state_dict'])
    win_model = win_model.to(device)

    t_loss, t_acc, t_f1, t_rec, t_prec, t_spec, y_t, p_t, prob_t = evaluate_split(win_model, test_loader, device)
    cm = confusion_matrix(y_t, p_t, labels=[0, 1])
    try:
        t_auc = float(roc_auc_score(y_t, prob_t))
    except Exception:
        t_auc = 0.5

    clf_rep = classification_report(y_t, p_t, target_names=['Normal', 'Malignant'], output_dict=True, zero_division=0)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Chest X-Ray Winner ({winner["exp_name"]})\nTest Confusion Matrix', fontweight='bold', pad=12)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Malignant'], fontsize=11)
    plt.yticks(tick_marks, ['Normal', 'Malignant'], fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            plt.text(j, i, f'{count}\n({pct:.1f}%)', ha='center', va='center',
                     color='white' if count > thresh else 'black', fontweight='bold')
    plt.ylabel('True Class', fontsize=11); plt.xlabel('Predicted Class', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / 'winner_test_confusion_matrix.png', dpi=200)
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_t, prob_t)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ROC Curve (AUC = {t_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Chance (AUC = 0.50)')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'Chest X-Ray Winner ({winner["exp_name"]}) — Test ROC', fontweight='bold', pad=12)
    plt.legend(loc='lower right'); plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig(out_dir / 'winner_test_roc_curve.png', dpi=200)
    plt.close()

    final_test_metrics = {
        'selected_experiment': winner['exp_name'],
        'selection_criterion': 'Best Validation Macro F1 score on validation partition',
        'best_validation_macro_f1': winner['best_val_macro_f1'],
        'best_validation_epoch': int(winner['best_epoch']),
        'test_accuracy': t_acc,
        'test_macro_f1': t_f1,
        'malignant_sensitivity_recall': t_rec,
        'malignant_precision': t_prec,
        'specificity_normal': t_spec,
        'roc_auc': t_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': clf_rep,
        'winner_checkpoint': winner['checkpoint_path'],
    }

    test_json = out_dir / 'winner_heldout_test_metrics.json'
    test_json.write_text(json.dumps(final_test_metrics, indent=2))

    print('\n' + '=' * 80)
    print('FINAL HELD-OUT TEST METRICS (CHOSEN MODEL EVALUATED ONCE)')
    print('=' * 80)
    print(f'Test Accuracy:         {t_acc:.4f} ({t_acc*100:.2f}%)')
    print(f'Test Macro F1:         {t_f1:.4f}')
    print(f'Malignant Sensitivity: {t_rec:.4f} ({t_rec*100:.2f}%)')
    print(f'Specificity (Normal):  {t_spec:.4f} ({t_spec*100:.2f}%)')
    print(f'Malignant Precision:   {t_prec:.4f} ({t_prec*100:.2f}%)')
    print(f'Test ROC-AUC:          {t_auc:.4f}')
    print(f'Confusion Matrix:      {cm.tolist()}')
    print(f'Saved test metrics to: {test_json}')
    print('=' * 80)


if __name__ == '__main__':
    main()
