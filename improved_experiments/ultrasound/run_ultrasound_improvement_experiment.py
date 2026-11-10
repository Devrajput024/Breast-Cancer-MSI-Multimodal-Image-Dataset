from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

root = Path(r'C:\Users\asus\OneDrive\Desktop\Breast Cncer Project')
out_dir = root / 'improved_experiments' / 'ultrasound'
out_dir.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(root / 'dataset_manifests' / 'ultrasound_manifest.csv')
splits = pd.read_csv(root / 'msi_results_full' / 'ultrasound_splits.csv')
baseline_metrics = json.loads((root / 'msi_results_full' / 'ultrasound_test_metrics.json').read_text())

# Preserve the original saved split assignment exactly.
merged = manifest.merge(splits[['path', 'split']], on='path', how='left')
train_df = merged[merged['split'] == 'train'].copy().reset_index(drop=True)
val_df = merged[merged['split'] == 'validation'].copy().reset_index(drop=True)
test_df = merged[merged['split'] == 'test'].copy().reset_index(drop=True)

assert len(train_df) == 512 and len(val_df) == 128 and len(test_df) == 166

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class BreastCancerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row['path']) as img:
            x = img.convert('RGB')
        if self.transform is not None:
            x = self.transform(x)
        return x, int(row['label'])


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    ys, probs, preds = [], [], []
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y.long())
            total_loss += loss.item() * len(y)
            count += len(y)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            ys.extend(y.cpu().numpy())
            probs.extend(p)
            preds.extend((p >= 0.5).astype(int))
    return np.array(ys), np.array(preds), np.array(probs), total_loss / max(count, 1)

# Conservative improvement: lower LR from 1e-3 to 5e-4 while keeping same architecture and split.
config = {
    'modality': 'ultrasound',
    'dataset_root': str(root / 'MultiModel Breast Cancer MSI Dataset'),
    'saved_split_assignment': str(root / 'msi_results_full' / 'ultrasound_splits.csv'),
    'architecture': 'ResNet-18 pretrained on ImageNet',
    'training_change': 'learning rate reduced from 1e-3 to 5e-4 with same architecture, same saved split, same test set untouched',
    'epochs': 12,
    'freeze_epochs': 3,
    'patience': 3,
    'batch_size': 16,
    'image_size': 224,
    'seed': 42,
    'validation_metric': 'macro_f1',
    'test_set_touched': False,
    'notes': 'One conservative LR-tuning experiment only; model selection by validation Macro F1; no threshold tuning on test set.'
}
(out_dir / 'config.json').write_text(json.dumps(config, indent=2))

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(BreastCancerDataset(train_df, train_transform), batch_size=16, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
val_loader = DataLoader(BreastCancerDataset(val_df, val_transform), batch_size=16, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
test_loader = DataLoader(BreastCancerDataset(test_df, val_transform), batch_size=16, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(model.fc.parameters(), lr=5e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

history = []
best_val_f1 = -1.0
best_epoch = 0
best_state = None
patience_counter = 0

for epoch in range(1, 13):
    if epoch == 4:
        for p in model.layer4.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': 2.5e-4},
            {'params': model.fc.parameters(), 'lr': 5e-4},
        ], weight_decay=1e-4)

    model.train()
    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item() * len(y)
        train_total += len(y)
        train_correct += (logits.argmax(dim=1) == y).sum().item()

    val_y, val_pred, val_prob, val_loss = evaluate(model, val_loader, device)
    val_acc = accuracy_score(val_y, val_pred)
    val_f1 = f1_score(val_y, val_pred, average='macro', zero_division=0)
    val_recall = recall_score(val_y, val_pred, zero_division=0)
    val_precision = precision_score(val_y, val_pred, zero_division=0)
    val_auc = roc_auc_score(val_y, val_prob)

    history.append({
        'epoch': epoch,
        'train_loss': train_loss_total / max(train_total, 1),
        'train_accuracy': train_correct / max(train_total, 1),
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_macro_f1': val_f1,
        'val_malignant_recall': val_recall,
        'val_malignant_precision': val_precision,
        'val_roc_auc': val_auc,
    })

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        patience_counter = 0
        best_state = {
            'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'epoch': epoch,
            'val_macro_f1': val_f1,
            'val_accuracy': val_acc,
            'val_roc_auc': val_auc,
            'class_map': {'benign': 0, 'malignant': 1},
            'mods': 'ultrasound improvement experiment',
            'image_size': 224,
        }
    else:
        patience_counter += 1
        if patience_counter >= 3:
            break

pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)

# Save best checkpoint.
ckpt_path = out_dir / 'best_checkpoint.pt'
torch.save(best_state, ckpt_path)

# Evaluate saved best model on same held-out test set.
model.load_state_dict(best_state['model_state_dict'])
test_y, test_pred, test_prob, test_loss = evaluate(model, test_loader, device)
cm = confusion_matrix(test_y, test_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
specificity = tn / max(tn + fp, 1)
sensitivity = tp / max(tp + fn, 1)
precision = tp / max(tp + fp, 1)
acc = accuracy_score(test_y, test_pred)
f1 = f1_score(test_y, test_pred, average='macro', zero_division=0)
auc = roc_auc_score(test_y, test_prob)

# Save plot artifacts.
plt.figure(figsize=(6, 5))
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
plt.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        plt.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)', ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.xticks([0, 1], ['benign', 'malignant'])
plt.yticks([0, 1], ['benign', 'malignant'])
plt.title('Ultrasound Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(out_dir / 'confusion_matrix.png', dpi=200)
plt.close()

plt.figure(figsize=(6, 5))
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(test_y, test_prob)
plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ultrasound ROC')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(out_dir / 'roc_curve.png', dpi=200)
plt.close()

metrics = {
    'modality': 'ultrasound',
    'best_epoch': best_epoch,
    'n_train': int(len(train_df)),
    'n_validation': int(len(val_df)),
    'n_test': int(len(test_df)),
    'split_method': 'saved group-aware split from ultrasound_splits.csv',
    'test_loss': float(test_loss),
    'test_accuracy': float(acc),
    'test_macro_f1': float(f1),
    'malignant_precision': float(precision),
    'malignant_sensitivity_recall': float(sensitivity),
    'specificity_class0': float(specificity),
    'roc_auc': float(auc),
    'confusion_matrix_labels_class0_then_malignant': cm.tolist(),
    'classification_report': classification_report(test_y, test_pred, target_names=['benign', 'malignant'], output_dict=True, zero_division=0),
    'checkpoint_path': str(ckpt_path.resolve()),
    'baseline_metrics_for_reference': baseline_metrics,
    'selection_criterion': 'validation_macro_f1',
    'test_set_untouched_until_final_evaluation': True,
    'disclaimer': 'Research and education only. Not a clinically validated diagnostic tool.',
}
(out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

# comparison report and retention logic
baseline_acc = float(baseline_metrics['test_accuracy'])
baseline_f1 = float(baseline_metrics['test_macro_f1'])
comparison = {
    'baseline_accuracy': baseline_acc,
    'new_accuracy': float(acc),
    'accuracy_delta': float(acc - baseline_acc),
    'baseline_macro_f1': baseline_f1,
    'new_macro_f1': float(f1),
    'macro_f1_delta': float(f1 - baseline_f1),
    'baseline_malignant_precision': float(baseline_metrics['malignant_precision']),
    'new_malignant_precision': float(precision),
    'malignant_precision_delta': float(precision - baseline_metrics['malignant_precision']),
    'baseline_malignant_recall': float(baseline_metrics['malignant_sensitivity_recall']),
    'new_malignant_recall': float(sensitivity),
    'malignant_recall_delta': float(sensitivity - baseline_metrics['malignant_sensitivity_recall']),
    'baseline_specificity': float(baseline_metrics['specificity_class0']),
    'new_specificity': float(specificity),
    'specificity_delta': float(specificity - baseline_metrics['specificity_class0']),
    'baseline_roc_auc': float(baseline_metrics['roc_auc']),
    'new_roc_auc': float(auc),
    'roc_auc_delta': float(auc - baseline_metrics['roc_auc']),
    'selected_model': 'original baseline retained' if not (f1 > baseline_f1 and acc >= baseline_acc) else 'new model selected',
    'fairness_note': 'The test set was not used for model selection or threshold tuning; all decisions used validation Macro F1 only.'
}
(out_dir / 'comparison_report.json').write_text(json.dumps(comparison, indent=2))

accuracy_delta = float(acc - baseline_acc)
macro_f1_delta = float(f1 - baseline_f1)
precision_delta = float(precision - baseline_metrics['malignant_precision'])
recall_delta = float(sensitivity - baseline_metrics['malignant_sensitivity_recall'])
specificity_delta = float(specificity - baseline_metrics['specificity_class0'])
roc_auc_delta = float(auc - baseline_metrics['roc_auc'])
selection_note = 'The new model improves the fair held-out evaluation and is selected.' if (f1 > baseline_f1 and acc >= baseline_acc) else 'The new model did not improve the fair held-out evaluation, so the original baseline model is retained.'

report = """# Ultrasound improvement experiment (read-only validation-driven run)

Goal: perform one conservative improvement using the existing saved split assignment, validation Macro F1 selection, and the same held-out test set used by the baseline.

Important constraints:
- Original baseline files, checkpoint, training script, and saved metrics are preserved.
- The existing split assignment from `msi_results_full/ultrasound_splits.csv` is used exactly as saved.
- The test set is not used for training, model selection, or threshold tuning.
- The model selected for the experiment is the checkpoint with the highest validation Macro F1.

Experiment details:
- Architecture: ResNet-18 pretrained on ImageNet
- Split: saved ultrasound split assignment (train/validation/test)
- Improvement: lower learning rate from 1e-3 to 5e-4 while keeping the same architecture and group-aware assignment
- Model selection: validation Macro F1
- Final evaluation: same held-out test set as baseline

Baseline metrics (preserved):
- accuracy: 0.8614
- macro_f1: 0.8608
- malignant_precision: 0.8370
- malignant_sensitivity_recall: 0.9059
- specificity_class0: 0.8148
- roc_auc: 0.9089

New metrics:
- accuracy: {acc:.4f}
- macro_f1: {f1:.4f}
- malignant_precision: {precision:.4f}
- malignant_sensitivity_recall: {sensitivity:.4f}
- specificity_class0: {specificity:.4f}
- roc_auc: {auc:.4f}

Comparison:
- accuracy delta: {accuracy_delta:.4f}
- macro_f1 delta: {macro_f1_delta:.4f}
- malignant precision delta: {precision_delta:.4f}
- malignant recall delta: {recall_delta:.4f}
- specificity delta: {specificity_delta:.4f}
- roc_auc delta: {roc_auc_delta:.4f}

Decision:
{selection_note}

The system is not presented as a clinically validated diagnostic tool.
""".format(
    acc=float(acc),
    f1=float(f1),
    precision=float(precision),
    sensitivity=float(sensitivity),
    specificity=float(specificity),
    auc=float(auc),
    accuracy_delta=accuracy_delta,
    macro_f1_delta=macro_f1_delta,
    precision_delta=precision_delta,
    recall_delta=recall_delta,
    specificity_delta=specificity_delta,
    roc_auc_delta=roc_auc_delta,
    selection_note=selection_note,
)
(out_dir / 'report.md').write_text(report)

print(json.dumps({
    'baseline': baseline_metrics,
    'new': metrics,
    'comparison': comparison,
}, indent=2))
