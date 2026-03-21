import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.model import build_model
from src.dataloader import get_dataloaders



# Class names in label order (0-8)
 
CLASS_NAMES = [
    'actinic keratosis',       # 0 MALIGNANT
    'basal cell carcinoma',    # 1 MALIGNANT
    'dermatofibroma',          # 2 benign
    'melanoma',                # 3 MALIGNANT
    'nevus',                   # 4 benign
    'pigmented benign kerat.', # 5 benign
    'seborrheic keratosis',    # 6 benign
    'squamous cell carc.',     # 7 MALIGNANT
    'vascular lesion',         # 8 benign
]

SHORT_NAMES = ['AK','BCC','DF','MEL','NV','PBK','SK','SCC','VASC']
MALIGNANT_IDX = {0, 1, 3, 7}



# 1. Load model from checkpoint

def load_model(checkpoint_path: str, device: torch.device):
    checkpoint  = torch.load(checkpoint_path, map_location=device)
    architecture = checkpoint['architecture']
    num_classes  = checkpoint['num_classes']

    model = build_model(architecture, num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()

    print(f"Loaded : {architecture}")
    print(f"Epoch  : {checkpoint['epoch']}")
    print(f"Val loss: {checkpoint['val_loss']:.4f}  "
          f"Val acc: {checkpoint['val_acc']:.4f}")
    return model, architecture



# 2. Run inference on entire test set

def get_predictions(model, test_loader, device):
    all_labels  = []
    all_preds   = []
    all_probs   = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )



# 3. Confusion matrix

def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)

    # Normalise — show % of true class correctly predicted
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, data, title, fmt in zip(
        axes,
        [cm,     cm_norm],
        ['Confusion matrix — counts', 'Confusion matrix — normalised'],
        ['d',    '.2f'],
    ):
        im = ax.imshow(data, cmap='Blues')
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels(SHORT_NAMES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(SHORT_NAMES, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(title, fontsize=12, pad=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(9):
            for j in range(9):
                val = data[i, j]
                txt = f'{val:{fmt}}'
                color = 'white' if data[i, j] > data.max() * 0.6 else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=8, color=color)

    plt.suptitle('Model evaluation — test set', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")
    return cm



# 4. ROC curves

def plot_roc_curves(labels, probs, save_path):
    n_classes = 9
    labels_bin = label_binarize(labels, classes=range(n_classes))

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes = axes.flat

    aucs = []
    for i, ax in enumerate(axes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        auc = roc_auc_score(labels_bin[:, i], probs[:, i])
        aucs.append(auc)

        color = '#dc2626' if i in MALIGNANT_IDX else '#0d9668'
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'AUC = {auc:.3f}')
        ax.plot([0,1],[0,1], 'k--', lw=0.8, alpha=0.4)
        ax.set_title(SHORT_NAMES[i], fontsize=10)
        ax.set_xlabel('FPR', fontsize=8)
        ax.set_ylabel('TPR', fontsize=8)
        ax.legend(fontsize=8)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1.02])
        tag = ' ★' if i in MALIGNANT_IDX else ''
        ax.set_title(f'{SHORT_NAMES[i]}{tag}', fontsize=10,
                     color='#dc2626' if i in MALIGNANT_IDX else 'black')

    macro_auc = np.mean(aucs)
    fig.suptitle(
        f'ROC curves — macro AUC: {macro_auc:.3f}  '
        f'(red★ = malignant classes)',
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")
    return aucs, macro_auc

# 5. Classification report

def print_classification_report(labels, preds, save_path):
    report = classification_report(
        labels, preds,
        target_names=SHORT_NAMES,
        digits=3
    )
    print("\nClassification Report:")
    print("─" * 60)
    print(report)

    # Save to file
    with open(save_path, 'w') as f:
        f.write("Skin Cancer Detection — Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"Saved: {save_path}")
    return report


# 6. Per-class sensitivity (recall) and specificity

def print_sensitivity_specificity(labels, preds, aucs):
    cm = confusion_matrix(labels, preds)
    print("\nPer-class sensitivity & specificity:")
    print("─" * 60)
    print(f"{'Class':<26} {'Sens':>6} {'Spec':>6} "
          f"{'AUC':>6} {'Type':>10}")
    print("─" * 60)

    for i in range(9):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
        class_type   = 'MALIGNANT' if i in MALIGNANT_IDX else 'benign'

        print(f"{SHORT_NAMES[i]:<26} "
              f"{sensitivity:>6.3f} "
              f"{specificity:>6.3f} "
              f"{aucs[i]:>6.3f} "
              f"{class_type:>10}")