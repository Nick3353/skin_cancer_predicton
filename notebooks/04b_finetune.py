import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_CSV  = os.path.join(BASE_DIR, 'outputs', 'train.csv')
VAL_CSV    = os.path.join(BASE_DIR, 'outputs', 'val.csv')
TEST_CSV   = os.path.join(BASE_DIR, 'outputs', 'test.csv')
CKPT_DIR   = os.path.join(BASE_DIR, 'outputs', 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model import build_model
from src.dataloader import get_dataloaders


# ─────────────────────────────────────────────────────────────
# Plot helper
# ─────────────────────────────────────────────────────────────
def plot_history(history, architecture, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train', marker='o', ms=4)
    axes[0].plot(epochs, history['val_loss'],   label='Val',   marker='o', ms=4)
    axes[0].set_title(f'{architecture} Stage 2 — Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], label='Train', marker='o', ms=4)
    axes[1].plot(epochs, history['val_acc'],   label='Val',   marker='o', ms=4)
    axes[1].set_title(f'{architecture} Stage 2 — Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f'Stage 2 fine-tuning — {architecture}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Stage 2 training function
# ─────────────────────────────────────────────────────────────
def finetune(
    architecture : str,
    epochs       : int   = 10,
    batch_size   : int   = 32,
    learning_rate: float = 1e-5,   # very low — critical
    patience     : int   = 4,
    num_classes  : int   = 9,
    task         : str   = 'multiclass',
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Paths ─────────────────────────────────────────────────
    stage1_ckpt  = os.path.join(CKPT_DIR, f'best_{architecture}_{task}.pth')
    stage2_ckpt  = os.path.join(CKPT_DIR, f'best_{architecture}_{task}_stage2.pth')

    print(f"Architecture : {architecture}")
    print(f"Loading from : {stage1_ckpt}")
    print(f"Device       : {device}")
    print(f"LR           : {learning_rate}  |  Epochs: {epochs}")
    print("─" * 60)

    # ── Load Stage 1 checkpoint ───────────────────────────────
    checkpoint = torch.load(stage1_ckpt, map_location=device)
    print(f"Stage 1 best — val_loss: {checkpoint['val_loss']:.4f}  "
          f"val_acc: {checkpoint['val_acc']:.4f}")
    print("─" * 60)

    # ── Build model and load Stage 1 weights ──────────────────
    model = build_model(architecture, num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)

    # Confirm ALL parameters are now trainable
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")
    print("─" * 60)

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        train_csv  = TRAIN_CSV,
        val_csv    = VAL_CSV,
        test_csv   = TEST_CSV,
        task       = task,
        batch_size = batch_size,
    )

    # ── Loss with class weights ───────────────────────────────
    weights = np.load(os.path.join(OUTPUT_DIR, 'multiclass_class_weights.npy'))
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # ── Optimizer — very low LR for full network ──────────────
    # Use different LRs for backbone vs head:
    # backbone gets lr, head gets lr*10 — head can adapt faster
    backbone_params = []
    head_params     = []

    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = AdamW([
        {'params': backbone_params, 'lr': learning_rate},
        {'params': head_params,     'lr': learning_rate * 10},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # ── Training state ────────────────────────────────────────
    best_val_loss  = checkpoint['val_loss']   # must beat Stage 1
    patience_count = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
    }

    # ── Epoch loop ────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ──────────────────────────────────────────
        model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss   = loss_fn(logits, labels)
                total_loss += loss.item() * images.size(0)
                correct    += (logits.argmax(1) == labels).sum().item()
                total      += labels.size(0)

        val_loss = total_loss / total
        val_acc  = correct / total

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch:>2}/{epochs} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # ── Save if best ──────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                'epoch'       : epoch,
                'architecture': architecture,
                'task'        : task,
                'num_classes' : num_classes,
                'model_state' : model.state_dict(),
                'val_loss'    : val_loss,
                'val_acc'     : val_acc,
                'stage'       : 2,
            }, stage2_ckpt)
            print(f"           ✓ Saved Stage 2 best — val_loss: {val_loss:.4f}  "
                  f"acc: {val_acc:.4f}")
        else:
            patience_count += 1
            print(f"           No improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print("\n" + "─" * 60)
    print(f"Stage 2 complete. Best val loss: {best_val_loss:.4f}")
    print(f"Saved to: {stage2_ckpt}")
    return history, stage2_ckpt


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    # Start with MobileNet — fastest and already best performer
    # Then ResNet-50, then EfficientNet
    architectures = [ 'efficientnet_b4']

    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"  Stage 2 Fine-tuning: {arch}")
        print(f"{'='*60}")

        history, ckpt = finetune(
            architecture  = arch,
            epochs        = 10,
            batch_size    = 32,
            learning_rate = 1e-5,
            patience      = 4,
        )

        plot_history(
            history,
            arch,
            os.path.join(OUTPUT_DIR, f'stage2_curves_{arch}.png')
        )

        print(f"\nBest Stage 2 checkpoint: {ckpt}\n")


if __name__ == '__main__':
    main()