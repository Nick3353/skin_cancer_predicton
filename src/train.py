import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model import build_model
from src.dataloader import get_dataloaders



# 1. Loss function with class weights

def get_loss_fn(task: str = 'multiclass', device: torch.device = None, outputs_dir: str = 'outputs'):
    if task == 'multiclass':
        weights = np.load(os.path.join(outputs_dir, 'multiclass_class_weights.npy'))
    else:
        weights = np.load(os.path.join(outputs_dir, 'binary_class_weights.npy'))

    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    return nn.CrossEntropyLoss(weight=weights)



# 2. One epoch of training

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────
# 3. One epoch of validation
# ─────────────────────────────────────────────────────────────
def validate(model, loader, loss_fn, device):
    model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────
# 4. Full training loop
# ─────────────────────────────────────────────────────────────
def train(
    architecture   : str   = 'resnet50',
    task           : str   = 'multiclass',
    num_classes    : int   = 9,
    epochs         : int   = 15,
    batch_size     : int   = 32,
    learning_rate  : float = 1e-3,
    patience       : int   = 5,
    freeze_backbone: bool  = True,
    train_csv      : str   = 'outputs/train.csv',
    val_csv        : str   = 'outputs/val.csv',
    checkpoint_dir : str   = 'outputs/checkpoints',
):
    # ── Setup ─────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on : {device}")
    print(f"Architecture: {architecture}")
    print(f"Task        : {task} ({num_classes} classes)")
    print(f"Epochs      : {epochs}  |  LR: {learning_rate}  |  Batch: {batch_size}")
    print("─" * 60)

    # ── Directories ───────────────────────────────────────────
    os.makedirs(checkpoint_dir, exist_ok=True)
    outputs_dir = os.path.dirname(checkpoint_dir)

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        train_csv  = train_csv,
        val_csv    = val_csv,
        test_csv   = os.path.join(outputs_dir, 'test.csv'),  # ← fixed
        task       = task,
        batch_size = batch_size,
    )

    # ── Model ─────────────────────────────────────────────────
    model = build_model(architecture, num_classes, freeze_backbone)
    model = model.to(device)

    # ── Loss, optimizer, scheduler ────────────────────────────
    loss_fn = get_loss_fn(task, device, outputs_dir)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    )

    # ── Training state ────────────────────────────────────────
    best_val_loss  = float('inf')
    patience_count = 0
    history        = {
        'train_loss': [], 'train_acc': [],
        'val_loss':   [], 'val_acc':   [],
        'lr':         [],
    }

    checkpoint_path = os.path.join(
        checkpoint_dir, f'best_{architecture}_{task}.pth'
    )

    # ── Epoch loop ────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, loss_fn, device
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        elapsed = time.time() - start

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # ── Checkpoint — save if best val loss ────────────────
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
            }, checkpoint_path)
            print(f"           ✓ Saved best model — val_loss: {val_loss:.4f}")

        else:
            patience_count += 1
            print(f"           No improvement ({patience_count}/{patience})")
            if patience_count >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print("\n" + "─" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {checkpoint_path}")

    return history, checkpoint_path