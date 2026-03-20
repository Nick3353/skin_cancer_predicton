import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── ADD THESE 4 LINES ─────────────────────────────────────────
BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_CSV = os.path.join(BASE_DIR, 'outputs', 'train.csv')
VAL_CSV   = os.path.join(BASE_DIR, 'outputs', 'val.csv')
CKPT_DIR  = os.path.join(BASE_DIR, 'outputs', 'checkpoints')
# ─────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from src.train import train

def plot_history(history, architecture, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train', marker='o', ms=4)
    axes[0].plot(epochs, history['val_loss'],   label='Val',   marker='o', ms=4)
    axes[0].set_title(f'{architecture} — Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], label='Train', marker='o', ms=4)
    axes[1].plot(epochs, history['val_acc'],   label='Val',   marker='o', ms=4)
    axes[1].set_title(f'{architecture} — Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history['lr'], color='orange', marker='o', ms=4)
    axes[2].set_title(f'{architecture} — Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].grid(alpha=0.3)

    plt.suptitle(f'Training curves — {architecture}', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


def main():
    architectures = ['mobilenet_v3']

    for arch in architectures:
        print(f"\n{'='*60}")
        print(f"  Training: {arch}")
        print(f"{'='*60}")

        history, ckpt = train(
            architecture    = arch,
            task            = 'multiclass',
            num_classes     = 9,
            epochs          = 15,
            batch_size      = 32,
            learning_rate   = 1e-3,
            patience        = 5,
            freeze_backbone = True,
            # ── ADD THESE 3 LINES ────────────────────────────
            train_csv       = TRAIN_CSV,
            val_csv         = VAL_CSV,
            checkpoint_dir  = CKPT_DIR,
            # ─────────────────────────────────────────────────
        )

        save_dir = os.path.join(BASE_DIR, 'outputs')
        plot_history(
            history,
            arch,
            os.path.join(save_dir, f'training_curves_{arch}.png')
        )

        print(f"\nBest checkpoint: {ckpt}")


if __name__ == '__main__':
    main()