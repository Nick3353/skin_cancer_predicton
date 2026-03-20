import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 


import torch
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import SkinLesionDataset, train_transforms, val_test_transforms
from src.dataloader import get_dataloaders

# ── 1. Load a single batch ────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

train_loader, val_loader, test_loader = get_dataloaders(
    train_csv  = os.path.join(BASE_DIR, 'outputs', 'train.csv'),
    val_csv    = os.path.join(BASE_DIR, 'outputs', 'val.csv'),
    test_csv   = os.path.join(BASE_DIR, 'outputs', 'test.csv'),
    batch_size = 8,
    task       = 'multiclass'
)

images, labels = next(iter(train_loader))

print(f"Batch image shape : {images.shape}")
# Expected: torch.Size([8, 3, 224, 224])
# [batch, channels, height, width]

print(f"Batch label shape : {labels.shape}")
# Expected: torch.Size([8])

print(f"Pixel value range : min={images.min():.3f}  max={images.max():.3f}")
# After normalization values will be roughly -2.1 to +2.6
# (no longer 0-1 — that is correct)

print(f"Labels in batch   : {labels.tolist()}")
# e.g. [3, 0, 5, 3, 1, 4, 8, 2]

# ── 2. Visualise a batch (denormalize first) ──────────────────
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

idx_to_class = {
    0: 'actinic keratosis',
    1: 'basal cell carc.',
    2: 'dermatofibroma',
    3: 'melanoma',
    4: 'nevus',
    5: 'pig. benign kerat.',
    6: 'seborrheic kerat.',
    7: 'squamous cell carc.',
    8: 'vascular lesion',
}

MALIGNANT = {0, 1, 3, 7}

def denormalize(tensor):
    """Undo ImageNet normalization for display."""
    return (tensor * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    img = denormalize(images[i]).permute(1, 2, 0).numpy()
    label_idx = labels[i].item()
    class_name = idx_to_class[label_idx]
    color = 'red' if label_idx in MALIGNANT else 'green'
    ax.imshow(img)
    ax.set_title(class_name, color=color, fontsize=9)
    ax.axis('off')

plt.suptitle('Sample batch — red title = malignant, green = benign', y=1.01)
plt.tight_layout()
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'sample_batch.png'), dpi=150, bbox_inches='tight')
plt.show()

# ── 3. Confirm all three loaders ─────────────────────────────
print(f"\nDataLoader summary:")
print(f"  Train batches : {len(train_loader)}  "
      f"({len(train_loader.dataset)} images ÷ batch 32)")
print(f"  Val   batches : {len(val_loader)}")
print(f"  Test  batches : {len(test_loader)}")

# ── 4. Verify class balance across one full train epoch ───────
from collections import Counter

all_labels = []
for _, batch_labels in train_loader:
    all_labels.extend(batch_labels.tolist())

counts = Counter(all_labels)
print(f"\nClass counts in full train set:")
for idx in sorted(counts):
    print(f"  {idx_to_class[idx]:<30} {counts[idx]:>4}")
 

'''Expected output:
 
Batch image shape : torch.Size([8, 3, 224, 224])
Batch label shape : torch.Size([8])
Pixel value range : min=-2.118  max=2.640
Labels in batch   : [3, 0, 5, 3, 1, 4, 8, 2]

DataLoader summary:
  Train batches : 56  (1791 images ÷ batch 32)
  Val   batches : 14
  Test  batches : 4
 

---

## Key decisions explained

**Why `Resize(256)` then `CenterCrop(224)` and not just `Resize(224)`?**
Resizing directly to 224 distorts the aspect ratio — a 600×450 image becomes squashed. Resizing the shortest edge to 256 first preserves the shape, then cropping to 224 gives a clean square without distortion.

**Why no augmentation on val and test?**
Val and test measure real-world performance. If we augmented them, we'd be evaluating on artificially modified images — not representative of what a doctor would upload. We only augment train to make the model more robust.

**Why `pin_memory=True`?**
It tells PyTorch to keep the data in pinned (page-locked) memory, which makes the CPU-to-GPU transfer significantly faster during training.

**Why `shuffle=True` only on train?**
Shuffling ensures the model sees classes in random order every epoch, preventing it from learning spurious patterns from batch ordering. Val and test are never shuffled so results are reproducible.

---

## Your project structure now
```
skin_cancer_detection/
├── src/
│   ├── dataset.py       ← built today
│   └── dataloader.py    ← built today
├── notebooks/
│   ├── 01_data_exploration.py
│   └── 02_verify_preprocessing.py   ← run to verify
├── outputs/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── binary_class_weights.npy
│   └── multiclass_class_weights.npy   '''