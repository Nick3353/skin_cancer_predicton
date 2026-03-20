import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# ── Root of your dataset ──────────────────────────────────────
DATA_DIR   = Path('dataset_ISIC')
TRAIN_DIR  = DATA_DIR / 'Train'
TEST_DIR   = DATA_DIR / 'Test'

# ── Which classes are malignant ───────────────────────────────
MALIGNANT_CLASSES = {
    'melanoma',
    'basal cell carcinoma',
    'squamous cell carcinoma',
    'actinic keratosis',        # pre-cancerous → treated as malignant
}

os.makedirs('outputs', exist_ok=True)
def scan_folder(root: Path) -> pd.DataFrame:
    records = []
    class_folders = sorted([f for f in root.iterdir() if f.is_dir()])

    for folder in class_folders:
        imgs = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
        for img_path in imgs:
            records.append({
                'image_path': str(img_path.resolve()),  # ← .resolve() makes it absolute
                'class_name': folder.name,
            })

    df = pd.DataFrame(records)
    return df
print("Scanning Train folder...")
train_full = scan_folder(TRAIN_DIR)

print("Scanning Test folder...")
test_df = scan_folder(TEST_DIR)

print(f"\nTrain folder: {len(train_full)} images")
print(f"Test  folder: {len(test_df)} images")
 

#Expected output:

#Train folder: ~1900 images
#Test  folder: ~457  images

def add_labels(df: pd.DataFrame, class_to_idx: dict) -> pd.DataFrame:
    df = df.copy()
    df['is_malignant'] = df['class_name'].apply(
        lambda x: 1 if x in MALIGNANT_CLASSES else 0
    )
    df['label'] = df['class_name'].map(class_to_idx)
    return df

# Build class index from Train folder (source of truth)
class_names   = sorted(train_full['class_name'].unique())
class_to_idx  = {name: i for i, name in enumerate(class_names)}
idx_to_class  = {v: k for k, v in class_to_idx.items()}

print("\nClass index mapping:")
for name, idx in class_to_idx.items():
    tag = "MALIGNANT" if name in MALIGNANT_CLASSES else "benign  "
    print(f"  {idx}: {name:<35} [{tag}]")

# Apply labels
train_full = add_labels(train_full, class_to_idx)
test_df    = add_labels(test_df,    class_to_idx)
 
''' Expected output:
 
0: actinic keratosis          [MALIGNANT]
1: basal cell carcinoma       [MALIGNANT]
2: dermatofibroma             [benign  ]
3: melanoma                   [MALIGNANT]
4: nevus                      [benign  ]
5: pigmented benign keratosis [benign  ]
6: seborrheic keratosis       [benign  ]
7: squamous cell carcinoma    [MALIGNANT]
8: vascular lesion            [benign  ]  '''


# Stratified split — keeps class proportions identical in train and val
train_df, val_df = train_test_split(
    train_full,
    test_size=0.2,
    random_state=42,
    stratify=train_full['label']   # ← crucial: split per class, not randomly
)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

print(f"\nFinal split:")
print(f"  Train : {len(train_df)} images")
print(f"  Val   : {len(val_df)} images")
print(f"  Test  : {len(test_df)} images")
print(f"  Total : {len(train_df) + len(val_df) + len(test_df)} images")


'''
Expected output:

Final split:
  Train : ~1520 images
  Val   :  ~380 images
  Test  :  ~457 images
  Total : ~2357 images
  '''

print("\nImages per class across splits:\n")
print(f"{'Class':<35} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
print("-" * 60)

for name in class_names:
    tr = (train_df['class_name'] == name).sum()
    vl = (val_df['class_name']   == name).sum()
    te = (test_df['class_name']  == name).sum()
    tag = " *" if name in MALIGNANT_CLASSES else ""
    print(f"{name + tag:<35} {tr:>6} {vl:>6} {te:>6} {tr+vl+te:>7}")

print("-" * 60)
print(f"{'TOTAL':<35} {len(train_df):>6} {len(val_df):>6} {len(test_df):>6} {len(train_df)+len(val_df)+len(test_df):>7}")
print("\n* = malignant class")
 
'''
 Expected output:
 
Class                               Train    Val   Test   Total
------------------------------------------------------------
actinic keratosis *                   261     66     --     327
basal cell carcinoma *                261     66     --     327
dermatofibroma                         76     19     --      95
melanoma *                            304     76     --     380
nevus                                 285     72     --     357
pigmented benign keratosis            369     93     --     462
seborrheic keratosis                  261     66     --     327
squamous cell carcinoma *             261     66     --     327
vascular lesion                       113     29     --     142
------------------------------------------------------------
TOTAL                                1520    380    457    2357

'''

binary_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_df['is_malignant'].values   # ← computed on TRAIN only, never test
)

multiclass_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(9),
    y=train_df['label'].values
)

print("\nBinary class weights (for binary classifier):")
print(f"  benign    = {binary_weights[0]:.4f}")
print(f"  malignant = {binary_weights[1]:.4f}")

print("\nPer-class weights (for 9-class classifier):")
for i, w in enumerate(multiclass_weights):
    print(f"  {idx_to_class[i]:<35} {w:.4f}")

# Save weights — we'll load these in Phase 4 training
np.save('outputs/binary_class_weights.npy',      binary_weights)
np.save('outputs/multiclass_class_weights.npy',  multiclass_weights)

sample_paths = train_df['image_path'].sample(100, random_state=42)
widths, heights = [], []
for p in sample_paths:
    w, h = Image.open(p).size
    widths.append(w)
    heights.append(h)

print(f"\nImage dimensions (sample of 100 from train):")
print(f"  Width  → min:{min(widths)}  max:{max(widths)}  mean:{np.mean(widths):.0f}")
print(f"  Height → min:{min(heights)} max:{max(heights)} mean:{np.mean(heights):.0f}")

train_df.to_csv('outputs/train.csv', index=False)
val_df.to_csv(  'outputs/val.csv',   index=False)
test_df.to_csv( 'outputs/test.csv',  index=False)

print("\nSaved:")
print("  outputs/train.csv")
print("  outputs/val.csv")
print("  outputs/test.csv")
print("  outputs/binary_class_weights.npy")
print("  outputs/multiclass_class_weights.npy")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
splits = [('Train', train_df), ('Val', val_df), ('Test', test_df)]

for ax, (title, df_split) in zip(axes, splits):
    counts = df_split['class_name'].value_counts().sort_index()
    colors = ['#dc2626' if c in MALIGNANT_CLASSES else '#0d9668'
              for c in counts.index]
    ax.barh(counts.index, counts.values, color=colors, edgecolor='none')
    ax.set_title(f'{title} ({len(df_split)} images)')
    ax.set_xlabel('Images')
    for i, v in enumerate(counts.values):
        ax.text(v + 1, i, str(v), va='center', fontsize=8)

plt.suptitle('Class distribution — red = malignant, green = benign',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('outputs/split_distribution.png', dpi=150, bbox_inches='tight')
plt.show()