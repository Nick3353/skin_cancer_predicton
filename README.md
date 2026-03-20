skin-cancer-isic/
├── Train/                          ← we split this into train + val (80/20)
│   ├── melanoma/
│   ├── nevus/
│   ├── basal cell carcinoma/
│   ├── squamous cell carcinoma/
│   ├── actinic keratosis/
│   ├── pigmented benign keratosis/
│   ├── seborrheic keratosis/
│   ├── dermatofibroma/
│   └── vascular lesion/
│
└── Test/                           ← held out entirely, never touched during training
    └── (same 9 class folders)


    skin/
└── webapp/
    ├── app.py                  ← Flask backend
    ├── templates/
    │   └── index.html          ← frontend UI
    └── static/
        └── uploads/            ← temp image storage (auto-created)

        C:\Users\Nick\Desktop\skin\
├── dataset_ISIC\
│   ├── Train\  (9 class folders, 2239 images)
│   └── Test\   (9 class folders, 118 images)
├── src\
│   ├── __init__.py
│   ├── dataset.py
│   ├── dataloader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── notebooks\
│   ├── data_exploration.py
│   ├── 02_verify_preprocessing.py
│   ├── 03_verify_models.py
│   ├── 04_train.py
│   ├── 04b_finetune.py
│   └── 05_evaluate.py
├── webapp\
│   ├── app.py
│   └── templates\
│       └── index.html
├── outputs\
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── binary_class_weights.npy
│   ├── multiclass_class_weights.npy
│   ├── training_curves_*.png
│   ├── stage2_curves_*.png
│   ├── checkpoints\
│   │   ├── best_mobilenet_v3_multiclass_stage2.pth  ← best model
│   │   ├── best_resnet50_multiclass_stage2.pth
│   │   └── best_efficientnet_b4_multiclass_stage2.pth
│   └── evaluation\
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       ├── classification_report.txt
│       └── gradcam_samples.png
└── README.md