import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_CSV  = os.path.join(BASE_DIR, 'outputs', 'test.csv')
CKPT_DIR  = os.path.join(BASE_DIR, 'outputs', 'checkpoints')
EVAL_DIR  = os.path.join(BASE_DIR, 'outputs', 'evaluation')
os.makedirs(EVAL_DIR, exist_ok=True)

import torch
import numpy as np
from src.evaluate import (
    load_model,
    get_predictions,
    plot_confusion_matrix,
    plot_roc_curves,
    print_classification_report,
    print_sensitivity_specificity,
)
from src.gradcam import visualize_gradcam
from src.dataloader import get_dataloaders


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 60)

    #  Load best model  
    ckpt_path = os.path.join(
        CKPT_DIR, 'best_mobilenet_v3_multiclass_stage2.pth'
    )
    model, architecture = load_model(ckpt_path, device)

    #Test dataloader 
    _, _, test_loader = get_dataloaders(
        train_csv = os.path.join(BASE_DIR, 'outputs', 'train.csv'),
        val_csv   = os.path.join(BASE_DIR, 'outputs', 'val.csv'),
        test_csv  = TEST_CSV,
        task      = 'multiclass',
        batch_size= 16,
    )

    #  Get predictions  
    print("\nRunning inference on test set...")
    labels, preds, probs = get_predictions(model, test_loader, device)
    print(f"Test samples: {len(labels)}")
    acc = (labels == preds).mean()
    print(f"Test accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("=" * 60)

    #   Confusion matrix 
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        labels, preds,
        os.path.join(EVAL_DIR, 'confusion_matrix.png')
    )

    #   ROC curves  
    print("\nGenerating ROC curves...")
    aucs, macro_auc = plot_roc_curves(
        labels, probs,
        os.path.join(EVAL_DIR, 'roc_curves.png')
    )
    print(f"Macro AUC: {macro_auc:.4f}")

    #  Classification report  
    print_classification_report(
        labels, preds,
        os.path.join(EVAL_DIR, 'classification_report.txt')
    )

    #  Sensitivity & specificity 
    print_sensitivity_specificity(labels, preds, aucs)

    #   Grad-CAM  
    print("\nGenerating Grad-CAM heatmaps...")
    visualize_gradcam(
        model        = model,
        architecture = architecture,
        test_loader  = test_loader,
        device       = device,
        save_path    = os.path.join(EVAL_DIR, 'gradcam_samples.png'),
        num_samples  = 9,
    )

    print("\n" + "=" * 60)
    print("Evaluation complete. All outputs saved to:")
    print(f"  {EVAL_DIR}")


if __name__ == '__main__':
    main()