import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.model import build_model, count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

architectures = ['resnet50', 'efficientnet_b4', 'mobilenet_v3']

for arch in architectures:
    print(f"{'─'*50}")
    print(f"Model: {arch}")

    model = build_model(arch, num_classes=9, freeze_backbone=True)
    model = model.to(device)
    count_parameters(model)

    # Forward pass with a dummy batch — confirms shapes are correct
    dummy = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy)

    print(f"  Input shape          :   (4, 3, 224, 224)")
    print(f"  Output shape         :   {tuple(output.shape)}")
    # Expected: (4, 9) — 4 images, 9 class scores each
    print()