import torch
import torch.nn as nn
from torchvision import models


# Configuration

NUM_CLASSES_MULTI  = 9   # 9-class task
NUM_CLASSES_BINARY = 2   # binary task (malignant vs benign)



# 1. ResNet-50

def build_resnet50(num_classes: int = 9, freeze_backbone: bool = True):
    """
    ResNet-50 with pretrained ImageNet weights.
    Replaces the final FC layer with a new classification head.

    Architecture:
        Conv1 → BatchNorm → ReLU → MaxPool
        → Layer1 (3 blocks)
        → Layer2 (4 blocks)
        → Layer3 (6 blocks)
        → Layer4 (3 blocks)
        → AdaptiveAvgPool
        → FC(2048 → num_classes)   ← this is what we replace
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    if freeze_backbone:
        # Freeze all layers — only the new head will train initially
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer
    # Original: FC(2048 → 1000) for ImageNet's 1000 classes
    # Ours:     FC(2048 → num_classes)
    in_features = model.fc.in_features   # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),               # regularisation — prevents overfitting
        nn.Linear(in_features, num_classes)
    )

    return model



# 2. EfficientNet-B4

def build_efficientnet_b4(num_classes: int = 9, freeze_backbone: bool = True):
    """
    EfficientNet-B4 with pretrained ImageNet weights.
    Replaces the classifier head with our own.

    Architecture uses compound scaling — width, depth, and
    resolution are all scaled together for optimal efficiency.
    Final classifier: Dropout → FC(1792 → num_classes)
    """
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # EfficientNet's classifier is model.classifier
    # Original: [Dropout(0.4), Linear(1792 → 1000)]
    in_features = model.classifier[1].in_features   # 1792
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )

    return model



# 3. MobileNet-V3 Large

def build_mobilenet_v3(num_classes: int = 9, freeze_backbone: bool = True):
    """
    MobileNet-V3-Large with pretrained ImageNet weights.
    Lightest model — best for deployment and fast inference.

    Uses depthwise separable convolutions + squeeze-and-excitation
    blocks for efficiency.
    Final classifier: FC(960→1280) → Hardswish → Dropout → FC(1280→num_classes)
    """
    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.DEFAULT
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # MobileNet's classifier is a Sequential of 3 layers
    # We keep the first two (Linear 960→1280 + Hardswish) and replace only
    # the final Linear(1280→1000)
    in_features = model.classifier[3].in_features   # 1280
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model



# 4. Unified build function

def build_model(
    architecture   : str  = 'resnet50',
    num_classes    : int  = 9,
    freeze_backbone: bool = True,
):
    """
    Single entry point to build any of the three models.

    Args:
        architecture    : 'resnet50' | 'efficientnet_b4' | 'mobilenet_v3'
        num_classes     : 9 for multiclass, 2 for binary
        freeze_backbone : True = only train the head (faster, less data needed)
                          False = train all layers (better accuracy, needs more data)
    Returns:
        model (nn.Module)
    """
    architecture = architecture.lower()

    if architecture == 'resnet50':
        return build_resnet50(num_classes, freeze_backbone)
    elif architecture == 'efficientnet_b4':
        return build_efficientnet_b4(num_classes, freeze_backbone)
    elif architecture == 'mobilenet_v3':
        return build_mobilenet_v3(num_classes, freeze_backbone)
    else:
        raise ValueError(
            f"Unknown architecture: '{architecture}'. "
            f"Choose from: 'resnet50', 'efficientnet_b4', 'mobilenet_v3'"
        )



# 5. Helper — count trainable parameters

def count_parameters(model: nn.Module):
    """Prints total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"  Total parameters     : {total:>12,}")
    print(f"  Trainable parameters : {trainable:>12,}")
    print(f"  Frozen parameters    : {frozen:>12,}")
    print(f"  % being trained      : {100*trainable/total:>11.1f}%")