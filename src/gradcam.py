import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms


# ─────────────────────────────────────────────────────────────
# Grad-CAM implementation
# ─────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    How it works:
    1. Run a forward pass — record the feature maps at the
       last convolutional layer
    2. Run a backward pass for the predicted class —
       record the gradients at the same layer
    3. Average the gradients spatially → importance weights
    4. Weight the feature maps by those importances
    5. ReLU the result → only keep positive activations
    6. Resize to input image size → heatmap
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        """Hooks capture gradients and activations during forward/backward."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        Generates a Grad-CAM heatmap for the given input.

        Args:
            input_tensor : preprocessed image tensor (1, 3, 224, 224)
            class_idx    : class to explain. None = use predicted class

        Returns:
            heatmap : numpy array (224, 224) values in [0, 1]
            pred_idx: predicted class index
            pred_prob: predicted class probability
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        probs  = torch.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        pred_prob = probs[0, class_idx].item()

        # Backward pass for target class
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool the gradients → weights (C,)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weight the activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam = torch.relu(cam)  # only positive influence

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Resize to 224×224
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize((224, 224), Image.BILINEAR)
        heatmap = np.array(cam_img) / 255.0

        return heatmap, class_idx, pred_prob


# ─────────────────────────────────────────────────────────────
# Get the correct target layer per architecture
# ─────────────────────────────────────────────────────────────
def get_target_layer(model, architecture: str):
    """
    Returns the last convolutional layer for each architecture.
    This is where Grad-CAM hooks are registered.
    """
    arch = architecture.lower()
    if arch == 'resnet50':
        return model.layer4[-1].conv3
    elif arch == 'efficientnet_b4':
        return model.features[-1][0]
    elif arch == 'mobilenet_v3':
        return model.features[-1][0]
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ─────────────────────────────────────────────────────────────
# Overlay heatmap on image
# ─────────────────────────────────────────────────────────────
def overlay_heatmap(image_np, heatmap, alpha=0.5):
    """
    Blends a Grad-CAM heatmap over the original image.

    Args:
        image_np : numpy array (224, 224, 3) in [0, 1]
        heatmap  : numpy array (224, 224) in [0, 1]
        alpha    : heatmap opacity

    Returns:
        blended numpy array (224, 224, 3)
    """
    colormap   = plt.get_cmap('jet')
    heatmap_rgb = colormap(heatmap)[:, :, :3]  # drop alpha channel
    blended     = (1 - alpha) * image_np + alpha * heatmap_rgb
    return np.clip(blended, 0, 1)


# ─────────────────────────────────────────────────────────────
# Visualise Grad-CAM for a batch of test images
# ─────────────────────────────────────────────────────────────
def visualize_gradcam(
    model,
    architecture,
    test_loader,
    device,
    save_path,
    num_samples=9,
):
    CLASS_NAMES = [
        'actinic kerat.', 'basal cell carc.', 'dermatofibroma',
        'melanoma',        'nevus',             'pig. benign k.',
        'seborrheic k.',   'squamous cell c.',  'vascular lesion',
    ]
    MALIGNANT_IDX = {0, 1, 3, 7}

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    target_layer = get_target_layer(model, architecture)
    gradcam      = GradCAM(model, target_layer)

    # Collect images and labels from test loader
    images_list, labels_list = [], []
    for images, labels in test_loader:
        images_list.append(images)
        labels_list.append(labels)
        if sum(len(l) for l in labels_list) >= num_samples:
            break

    all_images = torch.cat(images_list)[:num_samples]
    all_labels = torch.cat(labels_list)[:num_samples]

    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 3.5))

    for i in range(num_samples):
        row = i // cols
        col = (i % cols) * 2

        input_tensor = all_images[i].unsqueeze(0).to(device)
        true_label   = all_labels[i].item()

        # Generate heatmap
        heatmap, pred_idx, pred_prob = gradcam.generate(input_tensor)

        # Denormalize image for display
        img_display = (all_images[i] * IMAGENET_STD + IMAGENET_MEAN)
        img_display = img_display.clamp(0, 1).permute(1,2,0).numpy()

        # Overlay
        blended = overlay_heatmap(img_display, heatmap, alpha=0.45)

        correct = pred_idx == true_label
        color   = 'green' if correct else 'red'

        # Original image
        axes[row][col].imshow(img_display)
        axes[row][col].set_title(
            f'True: {CLASS_NAMES[true_label]}',
            fontsize=7, color='black'
        )
        axes[row][col].axis('off')

        # Grad-CAM overlay
        axes[row][col+1].imshow(blended)
        axes[row][col+1].set_title(
            f'Pred: {CLASS_NAMES[pred_idx]} ({pred_prob:.2f})',
            fontsize=7, color=color
        )
        axes[row][col+1].axis('off')

    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = (i % cols) * 2
        if row < len(axes) and col < len(axes[row]):
            axes[row][col].axis('off')
            axes[row][col+1].axis('off')

    plt.suptitle(
        'Grad-CAM — original (left) vs heatmap overlay (right)\n'
        'green title = correct · red title = wrong',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")