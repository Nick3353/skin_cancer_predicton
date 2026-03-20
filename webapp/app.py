import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import base64
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — essential for Flask
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision import transforms

from src.model import build_model
from src.gradcam import GradCAM, get_target_layer, overlay_heatmap

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CKPT_PATH  = os.path.join(
    BASE_DIR, 'outputs', 'checkpoints',
    'best_mobilenet_v3_multiclass_stage2.pth'
)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

CLASS_NAMES = [
    'Actinic Keratosis',
    'Basal Cell Carcinoma',
    'Dermatofibroma',
    'Melanoma',
    'Nevus',
    'Pigmented Benign Keratosis',
    'Seborrheic Keratosis',
    'Squamous Cell Carcinoma',
    'Vascular Lesion',
]

MALIGNANT_IDX = {0, 1, 3, 7}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────
# Load model once at startup
# ─────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading model from: {CKPT_PATH}")

checkpoint   = torch.load(CKPT_PATH, map_location=device)
architecture = checkpoint['architecture']
num_classes  = checkpoint['num_classes']

model = build_model(architecture, num_classes, freeze_backbone=False)
model.load_state_dict(checkpoint['model_state'])
model = model.to(device)
model.eval()

target_layer = get_target_layer(model, architecture)
gradcam      = GradCAM(model, target_layer)

print(f"Model loaded: {architecture}  |  Device: {device}")

# ─────────────────────────────────────────────────────────────
# Image transform
# ─────────────────────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def image_to_base64(img_array):
    """Converts a numpy image array to a base64 PNG string for HTML display."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img_array)
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # ── Validate upload ───────────────────────────────────────
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File must be JPG or PNG'}), 400

    # ── Load and preprocess image ─────────────────────────────
    image_pil = Image.open(file.stream).convert('RGB')
    input_tensor = inference_transform(image_pil).unsqueeze(0).to(device)

    # ── Run inference ─────────────────────────────────────────
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx  = int(probs.argmax())
    pred_prob = float(probs[pred_idx])

    # ── Generate Grad-CAM ─────────────────────────────────────
    heatmap, _, _ = gradcam.generate(input_tensor, class_idx=pred_idx)

    # Prepare display image (denormalized)
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(3,1,1)
    img_display = (input_tensor[0].cpu() * std + mean).clamp(0,1)
    img_np      = img_display.permute(1,2,0).numpy()

    # Overlay
    blended = overlay_heatmap(img_np, heatmap, alpha=0.5)

    # Convert both images to base64 for HTML
    original_b64 = image_to_base64(img_np)
    gradcam_b64  = image_to_base64(blended)

    # Build top-5 predictions
    top5_idx   = probs.argsort()[::-1][:5]
    top5 = [
        {
            'class'     : CLASS_NAMES[i],
            'probability': round(float(probs[i]) * 100, 1),
            'malignant' : i in MALIGNANT_IDX,
        }
        for i in top5_idx
    ]

    return jsonify({
        'prediction'  : CLASS_NAMES[pred_idx],
        'probability' : round(pred_prob * 100, 1),
        'is_malignant': pred_idx in MALIGNANT_IDX,
        'top5'        : top5,
        'original_img': original_b64,
        'gradcam_img' : gradcam_b64,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)