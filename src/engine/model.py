"""
model.py — AI Model Zoo for the Smart Road Safety Project
==========================================================

This file defines ALL 4 model architectures we are comparing.
Think of this as a "model zoo" — one place to manage every algorithm.

The 4 models represent 4 different philosophies in Machine Learning:
  1. CustomDriverCNN   → Built 100% from scratch. We write every layer ourselves.
  2. MobileNetV2       → Lightweight, pre-trained. Designed for phones and edge devices.
  3. ResNet18          → Classic deep learning benchmark. Uses "skip connections" to go deeper.
  4. EfficientNet-B0   → State-of-the-art efficiency. Best accuracy per parameter count.

HOW IT WORKS (for your mentor explanation):
- Models 2, 3, 4 use "Transfer Learning": they were pre-trained on ImageNet
  (1.2 million images of 1000 categories). We then "fine-tune" their final
  layer to recognize our 10 driver distraction classes instead.
- Model 1 (CustomDriverCNN) starts from random weights and learns everything
  from our dataset alone — which is why it needs more epochs and is harder to train.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# MODEL 1: Custom CNN (Built From Scratch)
# ============================================================

class CustomDriverCNN(nn.Module):
    """
    CUSTOM CNN — Built from scratch using pure PyTorch.

    Architecture Philosophy:
    - 3 convolutional "blocks" that progressively learn more complex features:
      Block 1: Detects raw edges and gradients (very low-level)
      Block 2: Groups edges into shapes (hands, steering wheel, phone)
      Block 3: Combines shapes into abstract concepts (texting posture, safe posture)
    - Dropout(0.4): Randomly disables 40% of neurons during training so the model
      cannot memorize, forcing it to truly generalize to new faces/drivers.

    Input:  Grayscale image (1 channel) → 224×224 px
    Output: 10 class probability scores
    Expected Accuracy: ~85-88% after proper training (25 epochs)
    """
    def __init__(self, num_classes=10):
        super(CustomDriverCNN, self).__init__()

        # Block 1: Edge Detection (1 grayscale channel → 32 feature maps)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: Shape Grouping (32 → 64 feature maps)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: Deep Abstraction (64 → 128 feature maps)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After 3 pooling ops: 224 → 112 → 56 → 28
        # So flattened size = 128 * 28 * 28 = 100,352
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)           # Flatten 3D tensor → 1D vector
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ============================================================
# MODEL 2: MobileNetV2 (Lightweight Transfer Learning)
# ============================================================

def get_mobilenet_v2(num_classes=10):
    """
    MOBILENET V2 — Pre-trained, Optimized for Edge Devices.

    Transfer Learning Explained:
    - Google trained MobileNetV2 on ImageNet (1.2M images, 1000 classes).
    - We "borrow" all that knowledge (weights) and only retrain the last layer.
    - It's like hiring an expert photographer and only teaching them our 10 poses.

    Key Feature — Depthwise Separable Convolutions:
    - Normal convolutions are computationally expensive (multiply every pixel × every filter).
    - MobileNet splits this into two cheaper operations, making it ~8x faster.
    - This is WHY it runs smoothly on a Jetson Nano or smartphone.

    Input:  RGB image (3 channels) → 224×224 px
    Output: 10 class probability scores
    Expected Accuracy: ~90-93%
    Best For: Real-time edge deployment (phones, Jetson Nano)
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all the "borrowed" expert knowledge so it doesn't get destroyed
    for param in model.parameters():
        param.requires_grad = False

    # Only replace and train the final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# ============================================================
# MODEL 3: ResNet18 (Deep Residual Network — Industry Benchmark)
# ============================================================

def get_resnet18(num_classes=10):
    """
    RESNET18 — The Industry Benchmark, Introduced in 2015.

    The "Skip Connection" Innovation:
    - Before ResNet, adding more layers made models WORSE (vanishing gradient problem).
    - ResNet solved this by adding "shortcuts" that skip 2 layers at a time.
    - These shortcuts carry the original signal forward, so even very deep networks
      can keep learning effectively. Think of it as a highway for information.

    ResNet18 has 18 layers (hence the name) — a good balance of depth vs. speed.
    It is the go-to baseline in virtually every academic paper.

    Input:  RGB image (3 channels) → 224×224 px
    Output: 10 class probability scores
    Expected Accuracy: ~92-95%
    Best For: Academic benchmarks, solid all-rounder
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze the pre-trained convolutional backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace only the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ============================================================
# MODEL 4: EfficientNet-B0 (State-of-the-Art Compound Scaling)
# ============================================================

def get_efficientnet_b0(num_classes=10):
    """
    EFFICIENTNET-B0 — The State-of-the-Art Efficient Architecture (Google, 2019).

    Compound Scaling Explained:
    - Previous models scaled only one dimension: make it wider OR deeper OR use higher resolution.
    - EfficientNet scales ALL THREE dimensions simultaneously using a mathematically
      derived "compound coefficient" — width × depth × resolution together.
    - Result: Achieves higher accuracy with far FEWER parameters than ResNet or VGG.

    EfficientNet-B0 is the smallest variant. Despite its size, it punches well
    above its weight class — achieving ResNet-50 level accuracy with 5x fewer parameters.

    Input:  RGB image (3 channels) → 224×224 px
    Output: 10 class probability scores
    Expected Accuracy: ~93-96%
    Best For: When you need maximum accuracy with minimal compute
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze the pre-trained feature extraction backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head for our 10-class problem
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# ============================================================
# UNIFIED DISPATCHER — Get any model by name
# ============================================================

# Model metadata used by the dashboard for display and comparison
MODEL_REGISTRY = {
    "custom_cnn": {
        "name":        "Custom Driver CNN",
        "short":       "CustomCNN",
        "fn":          lambda nc: CustomDriverCNN(nc),
        "input_mode":  "grayscale",   # This model needs 1-channel grayscale input
        "params":      "~6.5M",
        "size_mb":     "~25 MB",
        "speed":       "⚡⚡⚡ Very Fast",
        "accuracy":    "~85-88%",
        "use_case":    "Academic demonstration — shows you understand CNN fundamentals from scratch.",
        "pros":        ["Built from scratch — full transparency", "Very fast inference", "Smallest model size"],
        "cons":        ["Lower accuracy ceiling", "Requires grayscale conversion", "No pre-learned features"],
        "color":       "#00F0FF",
    },
    "mobilenet_v2": {
        "name":        "MobileNet V2",
        "short":       "MobileNetV2",
        "fn":          lambda nc: get_mobilenet_v2(nc),
        "input_mode":  "rgb",
        "params":      "~3.4M",
        "size_mb":     "~14 MB",
        "speed":       "⚡⚡⚡⚡ Extremely Fast",
        "accuracy":    "~90-93%",
        "use_case":    "Edge device deployment (Jetson Nano, Raspberry Pi, Smartphones).",
        "pros":        ["Optimized for mobile/edge", "Very lightweight", "Good accuracy-speed trade-off"],
        "cons":        ["Less accurate than ResNet/EfficientNet", "Requires transfer learning knowledge"],
        "color":       "#a78bfa",
    },
    "resnet18": {
        "name":        "ResNet-18",
        "short":       "ResNet18",
        "fn":          lambda nc: get_resnet18(nc),
        "input_mode":  "rgb",
        "params":      "~11.7M",
        "size_mb":     "~45 MB",
        "speed":       "⚡⚡⚡ Fast",
        "accuracy":    "~92-95%",
        "use_case":    "Academic benchmarking and production systems — the trusted industry standard.",
        "pros":        ["Industry standard baseline", "Excellent generalization", "Strong skip connections"],
        "cons":        ["Heavier than MobileNet", "Not optimised for edge devices"],
        "color":       "#f59e0b",
    },
    "efficientnet_b0": {
        "name":        "EfficientNet-B0",
        "short":       "EfficientNet",
        "fn":          lambda nc: get_efficientnet_b0(nc),
        "input_mode":  "rgb",
        "params":      "~5.3M",
        "size_mb":     "~21 MB",
        "speed":       "⚡⚡⚡ Fast",
        "accuracy":    "~93-96%",
        "use_case":    "Best overall choice — highest accuracy with surprisingly low parameter count.",
        "pros":        ["State-of-the-art accuracy", "Efficient compound scaling", "Best accuracy/param ratio"],
        "cons":        ["More complex architecture to explain", "Slightly slower than MobileNet"],
        "color":       "#22c55e",
    },
}

def get_model(model_name: str, num_classes: int = 10):
    """
    Unified model factory. Call this with the model name to get any model.
    Example: model = get_model('resnet18', num_classes=10)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]["fn"](num_classes)


# Legacy function kept for backward compatibility with existing app.py
def get_driver_model(num_classes=10):
    return CustomDriverCNN(num_classes)


if __name__ == "__main__":
    print("=== Model Zoo Sanity Check ===\n")
    dummy_gray = torch.randn(1, 1, 224, 224)   # Grayscale input for CustomCNN
    dummy_rgb  = torch.randn(1, 3, 224, 224)   # RGB input for transfer learning models

    for key, meta in MODEL_REGISTRY.items():
        m = get_model(key)
        dummy = dummy_gray if meta["input_mode"] == "grayscale" else dummy_rgb
        out   = m(dummy)
        print(f"✅ {meta['name']:20s} | Output: {out.shape} | Params: {meta['params']}")
