"""
train.py — Single Model Trainer (Upgraded)
==========================================

This script trains ONE model of your choice and saves:
  1. The best model weights (.pth file)
  2. A training metrics log (.json file) — used by the dashboard to plot learning curves

Upgrades from the original version:
  - Epochs: 5 → 25  (model can now fully converge)
  - LR Scheduler: Automatically halves learning rate when progress stalls
  - Early Stopping: Stops training if no improvement after N epochs (saves time)
  - Richer augmentation via upgraded dataset.py
  - JSON metrics export for dashboard comparison charts

Usage:
    python src/engine/train.py --model custom_cnn
    python src/engine/train.py --model resnet18
    python src/engine/train.py --model mobilenet_v2 --quick
"""

import sys
import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Allow running from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.engine.dataset import get_dataloaders
from src.engine.model import get_model, MODEL_REGISTRY


# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE     = 32
EPOCHS         = 25
LEARNING_RATE  = 0.001
EARLY_STOP_PAT = 5      # Stop if val_loss doesn't improve for 5 consecutive epochs


def train_model(model_name: str, quick_mode: bool = False):
    """
    Full training loop for a given model.
    Saves weights and metrics to disk on completion.
    """
    if model_name not in MODEL_REGISTRY:
        print(f"❌ Unknown model: '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
        return

    meta = MODEL_REGISTRY[model_name]
    print(f"\n{'='*55}")
    print(f"  Training: {meta['name']}")
    print(f"  Mode:     {'QUICK (5 epochs)' if quick_mode else f'FULL ({EPOCHS} epochs)'}")
    print(f"{'='*55}\n")

    # ----------------------------------------------------------
    # 1. Hardware Detection
    # ----------------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Apple Silicon MPS detected — using Metal GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 NVIDIA CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("🐢 No GPU found — using CPU (training will be slow)")

    # ----------------------------------------------------------
    # 2. Data
    # ----------------------------------------------------------
    input_mode = meta["input_mode"]
    train_loader, val_loader = get_dataloaders(
        batch_size = BATCH_SIZE,
        input_mode = input_mode,
        quick_mode = quick_mode
    )

    # ----------------------------------------------------------
    # 3. Model, Loss, Optimizer, Scheduler
    # ----------------------------------------------------------
    model     = get_model(model_name, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    # Only optimize parameters that require gradients (the trainable head)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # ReduceLROnPlateau: halves the LR if val_loss stops improving for 2 epochs
    # Think of it as: "if you're not making progress, take smaller steps"
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # ----------------------------------------------------------
    # 4. Training Loop
    # ----------------------------------------------------------
    epochs_to_run  = 5 if quick_mode else EPOCHS
    best_val_loss  = float('inf')
    no_improve     = 0    # Counter for early stopping
    metrics_log    = []   # Will be saved to JSON for dashboard charts

    for epoch in range(epochs_to_run):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"  Epoch [{epoch+1}/{epochs_to_run}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss  = val_loss / len(val_loader)
        val_accuracy  = 100 * correct / total
        epoch_time    = time.time() - epoch_start

        # Step the scheduler — it watches val_loss
        scheduler.step(avg_val_loss)

        print(f"\n✅ Epoch {epoch+1}/{epochs_to_run} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}% | "
              f"Time: {epoch_time:.1f}s\n")

        # Log this epoch's metrics for the dashboard
        metrics_log.append({
            "epoch":        epoch + 1,
            "train_loss":   round(avg_train_loss, 4),
            "val_loss":     round(avg_val_loss, 4),
            "val_accuracy": round(val_accuracy, 2),
        })

        # --- Save Best Weights ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve    = 0
            os.makedirs("./models", exist_ok=True)
            save_path = f"./models/{model_name}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"⭐ New best model saved → {save_path}\n")
        else:
            no_improve += 1
            print(f"  (No improvement for {no_improve}/{EARLY_STOP_PAT} epochs)\n")

        # --- Early Stopping ---
        if no_improve >= EARLY_STOP_PAT:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}. "
                  f"Val loss didn't improve for {EARLY_STOP_PAT} epochs.")
            break

    # ----------------------------------------------------------
    # 5. Save Metrics JSON for Dashboard
    # ----------------------------------------------------------
    os.makedirs("./results", exist_ok=True)
    results_path = f"./results/{model_name}_metrics.json"

    summary = {
        "model_name":      model_name,
        "display_name":    meta["name"],
        "input_mode":      input_mode,
        "best_val_loss":   round(best_val_loss, 4),
        "best_accuracy":   max(e["val_accuracy"] for e in metrics_log),
        "epochs_trained":  len(metrics_log),
        "history":         metrics_log,
    }

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n📊 Metrics saved → {results_path}")
    print(f"🏁 Training complete for {meta['name']}!")
    print(f"   Best Validation Accuracy: {summary['best_accuracy']:.2f}%")
    print(f"   Best Validation Loss:     {summary['best_val_loss']:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single model")
    parser.add_argument(
        "--model", type=str, default="custom_cnn",
        choices=list(MODEL_REGISTRY.keys()),
        help="Which model to train"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 5 epochs + 20%% data for fast testing"
    )
    args = parser.parse_args()
    train_model(args.model, quick_mode=args.quick)
