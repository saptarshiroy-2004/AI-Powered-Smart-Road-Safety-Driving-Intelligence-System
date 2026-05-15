"""
train_all.py — Batch Trainer for All 4 Models
==============================================

Run this script ONCE to train and benchmark all 4 models sequentially.
Each model is trained on the same data split for a fair comparison.
Results are saved to ./results/<model_name>_metrics.json

Usage:
    python src/engine/train_all.py           # Full training (25 epochs each)
    python src/engine/train_all.py --quick   # Quick mode (5 epochs, 20% data)

After this finishes, launch the Streamlit dashboard to see the comparison:
    streamlit run app.py
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.engine.train import train_model
from src.engine.model import MODEL_REGISTRY

ALL_MODELS = list(MODEL_REGISTRY.keys())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all 4 models for comparison")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 5 epochs + 20%% data for fast testing"
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Specific models to train (default: all 4)"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  🚀 SMART ROAD SAFETY — MULTI-MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"  Models to train: {', '.join(args.models)}")
    print(f"  Mode: {'QUICK (5 epochs)' if args.quick else 'FULL (25 epochs)'}")
    print("="*60 + "\n")

    for i, model_name in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] Starting: {MODEL_REGISTRY[model_name]['name']}")
        print("-"*60)
        train_model(model_name, quick_mode=args.quick)

    print("\n" + "="*60)
    print("  ✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("  📊 Results saved to ./results/")
    print("  🖥️  Launch dashboard: streamlit run app.py")
    print("="*60 + "\n")
