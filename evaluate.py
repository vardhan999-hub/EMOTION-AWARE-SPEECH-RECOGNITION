# evaluate.py
"""
Evaluation script — generates classification report and confusion matrix.

Usage:
    python evaluate.py
    python evaluate.py --manifest dataset_manifest.csv --model checkpoints/best_model.pth
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from dataset import SERDataset, pad_collate
from ser_webapp.models import HybridSER
from ser_webapp.config import DEVICE, EMOTIONS, N_MELS, N_MFCC


def load_model(path="checkpoints/best_model.pth"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}. Run train.py first.")
    chk   = torch.load(path, map_location=DEVICE)
    model = HybridSER(n_mels=N_MELS, n_mfcc=N_MFCC, num_classes=len(EMOTIONS)).to(DEVICE)
    model.load_state_dict(chk["model_state"])
    model.eval()
    print(f"[evaluate] Model loaded from {path}")
    return model


def evaluate(manifest="dataset_manifest.csv", model_path="checkpoints/best_model.pth"):
    model  = load_model(model_path)
    ds     = SERDataset(manifest)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=pad_collate, num_workers=0)

    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            mel    = batch["mel"].to(DEVICE)
            mfcc   = batch["mfcc"].to(DEVICE)
            labels = batch["labels"].numpy()
            logits = model(mel, mfcc)
            pred   = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(labels.tolist())

    # ── Classification Report ──────────────────────────────────
    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(trues, preds, target_names=EMOTIONS, zero_division=0))

    # ── Confusion Matrix ──────────────────────────────────────
    cm = confusion_matrix(trues, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=axes[1])
    axes[1].set_title('Confusion Matrix (%)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

    plt.suptitle('HybridSER — Emotion Classification Results', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()
    print("[evaluate] Saved confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="dataset_manifest.csv")
    parser.add_argument("--model",    default="checkpoints/best_model.pth")
    args = parser.parse_args()
    evaluate(manifest=args.manifest, model_path=args.model)
