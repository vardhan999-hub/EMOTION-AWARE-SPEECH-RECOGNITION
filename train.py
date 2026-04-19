# train.py
"""
Training script for the Hybrid CNN-BiLSTM Speech Emotion Recognition model.

Usage:
    python train.py
    python train.py --manifest dataset_manifest.csv --epochs 50 --out checkpoints
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from dataset import SERDataset, pad_collate
from ser_webapp.models import HybridSER
from ser_webapp.config import DEVICE, BATCH_SIZE, NUM_EPOCHS, LR, EMOTIONS, N_MELS, N_MFCC
from ser_webapp.preprocess import scan_and_build_csv


# ── Training / Evaluation helpers ───────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses, preds, trues = [], [], []

    for batch in loader:
        mel    = batch['mel'].to(DEVICE)
        mfcc   = batch['mfcc'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(mel, mfcc)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        trues.extend(labels.cpu().tolist())

    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average='macro', zero_division=0)
    return sum(losses) / len(losses), acc, f1


def eval_model(model, loader, criterion):
    model.eval()
    losses, preds, trues = [], [], []

    with torch.no_grad():
        for batch in loader:
            mel    = batch['mel'].to(DEVICE)
            mfcc   = batch['mfcc'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits = model(mel, mfcc)
            loss   = criterion(logits, labels)

            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            trues.extend(labels.cpu().tolist())

    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average='macro', zero_division=0)
    return sum(losses) / len(losses), acc, f1, trues, preds


# ── Plot helpers ─────────────────────────────────────────────────

def save_plots(train_accs, val_accs, train_losses, val_losses):
    epochs = range(1, len(train_accs) + 1)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, marker='o', linewidth=2, label='Train Accuracy')
    plt.plot(epochs, val_accs,   marker='o', linewidth=2, label='Val Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=300)
    plt.close()
    print("[train] Saved accuracy_curve.png")

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', linewidth=2, label='Train Loss')
    plt.plot(epochs, val_losses,   marker='o', linewidth=2, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()
    print("[train] Saved loss_curve.png")


# ── Main ─────────────────────────────────────────────────────────

def main(manifest="dataset_manifest.csv", out_dir="checkpoints", epochs=NUM_EPOCHS):
    os.makedirs(out_dir, exist_ok=True)

    # Build CSV if missing
    if not os.path.exists(manifest):
        print(f"[train] Manifest not found — scanning datasets...")
        scan_and_build_csv(manifest)

    # Dataset + split
    dataset = SERDataset(manifest)
    n       = len(dataset)
    train_n = int(0.8 * n)
    val_n   = n - train_n
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_n, val_n], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=pad_collate, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=pad_collate, num_workers=0, pin_memory=False)

    # Class weights for imbalanced dataset
    all_labels   = [EMOTIONS.index(e) for e in dataset.df['emotion']]
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"[train] Class weights: {dict(zip(EMOTIONS, class_weights.round(3)))}")

    # Model, loss, optimizer, scheduler
    model     = HybridSER(n_mels=N_MELS, n_mfcc=N_MFCC, num_classes=len(EMOTIONS)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5,
                                                      factor=0.5)

    print(f"\n[train] Device: {DEVICE}  |  Epochs: {epochs}  |  Batch: {BATCH_SIZE}")
    print(f"[train] Train: {train_n}  Val: {val_n}\n")

    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    no_improve = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, vl_f1, vl_trues, vl_preds = eval_model(model, val_loader, criterion)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        scheduler.step(vl_acc)

        print(f"Epoch {epoch:3d}/{epochs} │ "
              f"TrLoss={tr_loss:.4f}  TrAcc={tr_acc:.4f}  TrF1={tr_f1:.4f} │ "
              f"VlLoss={vl_loss:.4f}  VlAcc={vl_acc:.4f}  VlF1={vl_f1:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save(
                {"model_state": model.state_dict(), "config": {"emotions": EMOTIONS}},
                os.path.join(out_dir, "best_model.pth")
            )
            print(f"  ✓ Saved best model (val_acc={vl_acc:.4f})")
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= 10:
            print(f"[train] Early stopping at epoch {epoch}")
            break

    print(f"\n[train] Done. Best val accuracy: {best_val_acc:.4f}")

    # Print final validation report
    _, _, _, vl_trues, vl_preds = eval_model(model, val_loader, criterion)
    print("\n[train] Final Validation Report:")
    print(classification_report(vl_trues, vl_preds, target_names=EMOTIONS, zero_division=0))

    # Save plots
    save_plots(train_accs, val_accs, train_losses, val_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="dataset_manifest.csv")
    parser.add_argument("--out",      default="checkpoints")
    parser.add_argument("--epochs",   type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    main(manifest=args.manifest, out_dir=args.out, epochs=args.epochs)
