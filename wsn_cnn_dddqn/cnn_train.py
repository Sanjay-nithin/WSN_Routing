"""
Module 3: CNN Training
======================
Trains the CNNClassifier on the merged dataset, evaluates on a held-out
test split, and saves the trained model + scaler.
"""

import os, time, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

from wsn_cnn_dddqn.cnn_model import CNNClassifier

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Healthy", "Congested", "Unhealthy"]


def train_cnn(
    X: np.ndarray,
    y: np.ndarray,
    scaler,
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    test_ratio: float = 0.2,
) -> dict:
    """
    Train CNN and return a results dict with model, history, and metrics.
    """
    n_features = X.shape[1]
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    dataset = TensorDataset(X_t, y_t)
    n_test  = int(len(dataset) * test_ratio)
    n_train = len(dataset) - n_test
    train_ds, test_ds = random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = CNNClassifier(n_features=n_features, n_classes=3).to(DEVICE)

    # Compute class weights to handle imbalance
    class_counts = np.bincount(y)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * 3
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n[CNN] Training on {n_train} samples | Validating on {n_test} | Device: {DEVICE}")
    print(f"      Features: {n_features}  |  Epochs: {n_epochs}  |  Batch: {batch_size}")
    print("-" * 65)

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            ep_loss    += loss.item() * len(xb)
            ep_correct += (logits.argmax(1) == yb).sum().item()
            ep_total   += len(xb)
        scheduler.step()
        tr_loss = ep_loss / ep_total
        tr_acc  = ep_correct / ep_total

        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                vl_loss    += criterion(logits, yb).item() * len(xb)
                vl_correct += (logits.argmax(1) == yb).sum().item()
                vl_total   += len(xb)
        vl_loss /= vl_total
        vl_acc   = vl_correct / vl_total

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:03d}/{n_epochs} | "
                  f"loss {tr_loss:.4f} → {vl_loss:.4f} | "
                  f"acc {tr_acc:.3f} → {vl_acc:.3f}")

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")

    # ── Final evaluation ─────────────────────────────────────────────────────
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    acc = accuracy_score(all_true, all_preds)
    cm  = confusion_matrix(all_true, all_preds)
    report = classification_report(
        all_true, all_preds, target_names=LABEL_NAMES, output_dict=True
    )

    print(f"\n  Test Accuracy : {acc:.4f}")
    print(classification_report(all_true, all_preds, target_names=LABEL_NAMES))

    # ── Save artefacts ───────────────────────────────────────────────────────
    model_path  = os.path.join(SAVE_DIR, "cnn_model.pt")
    scaler_path = os.path.join(SAVE_DIR, "feature_scaler.pkl")
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Model saved  → {model_path}")
    print(f"  Scaler saved → {scaler_path}")

    return {
        "model":      model,
        "history":    history,
        "test_acc":   acc,
        "cm":         cm,
        "report":     report,
        "test_true":  all_true,
        "test_preds": all_preds,
        "n_features": n_features,
        "scaler":     scaler,
    }
