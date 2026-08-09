"""
Module 4: Evaluation & Visualization
======================================
Generates all CNN evaluation plots and saves them to outputs/.

Plots produced:
  1. CNN training curves (loss + accuracy)
  2. CNN confusion matrix (normalized)
  3. CNN per-class precision / recall / F1 bar chart
  4. Node-condition CNN probability heatmap
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

LABEL_NAMES = ["Healthy", "Congested", "Unhealthy"]
PALETTE     = ["#2ecc71", "#e74c3c", "#e67e22"]
PLOT_STYLE  = "dark_background"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _savefig(name: str, fig=None):
    path = os.path.join(SAVE_DIR, name)
    (fig or plt).savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved → {path}")
    return path


# ── 1. CNN training curves ────────────────────────────────────────────────────

def plot_cnn_training(history: dict) -> str:
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history["train_loss"]) + 1)

        axes[0].plot(epochs, history["train_loss"], label="Train", color="#3498db", lw=2)
        axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#e74c3c", lw=2)
        axes[0].set_title("CNN Loss",     fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].legend()

        axes[1].plot(epochs, [a*100 for a in history["train_acc"]], label="Train", color="#3498db", lw=2)
        axes[1].plot(epochs, [a*100 for a in history["val_acc"]],   label="Val",   color="#e74c3c", lw=2)
        axes[1].set_title("CNN Accuracy",  fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
        axes[1].legend()

        fig.suptitle("CNN Node-Condition Classifier – Training History",
                     fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
    return _savefig("cnn_training_curves.png", fig)


# ── 2. CNN confusion matrix ───────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray) -> str:
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
            linewidths=0.5, ax=ax
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True",      fontsize=12)
        ax.set_title("CNN Confusion Matrix (Normalized)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
    return _savefig("cnn_confusion_matrix.png", fig)


# ── 3. CNN per-class F1 bar ───────────────────────────────────────────────────

def plot_cnn_metrics(report: dict) -> str:
    metrics = ["precision", "recall", "f1-score"]
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(LABEL_NAMES))
        w = 0.25
        for i, m in enumerate(metrics):
            vals = [report.get(lbl, {}).get(m, 0) for lbl in LABEL_NAMES]
            bars = ax.bar(x + i*w, vals, width=w, label=m.capitalize(),
                          color=PALETTE[i], edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9, color="white")
        ax.set_xticks(x + w)
        ax.set_xticklabels(LABEL_NAMES)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score"); ax.set_title("CNN Per-Class Metrics", fontweight="bold")
        ax.legend()
        fig.tight_layout()
    return _savefig("cnn_per_class_metrics.png", fig)


# ── 4. Node condition heatmap ─────────────────────────────────────────────────

def plot_node_condition_heatmap(node_cnn_probs: dict) -> str:
    node_ids = sorted(node_cnn_probs.keys())[:50]  # first 50 nodes
    probs = np.array([node_cnn_probs[n] for n in node_ids])

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(probs.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(LABEL_NAMES)
        ax.set_xlabel("Node ID", fontsize=12)
        ax.set_title("CNN Node-Condition Probability Heatmap\n(first 50 nodes)",
                     fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Probability")
        ax.set_xticks(range(len(node_ids)))
        ax.set_xticklabels([str(n) for n in node_ids], rotation=90, fontsize=7)
        fig.tight_layout()
    return _savefig("node_condition_heatmap.png", fig)


# ── 5. Save CNN metrics JSON ──────────────────────────────────────────────────

def save_metrics_table(cnn_results: dict, path: str = None) -> str:
    if path is None:
        path = os.path.join(SAVE_DIR, "cnn_evaluation_results.json")

    summary = {
        "cnn_classifier": {
            "test_accuracy": round(cnn_results["test_acc"], 4),
            "per_class_f1": {
                lbl: round(cnn_results["report"].get(lbl, {}).get("f1-score", 0), 4)
                for lbl in LABEL_NAMES
            },
            "macro_avg_f1": round(
                cnn_results["report"].get("macro avg", {}).get("f1-score", 0), 4
            ),
            "weighted_avg_f1": round(
                cnn_results["report"].get("weighted avg", {}).get("f1-score", 0), 4
            ),
        }
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Metrics saved → {path}")
    return path
