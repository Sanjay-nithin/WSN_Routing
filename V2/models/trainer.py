import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

class ModelBenchmarkEvaluator:
    @staticmethod
    def evaluate_predictions(y_true, y_pred, y_proba, latency_us=0.0, param_count=0):
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.5
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
        return {
            "accuracy": float(acc), "balanced_accuracy": float(bal_acc), "macro_f1": float(macro_f1),
            "macro_precision": float(macro_p), "macro_recall": float(macro_r), "roc_auc": float(auc),
            "healthy_f1": float(f1[0]), "congested_f1": float(f1[1]), "unhealthy_f1": float(f1[2]),
            "healthy_recall": float(r[0]), "congested_recall": float(r[1]), "unhealthy_recall": float(r[2]),
            "confusion_matrix": cm, "inference_latency_us": float(latency_us), "parameter_count": int(param_count)
        }

    @staticmethod
    def benchmark_inference_latency(predict_fn, sample_input, num_runs=1000):
        for _ in range(20):
            predict_fn(sample_input)
        start = time.perf_counter()
        for _ in range(num_runs):
            predict_fn(sample_input)
        return ((time.perf_counter() - start) / num_runs) * 1e6

def train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64, lr=0.002, device="cpu"):
    model = model.to(device)
    class_counts = np.bincount(y_train, minlength=3)
    weights = len(y_train) / (3.0 * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    return model, {}
