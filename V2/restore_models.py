models_files = {}

models_files["V2/models/classical.py"] = '''from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ClassicalMLSuite:
    @staticmethod
    def get_logistic_regression(random_state: int = 42):
        return LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=random_state)

    @staticmethod
    def get_decision_tree(max_depth: int = 6, random_state: int = 42):
        return DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", criterion="gini", random_state=random_state)

    @staticmethod
    def get_random_forest(n_estimators: int = 60, max_depth: int = 8, random_state: int = 42):
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced", n_jobs=-1, random_state=random_state)

    @staticmethod
    def get_svm(C: float = 1.0, gamma: str = "scale", random_state: int = 42):
        return SVC(C=C, kernel="rbf", gamma=gamma, probability=True, class_weight="balanced", random_state=random_state)
'''

models_files["V2/models/boosting.py"] = '''import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

class XGBoostClassifierWrapper:
    def __init__(self, n_estimators: int = 80, max_depth: int = 5, learning_rate: float = 0.08, subsample: float = 0.85, random_state: int = 42):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=subsample, objective="multi:softprob", num_class=3, random_state=random_state, n_jobs=-1, eval_metric="mlogloss"
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        sample_weights = compute_sample_weight("balanced", y)
        self.model.fit(X, y, sample_weight=sample_weights)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
'''

models_files["V2/models/mlp.py"] = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WSNMLP(nn.Module):
    def __init__(self, in_features: int = 12, num_classes: int = 3, hidden_dim1: int = 64, hidden_dim2: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.out = nn.Linear(hidden_dim2, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        return self.out(h)

    @torch.no_grad()
    def predict_proba(self, x_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
        logits = self.forward(x_tensor)
        return F.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        return self.predict_proba(x_np, device).argmax(axis=-1)
'''

models_files["V2/models/cnn_1d.py"] = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalCNN1D(nn.Module):
    def __init__(self, in_channels: int = 12, seq_len: int = 5, num_classes: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == self.seq_len and x.size(2) == self.in_channels:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        h = self.conv1(x)
        h = self.conv2(h)
        h = h.mean(dim=-1)
        h = self.dropout(h)
        h = F.relu(self.fc1(h))
        return self.out(h)

    @torch.no_grad()
    def predict_proba(self, x_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
        logits = self.forward(x_tensor)
        return F.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        return self.predict_proba(x_np, device).argmax(axis=-1)
'''

models_files["V2/models/cnn_lstm.py"] = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalCNNLSTM(nn.Module):
    def __init__(self, in_features: int = 12, hidden_dim: int = 48, lstm_layers: int = 1, num_classes: int = 3):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_proj = self.feature_proj(x)
        lstm_out, _ = self.lstm(x_proj)
        last_hidden = lstm_out[:, -1, :]
        h = F.relu(self.fc(last_hidden))
        return self.out(h)

    @torch.no_grad()
    def predict_proba(self, x_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)
        logits = self.forward(x_tensor)
        return F.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        return self.predict_proba(x_np, device).argmax(axis=-1)
'''

models_files["V2/models/gnn.py"] = '''import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(x, self.weight)
        return torch.matmul(norm_adj, support) + self.bias

class WSNGCN(nn.Module):
    def __init__(self, in_features: int = 12, hidden_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.gc1 = GraphConvLayer(in_features, hidden_dim)
        self.gc2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gc1(x, norm_adj))
        h = F.relu(self.gc2(h, norm_adj))
        return self.out(h)

    @torch.no_grad()
    def predict_proba(self, x_np: np.ndarray, norm_adj_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        x_t = torch.tensor(x_np, dtype=torch.float32).to(device)
        adj_t = torch.tensor(norm_adj_np, dtype=torch.float32).to(device)
        logits = self.forward(x_t, adj_t)
        return F.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, norm_adj_np: np.ndarray, device: str = "cpu") -> np.ndarray:
        return self.predict_proba(x_np, norm_adj_np, device).argmax(axis=-1)
'''

models_files["V2/models/trainer.py"] = '''import time
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
'''

models_files["V2/models/__init__.py"] = '''from .classical import ClassicalMLSuite
from .boosting import XGBoostClassifierWrapper
from .mlp import WSNMLP
from .cnn_1d import TemporalCNN1D
from .cnn_lstm import TemporalCNNLSTM
from .gnn import WSNGCN
from .trainer import ModelBenchmarkEvaluator, train_pytorch_model

__all__ = ["ClassicalMLSuite", "XGBoostClassifierWrapper", "WSNMLP", "TemporalCNN1D", "TemporalCNNLSTM", "WSNGCN", "ModelBenchmarkEvaluator", "train_pytorch_model"]
'''

for path, content in models_files.items():
    with open(path, "w") as f:
        f.write(content)
    print(f"Restored: {path}")
