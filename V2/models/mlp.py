import torch
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
