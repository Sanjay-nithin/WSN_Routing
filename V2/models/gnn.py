import torch
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
