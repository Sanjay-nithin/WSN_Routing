import torch
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
