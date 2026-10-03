import torch
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
