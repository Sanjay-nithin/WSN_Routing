"""
Module 2: CNN Model for Node Condition Classification
=====================================================
Architecture: 1-D CNN over a fixed-length feature vector.
  Input  : (batch, 1, n_features)  – treat features as a 1-D signal
  Conv1  : 32 filters, kernel=3, ReLU + BN
  Conv2  : 64 filters, kernel=3, ReLU + BN
  Conv3  : 128 filters, kernel=3, ReLU + BN
  GlobalAvgPool → FC(128→64) → FC(64→3)
  Output : 3-class softmax logits (Healthy / Congested / Unhealthy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    """1-D CNN for WSN node-condition classification."""

    def __init__(self, n_features: int, n_classes: int = 3):
        super().__init__()
        self.n_features = n_features

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) → (B, 1, n_features)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # Global average pool over feature dimension
        x = x.mean(dim=2)          # (B, 128)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))    # (B, 64)
        x = self.fc2(x)            # (B, n_classes)
        return x

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)
