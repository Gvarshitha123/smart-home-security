"""
federated/model_arch.py

A small, CPU-friendly MLP used as the model trained via hierarchical
FedProx across the 9 simulated homes. Kept intentionally simple (2 hidden
layers) so training many local rounds on a student laptop stays fast.
"""

import torch.nn as nn


class ThreatDetectorMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
