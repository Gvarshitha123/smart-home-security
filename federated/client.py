"""
federated/client.py — STAGE 9 (simulated smart-home clients)

Purpose
-------
Represents ONE simulated smart-home client (there is no physical device
behind this — Section 13). Holds a partition of the training data and
performs local FedProx training given the current global model weights.

Required packages
------------------
pip install torch numpy pandas

How to run
----------
Used by federated/aggregation.py — not run standalone, but can be smoke
tested directly: python -m federated.client
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from federated.fedprox import fedprox_proximal_term
from federated.model_arch import ThreatDetectorMLP


class SimulatedHomeClient:
    def __init__(self, home_id: str, X, y, input_dim: int, num_classes: int):
        self.home_id = home_id
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def local_train(self, global_state_dict: dict, mu: float = config.FEDPROX_MU,
                     epochs: int = config.LOCAL_EPOCHS,
                     batch_size: int = config.LOCAL_BATCH_SIZE,
                     lr: float = 1e-3) -> dict:
        """Runs local FedProx training starting from the global weights and
        returns the updated local state_dict (to be aggregated by the edge
        server)."""
        model = ThreatDetectorMLP(self.input_dim, self.num_classes)
        model.load_state_dict(copy.deepcopy(global_state_dict))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(self.X, self.y), batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss = loss + fedprox_proximal_term(model, global_state_dict, mu)
                loss.backward()
                optimizer.step()

        return copy.deepcopy(model.state_dict())

    def num_samples(self) -> int:
        return len(self.y)


if __name__ == "__main__":
    print("federated/client.py is a library module — run federated/aggregation.py "
          "to execute a full hierarchical FedProx simulation.")
