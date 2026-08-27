"""
federated/fedprox.py — STAGE 15

Purpose
-------
Implements the FedProx proximal term: during local training, each client
is penalized for drifting too far from the current global model. This
helps with the non-IID, heterogeneous data expected across simulated
smart-home clients (Section 12).

    local_loss = task_loss + (mu / 2) * ||local_weights - global_weights||^2

Required packages
------------------
pip install torch

How to run
----------
Used internally by federated/client.py — not run standalone.
"""

import torch


def fedprox_proximal_term(local_model: torch.nn.Module, global_state_dict: dict, mu: float) -> torch.Tensor:
    """Computes (mu/2) * sum of squared differences between the local
    model's current parameters and the global model's parameters."""
    prox_term = 0.0
    local_state = local_model.state_dict()
    for name, param in local_model.named_parameters():
        global_param = global_state_dict[name]
        prox_term = prox_term + torch.sum((param - global_param) ** 2)
    return (mu / 2.0) * prox_term
