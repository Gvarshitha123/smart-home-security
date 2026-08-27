"""
federated/edge_server.py — STAGE 10 (three edge servers, regional aggregation)

Purpose
-------
Each Edge Server aggregates the local model updates from its 3 assigned
simulated homes (FedAvg, weighted by number of local samples), producing a
regional model. Three regional models are then combined by global
aggregation (see aggregation.py).

Required packages
------------------
pip install torch

How to run
----------
Used by federated/aggregation.py — not run standalone.
"""

import copy

import torch


def fedavg_aggregate(state_dicts: list, sample_counts: list) -> dict:
    """Weighted average of a list of model state_dicts, weighted by each
    client's local sample count (standard FedAvg weighting)."""
    total_samples = sum(sample_counts)
    avg_state = copy.deepcopy(state_dicts[0])

    for key in avg_state.keys():
        avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)

    for state, n in zip(state_dicts, sample_counts):
        weight = n / total_samples
        for key in avg_state.keys():
            avg_state[key] += state[key].float() * weight

    return avg_state


class EdgeServer:
    def __init__(self, edge_id: str, client_ids: list):
        self.edge_id = edge_id
        self.client_ids = client_ids

    def aggregate(self, client_updates: dict, client_sample_counts: dict) -> dict:
        """client_updates: {home_id: state_dict}. Returns the regional
        (edge-aggregated) state_dict for this edge server's assigned homes."""
        states = [client_updates[c] for c in self.client_ids]
        counts = [client_sample_counts[c] for c in self.client_ids]
        return fedavg_aggregate(states, counts)
