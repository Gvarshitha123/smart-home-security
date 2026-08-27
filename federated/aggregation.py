"""
federated/aggregation.py — STAGES 9, 10, 11, 12, 16

Purpose
-------
Orchestrates the full hierarchical FedProx simulation:

  Global Model
    -> sent to 9 simulated home clients
    -> each client runs local FedProx training (client.py)
    -> 3 Edge Servers aggregate their 3 homes each (edge_server.py, FedAvg)
    -> Global Aggregation combines the 3 regional models (FedAvg again)
    -> New Global Model is evaluated on the held-out test set
  Repeat for config.FED_ROUNDS rounds.

Also optionally trains a CENTRALIZED model on the same architecture/data
for a fair Centralized vs Federated comparison (Section 16) — using
IDENTICAL total data and epochs-equivalent budget so the comparison is
meaningful, not just "more training = better."

Required packages
------------------
pip install torch numpy pandas scikit-learn

How to run
----------
python -m federated.aggregation

Expected output
---------------
- artifacts/models/global_model.pt
- artifacts/metrics/fl_round_history.json (per-round global accuracy/F1 —
  used by evaluation/plots.py for the convergence plot)
- artifacts/metrics/centralized_vs_federated.json

Common errors
-------------
- FileNotFoundError from data_utils: run preprocessing first.
- Very slow on a laptop: lower config.FED_ROUNDS / config.LOCAL_EPOCHS, or
  keep config.FAST_MODE = True (already reduces both by default).
"""

import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

import config
from federated.client import SimulatedHomeClient
from federated.edge_server import EdgeServer, fedavg_aggregate
from federated.model_arch import ThreatDetectorMLP
from models.data_utils import load_splits


def partition_non_iid(X, y, num_clients: int = config.NUM_CLIENTS):
    """Simple, transparent non-IID partitioning: sort by label, then split
    into shard groups so each client sees a skewed class mix rather than a
    uniform random sample — a common, explainable non-IID simulation
    approach for a BTech-level FL demo (Section 13)."""
    rng = np.random.RandomState(config.RANDOM_STATE)
    order = np.argsort(y, kind="stable")
    X_sorted, y_sorted = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)

    shards_per_client = 2
    num_shards = num_clients * shards_per_client
    shard_size = len(X_sorted) // num_shards
    shard_indices = list(range(num_shards))
    rng.shuffle(shard_indices)

    client_data = {i: [] for i in range(num_clients)}
    for i, shard_id in enumerate(shard_indices):
        client_id = i % num_clients
        start = shard_id * shard_size
        end = start + shard_size if shard_id < num_shards - 1 else len(X_sorted)
        client_data[client_id].append((start, end))

    partitions = []
    for client_id in range(num_clients):
        idxs = np.concatenate([np.arange(s, e) for s, e in client_data[client_id]]) \
            if client_data[client_id] else np.array([], dtype=int)
        partitions.append((X_sorted.iloc[idxs].to_numpy(), y_sorted.iloc[idxs].to_numpy()))
    return partitions


def build_clients(X_train, y_train, input_dim, num_classes):
    partitions = partition_non_iid(X_train, y_train)
    home_ids = [f"home_{i+1}" for i in range(config.NUM_CLIENTS)]
    clients = {}
    for home_id, (Xp, yp) in zip(home_ids, partitions):
        if len(Xp) == 0:
            continue
        clients[home_id] = SimulatedHomeClient(home_id, Xp, yp, input_dim, num_classes)
    return clients


def evaluate_global_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_t)
        preds = logits.argmax(dim=1).numpy()
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "f1_weighted": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
    }


def run_hierarchical_fedprox():
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, label_col = load_splits()
    input_dim = len(feature_cols)
    num_classes = int(y_train.nunique())

    clients = build_clients(X_train, y_train, input_dim, num_classes)
    edge_servers = {
        edge_id: EdgeServer(edge_id, [c for c in homes if c in clients])
        for edge_id, homes in config.EDGE_SERVER_MAP.items()
    }

    global_model = ThreatDetectorMLP(input_dim, num_classes)
    global_state = copy.deepcopy(global_model.state_dict())

    round_history = []
    comm_bytes_per_round = sum(p.numel() for p in global_model.parameters()) * 4  # float32

    for round_idx in range(1, config.FED_ROUNDS + 1):
        start = time.perf_counter()

        client_updates = {}
        client_sample_counts = {}
        for home_id, client in clients.items():
            client_updates[home_id] = client.local_train(global_state)
            client_sample_counts[home_id] = client.num_samples()

        edge_states = {}
        for edge_id, edge in edge_servers.items():
            if not edge.client_ids:
                continue
            edge_states[edge_id] = edge.aggregate(client_updates, client_sample_counts)

        edge_sample_counts = [
            sum(client_sample_counts[c] for c in edge.client_ids)
            for edge in edge_servers.values() if edge.client_ids
        ]
        global_state = fedavg_aggregate(list(edge_states.values()), edge_sample_counts)
        global_model.load_state_dict(global_state)

        round_time = time.perf_counter() - start
        eval_metrics = evaluate_global_model(global_model, X_val.to_numpy(), y_val.to_numpy())

        round_record = {
            "round": round_idx,
            "global_accuracy": eval_metrics["accuracy"],
            "global_f1_weighted": eval_metrics["f1_weighted"],
            "round_time_sec": round(round_time, 4),
            "approx_communication_bytes_this_round": comm_bytes_per_round * (len(clients) + len(edge_states)),
            "status": config.ClaimStatus.MEASURED,
        }
        round_history.append(round_record)
        print(f"[aggregation] Round {round_idx}/{config.FED_ROUNDS}: {round_record}")

    final_test_metrics = evaluate_global_model(global_model, X_test.to_numpy(), y_test.to_numpy())
    final_test_metrics["status"] = config.ClaimStatus.MEASURED
    final_test_metrics["model"] = "Hierarchical_FedProx_Global"

    torch.save(global_model.state_dict(), os.path.join(config.MODEL_DIR, "global_model.pt"))
    with open(os.path.join(config.METRICS_DIR, "fl_round_history.json"), "w") as f:
        json.dump(round_history, f, indent=2)
    with open(os.path.join(config.METRICS_DIR, "federated_test_metrics.json"), "w") as f:
        json.dump(final_test_metrics, f, indent=2)

    print(f"[aggregation] Final federated test metrics: {final_test_metrics}")
    return global_model, round_history, final_test_metrics


def run_centralized_baseline():
    """Trains the SAME architecture on ALL training data pooled centrally
    (no FL), for a fair Section 16 comparison."""
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, label_col = load_splits()
    input_dim = len(feature_cols)
    num_classes = int(y_train.nunique())

    model = ThreatDetectorMLP(input_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_t = torch.tensor(y_train.to_numpy(), dtype=torch.long)

    total_local_epochs = config.FED_ROUNDS * config.LOCAL_EPOCHS  # equal compute budget
    start = time.perf_counter()
    model.train()
    for _ in range(total_local_epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
    training_time = time.perf_counter() - start

    metrics = evaluate_global_model(model, X_test.to_numpy(), y_test.to_numpy())
    metrics["training_time_sec"] = round(training_time, 4)
    metrics["status"] = config.ClaimStatus.MEASURED
    metrics["model"] = "Centralized_MLP"

    with open(os.path.join(config.METRICS_DIR, "centralized_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[aggregation] Centralized baseline metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    fed_model, history, fed_metrics = run_hierarchical_fedprox()
    central_metrics = run_centralized_baseline()

    comparison = {"federated": fed_metrics, "centralized": central_metrics}
    with open(os.path.join(config.METRICS_DIR, "centralized_vs_federated.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[aggregation] Centralized vs Federated: {json.dumps(comparison, indent=2)}")
