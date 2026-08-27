"""
models/gru_model.py — STAGE 8 (threat progression prediction)

Purpose
-------
Predict the likely PROGRESSION of an observed risk pattern over a short
sequence of recent observations — NOT the exact next attack.

Input:  a sequence of the last N risk scores (0.0-1.0) for a device/home.
Output: one of STABLE / INCREASING / CRITICAL.

This is intentionally simple (GRU over a single scalar risk-score channel)
so it is CPU-friendly and explainable in a viva. It can be extended to a
richer feature sequence later if compute allows.

Required packages
------------------
pip install torch numpy scikit-learn joblib

How to run
----------
python -m models.gru_model

Expected output
---------------
- artifacts/models/gru_progression.pt (state_dict)
- artifacts/metrics/gru_metrics.json (measured accuracy on the held-out
  sequences — only if training data was available; otherwise this script
  prints a note and does not fabricate a result)

Common errors
-------------
- "torch not installed": pip install torch (CPU build is fine).
- Not enough data to form sequences: increase input data or reduce
  config.GRU_SEQUENCE_LENGTH.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config

LABELS = ["STABLE", "INCREASING", "CRITICAL"]


class GRUProgressionModel(nn.Module):
    def __init__(self, hidden_size=config.GRU_HIDDEN_SIZE, num_layers=config.GRU_NUM_LAYERS,
                 num_classes=len(LABELS)):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


def make_sequences_from_risk_scores(risk_scores: list, seq_len: int = config.GRU_SEQUENCE_LENGTH):
    """Turn a flat list of historical risk scores into (X, y) sequence
    samples. Label for each window is derived from the trend of the risk
    score itself (simple slope heuristic) — this is a PROPOSED labeling
    rule for bootstrapping training, documented as such, not a measured
    ground truth."""
    X, y = [], []
    for i in range(len(risk_scores) - seq_len):
        window = risk_scores[i:i + seq_len]
        next_val = risk_scores[i + seq_len]
        slope = next_val - window[-1]
        if slope > 0.15:
            label = 2  # CRITICAL
        elif slope > 0.03:
            label = 1  # INCREASING
        else:
            label = 0  # STABLE
        X.append(window)
        y.append(label)
    return np.array(X, dtype="float32"), np.array(y, dtype="int64")


def train_model(risk_score_history: list):
    if len(risk_score_history) <= config.GRU_SEQUENCE_LENGTH + 5:
        print("[gru_model] Not enough risk-score history to train yet "
              "(need more than sequence_length + 5 points). "
              "Skipping training — no fabricated metrics will be produced.")
        return None, None

    X, y = make_sequences_from_risk_scores(risk_score_history)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.tensor(X_train).unsqueeze(-1)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test).unsqueeze(-1)
    y_test_t = torch.tensor(y_test)

    model = GRUProgressionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)

    model.train()
    for epoch in range(config.GRU_EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[gru_model] Epoch {epoch + 1}/{config.GRU_EPOCHS} — loss: {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1)
        accuracy = (preds == y_test_t).float().mean().item() if len(y_test_t) else None

    metrics = {
        "model": "GRU_ThreatProgression",
        "status": config.ClaimStatus.MEASURED if accuracy is not None else config.ClaimStatus.PROPOSED,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "n_train_sequences": len(X_train),
        "n_test_sequences": len(X_test),
        "labeling_rule": "PROPOSED slope-based heuristic for bootstrap training "
                          "(next_risk - last_risk thresholds), not externally validated ground truth.",
        "note": "Predicts the likely progression of the observed risk pattern, "
                "not an exact future attack.",
    }

    model_path = os.path.join(config.MODEL_DIR, "gru_progression.pt")
    torch.save(model.state_dict(), model_path)
    with open(os.path.join(config.METRICS_DIR, "gru_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[gru_model] Saved model to {model_path}")
    print(f"[gru_model] Metrics: {json.dumps(metrics, indent=2)}")
    return model, metrics


def predict_progression(model: GRUProgressionModel, recent_risk_scores: list) -> str:
    """recent_risk_scores must have length == config.GRU_SEQUENCE_LENGTH."""
    if len(recent_risk_scores) < config.GRU_SEQUENCE_LENGTH:
        # Not enough history yet — degrade gracefully instead of guessing.
        return "STABLE"
    window = recent_risk_scores[-config.GRU_SEQUENCE_LENGTH:]
    x = torch.tensor([window], dtype=torch.float32).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        idx = logits.argmax(dim=1).item()
    return LABELS[idx]


if __name__ == "__main__":
    # Demo/bootstrap run using a synthetic risk-score walk, clearly labeled.
    # Replace with real historical risk scores produced by risk_assessment.py
    # once the pipeline has been run on real detector outputs.
    rng = np.random.RandomState(config.RANDOM_STATE)
    synthetic_walk = np.clip(np.cumsum(rng.normal(0, 0.05, 200)) * 0.5 + 0.3, 0, 1).tolist()
    print("[gru_model] Training on SYNTHETIC demo risk-score walk "
          "(replace with real risk history for actual results).")
    train_model(synthetic_walk)
