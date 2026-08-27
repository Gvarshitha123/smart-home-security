"""
risk/threat_progression.py — STAGE 8 (integration wrapper)

Purpose
-------
Thin wrapper around models/gru_model.py that maintains a rolling per-device
risk-score history and produces a STABLE / INCREASING / CRITICAL label for
the dashboard and decision engine, WITHOUT needing to reload/retrain the
GRU on every call.

If no trained GRU checkpoint exists yet, falls back to a transparent
rule-based trend estimate (simple slope over the recent window) and labels
the result PROPOSED rather than MEASURED.

Required packages
------------------
pip install torch numpy

How to run
----------
from risk.threat_progression import ThreatProgressionTracker
tracker = ThreatProgressionTracker()
tracker.update("CCTV Camera", 0.42)
trend = tracker.get_trend("CCTV Camera")
"""

import os
from collections import defaultdict, deque

import torch

import config
from models.gru_model import GRUProgressionModel, predict_progression


class ThreatProgressionTracker:
    def __init__(self, window: int = config.GRU_SEQUENCE_LENGTH):
        self.window = window
        self.history = defaultdict(lambda: deque(maxlen=window))
        self.model = self._load_model()

    def _load_model(self):
        path = os.path.join(config.MODEL_DIR, "gru_progression.pt")
        if not os.path.exists(path):
            return None
        model = GRUProgressionModel()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    def update(self, device: str, risk_score: float):
        self.history[device].append(risk_score)

    def get_trend(self, device: str) -> dict:
        hist = list(self.history[device])
        if self.model is not None and len(hist) >= self.window:
            trend = predict_progression(self.model, hist)
            status = config.ClaimStatus.MEASURED
        else:
            trend = self._rule_based_trend(hist)
            status = config.ClaimStatus.PROPOSED
        return {"device": device, "trend": trend, "status": status}

    @staticmethod
    def _rule_based_trend(hist: list) -> str:
        """Transparent fallback: compare the average of the second half of
        the window to the first half. Used only when no trained GRU exists
        yet, or there isn't enough history — labeled PROPOSED, not MEASURED."""
        if len(hist) < 2:
            return "STABLE"
        mid = len(hist) // 2
        first_half_avg = sum(hist[:mid]) / max(mid, 1)
        second_half_avg = sum(hist[mid:]) / max(len(hist) - mid, 1)
        delta = second_half_avg - first_half_avg
        if delta > 0.15:
            return "CRITICAL"
        elif delta > 0.03:
            return "INCREASING"
        return "STABLE"


if __name__ == "__main__":
    tracker = ThreatProgressionTracker()
    for score in [0.1, 0.2, 0.25, 0.4, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9]:
        tracker.update("CCTV Camera", score)
    print(tracker.get_trend("CCTV Camera"))
