"""
models/random_forest.py — STAGE 4 (baseline threat detector)

Purpose
-------
Train and evaluate a Random Forest classifier as the BASELINE threat
detector. Every metric reported here is MEASURED on the actual processed
data — nothing is invented.

Required packages
------------------
pip install scikit-learn joblib pandas

How to run
----------
python -m models.random_forest

Expected output
---------------
- Trained model saved to artifacts/models/random_forest.joblib
- Metrics (accuracy, precision, recall, F1, FPR, timings) saved to
  artifacts/metrics/random_forest_metrics.json
- Confusion matrix saved to evaluation/confusion_matrices/random_forest_cm.json

Common errors
-------------
- FileNotFoundError from data_utils: run preprocessing first.
"""

import json
import os
import time

import joblib
from sklearn.ensemble import RandomForestClassifier

import config
from evaluation.metrics import compute_classification_metrics, save_confusion_matrix
from models.data_utils import load_splits


def train_and_evaluate():
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, label_col = load_splits()

    model = RandomForestClassifier(
        n_estimators=100 if config.FAST_MODE else 300,
        max_depth=None,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start
    per_sample_latency = inference_time / max(len(X_test), 1)

    metrics = compute_classification_metrics(y_test, y_pred)
    metrics["training_time_sec"] = round(training_time, 4)
    metrics["inference_time_sec_total"] = round(inference_time, 4)
    metrics["detection_latency_sec_per_sample"] = round(per_sample_latency, 8)
    metrics["status"] = config.ClaimStatus.MEASURED
    metrics["model"] = "RandomForest"
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)

    model_path = os.path.join(config.MODEL_DIR, "random_forest.joblib")
    joblib.dump(model, model_path)

    metrics_path = os.path.join(config.METRICS_DIR, "random_forest_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(y_test, y_pred, "random_forest_cm")

    print(f"[random_forest] Saved model to {model_path}")
    print(f"[random_forest] Metrics: {json.dumps(metrics, indent=2)}")
    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
