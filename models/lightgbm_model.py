"""
models/lightgbm_model.py — STAGE 6

Purpose
-------
Train and evaluate a LightGBM classifier — the third threat-detection
candidate. LightGBM is NOT assumed to be the best model; the best detector
is chosen later in evaluation/metrics.py by comparing all three MEASURED
results side by side.

Required packages
------------------
pip install lightgbm scikit-learn joblib pandas

How to run
----------
python -m models.lightgbm_model

Expected output
---------------
- artifacts/models/lightgbm.joblib
- artifacts/metrics/lightgbm_metrics.json
- evaluation/confusion_matrices/lightgbm_cm.json

Common errors
-------------
- "lightgbm not installed": pip install lightgbm
- libgomp / OpenMP errors on some Linux setups: install libgomp1
  (e.g. `sudo apt-get install libgomp1`) or use conda instead of pip.
"""

import json
import os
import time

import joblib
from lightgbm import LGBMClassifier

import config
from evaluation.metrics import compute_classification_metrics, save_confusion_matrix
from models.data_utils import load_splits


def train_and_evaluate():
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, label_col = load_splits()

    n_classes = y_train.nunique()
    objective = "binary" if n_classes == 2 else "multiclass"

    model = LGBMClassifier(
        n_estimators=100 if config.FAST_MODE else 300,
        max_depth=-1,
        learning_rate=0.1,
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
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
    metrics["model"] = "LightGBM"
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)

    model_path = os.path.join(config.MODEL_DIR, "lightgbm.joblib")
    joblib.dump(model, model_path)

    metrics_path = os.path.join(config.METRICS_DIR, "lightgbm_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(y_test, y_pred, "lightgbm_cm")

    print(f"[lightgbm] Saved model to {model_path}")
    print(f"[lightgbm] Metrics: {json.dumps(metrics, indent=2)}")
    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
