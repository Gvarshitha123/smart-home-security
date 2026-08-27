"""
models/xgboost_model.py — STAGE 5

Purpose
-------
Train and evaluate an XGBoost classifier as a second threat-detection
candidate, compared against the Random Forest baseline.

Required packages
------------------
pip install xgboost scikit-learn joblib pandas

How to run
----------
python -m models.xgboost_model

Expected output
---------------
- artifacts/models/xgboost.joblib
- artifacts/metrics/xgboost_metrics.json
- evaluation/confusion_matrices/xgboost_cm.json

Common errors
-------------
- "xgboost not installed": pip install xgboost
- If the label is multiclass, XGBoost needs labels encoded as 0..N-1
  (already handled by preprocess.py's LabelEncoder).
"""

import json
import os
import time

import joblib
from xgboost import XGBClassifier

import config
from evaluation.metrics import compute_classification_metrics, save_confusion_matrix
from models.data_utils import load_splits


def train_and_evaluate():
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, label_col = load_splits()

    n_classes = y_train.nunique()
    objective = "binary:logistic" if n_classes == 2 else "multi:softmax"

    model = XGBClassifier(
        n_estimators=100 if config.FAST_MODE else 300,
        max_depth=6,
        learning_rate=0.1,
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
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
    metrics["model"] = "XGBoost"
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)

    model_path = os.path.join(config.MODEL_DIR, "xgboost.joblib")
    joblib.dump(model, model_path)

    metrics_path = os.path.join(config.METRICS_DIR, "xgboost_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(y_test, y_pred, "xgboost_cm")

    print(f"[xgboost] Saved model to {model_path}")
    print(f"[xgboost] Metrics: {json.dumps(metrics, indent=2)}")
    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
