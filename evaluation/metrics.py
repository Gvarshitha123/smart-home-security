"""
evaluation/metrics.py — Stage 9 (required threat-detection metrics)

Purpose
-------
One shared, honest metrics implementation used by every model script, so
Random Forest / XGBoost / LightGBM are all measured the exact same way and
are directly comparable. Also provides the model-comparison helper used to
pick the best detector by actual results (Section 32/40 — never assume
LightGBM wins).

Required packages
------------------
pip install scikit-learn pandas numpy

How to run
----------
Import compute_classification_metrics / save_confusion_matrix from model
scripts. To compare already-trained models, run:
    python -m evaluation.metrics --compare

Expected output
---------------
compute_classification_metrics -> dict of accuracy/precision/recall/f1/FPR
save_confusion_matrix -> JSON file in evaluation/confusion_matrices/
--compare -> prints a ranked table read from artifacts/metrics/*_metrics.json
"""

import glob
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config


def compute_classification_metrics(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    # False Positive Rate, computed per-class then averaged (multiclass-safe).
    fpr_per_class = []
    for i in range(cm.shape[0]):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_per_class.append(fpr)
    avg_fpr = float(np.mean(fpr_per_class))

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "false_positive_rate_avg": round(avg_fpr, 4),
    }


def save_confusion_matrix(y_true, y_pred, name: str):
    cm = confusion_matrix(y_true, y_pred)
    out_path = os.path.join(config.EVAL_CONFUSION_DIR, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump({"confusion_matrix": cm.tolist()}, f, indent=2)
    return out_path


def compare_all_models() -> list:
    """Reads every *_metrics.json in artifacts/metrics/ and ranks by F1.
    Only compares models that have ACTUALLY been trained and measured."""
    rows = []
    for path in glob.glob(os.path.join(config.METRICS_DIR, "*_metrics.json")):
        with open(path) as f:
            data = json.load(f)
        if "model" in data and "f1_weighted" in data:
            rows.append(data)

    rows.sort(key=lambda r: r["f1_weighted"], reverse=True)
    return rows


def _print_comparison():
    rows = compare_all_models()
    if not rows:
        print("No measured model metrics found yet. Train at least one model first "
              "(models/random_forest.py, models/xgboost_model.py, models/lightgbm_model.py).")
        return
    print(f"{'Model':<15}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}{'F1':<8}{'FPR':<8}")
    for r in rows:
        print(f"{r['model']:<15}{r['accuracy']:<10}{r['precision_weighted']:<11}"
              f"{r['recall_weighted']:<9}{r['f1_weighted']:<8}{r['false_positive_rate_avg']:<8}")
    best = rows[0]
    print(f"\nBest detector by MEASURED F1-score: {best['model']} (F1={best['f1_weighted']})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        _print_comparison()
