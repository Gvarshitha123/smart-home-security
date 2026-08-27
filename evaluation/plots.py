"""
evaluation/plots.py

Purpose
-------
Generate the visualizations listed in Section 30 of the project spec, but
ONLY for metrics that have actually been measured and saved to
artifacts/metrics/. Nothing here fabricates numbers — if a metrics file is
missing, that plot is skipped with a printed note.

Required packages
------------------
pip install matplotlib pandas numpy

How to run
----------
python -m evaluation.plots

Expected output
---------------
PNG files saved under evaluation/plots/, e.g.:
  model_comparison.png, confusion_matrix_<model>.png,
  training_time_comparison.png, inference_latency_comparison.png,
  fl_convergence.png (if federated round history exists),
  risk_level_distribution.png, alert_frequency.png (if alert log exists)
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from evaluation.metrics import compare_all_models


def plot_model_comparison():
    rows = compare_all_models()
    if not rows:
        print("[plots] Skipping model_comparison — no measured model metrics yet.")
        return
    models = [r["model"] for r in rows]
    metrics_to_plot = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]

    x = np.arange(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(metrics_to_plot):
        values = [r[m] for r in rows]
        ax.bar(x + i * width, values, width, label=m)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_title("Threat Detector Comparison (measured)")
    ax.legend()
    _savefig(fig, "model_comparison.png")


def plot_confusion_matrices():
    for path in glob.glob(os.path.join(config.EVAL_CONFUSION_DIR, "*_cm.json")):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            cm = np.array(json.load(f)["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name.replace('_cm', '')}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center")
        fig.colorbar(im)
        _savefig(fig, f"confusion_matrix_{name.replace('_cm', '')}.png")


def plot_timing_comparisons():
    rows = compare_all_models()
    if not rows:
        print("[plots] Skipping timing comparisons — no measured model metrics yet.")
        return

    for metric_key, title, filename in [
        ("training_time_sec", "Training Time Comparison (measured, seconds)", "training_time_comparison.png"),
        ("detection_latency_sec_per_sample", "Per-Sample Detection Latency (measured, seconds)", "inference_latency_comparison.png"),
    ]:
        if not all(metric_key in r for r in rows):
            print(f"[plots] Skipping {filename} — {metric_key} missing from one or more metrics files.")
            continue
        models = [r["model"] for r in rows]
        values = [r[metric_key] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(models, values, color="steelblue")
        ax.set_title(title)
        _savefig(fig, filename)


def plot_fl_convergence():
    path = os.path.join(config.METRICS_DIR, "fl_round_history.json")
    if not os.path.exists(path):
        print("[plots] Skipping fl_convergence — run federated training first "
              "(federated/aggregation.py) to produce fl_round_history.json.")
        return
    with open(path) as f:
        history = json.load(f)
    rounds = [h["round"] for h in history]
    acc = [h["global_accuracy"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rounds, acc, marker="o")
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Global Accuracy (measured)")
    ax.set_title("Federated Learning Convergence")
    _savefig(fig, "fl_convergence.png")


def plot_risk_level_distribution():
    path = os.path.join(config.METRICS_DIR, "alert_log.json")
    if not os.path.exists(path):
        print("[plots] Skipping risk_level_distribution — no alert_log.json yet "
              "(run the alert manager / dashboard simulation first).")
        return
    with open(path) as f:
        alerts = json.load(f)
    levels = [a["risk_level"] for a in alerts]
    unique, counts = np.unique(levels, return_counts=True) if levels else ([], [])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(unique, counts, color=["#2ecc71", "#f39c12", "#e74c3c"][:len(unique)])
    ax.set_title("Risk Level Distribution (from logged alerts)")
    _savefig(fig, "risk_level_distribution.png")


def plot_alert_frequency():
    path = os.path.join(config.METRICS_DIR, "alert_log.json")
    if not os.path.exists(path):
        print("[plots] Skipping alert_frequency — no alert_log.json yet.")
        return
    with open(path) as f:
        alerts = json.load(f)
    if not alerts:
        print("[plots] Skipping alert_frequency — alert_log.json is empty.")
        return
    import pandas as pd
    df = pd.DataFrame(alerts)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    counts = df.set_index("timestamp").resample("1min").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(ax=ax)
    ax.set_title("Alert Frequency Over Time")
    ax.set_ylabel("Alerts per minute")
    _savefig(fig, "alert_frequency.png")


def _savefig(fig, filename):
    out_path = os.path.join(config.EVAL_PLOTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plots] Saved {out_path}")


def run_all():
    plot_model_comparison()
    plot_confusion_matrices()
    plot_timing_comparisons()
    plot_fl_convergence()
    plot_risk_level_distribution()
    plot_alert_frequency()


if __name__ == "__main__":
    run_all()
