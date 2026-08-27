"""
preprocessing/feature_selection.py — STAGE 3

Purpose
-------
Reduce the feature set using two simple, explainable techniques suited for
a BTech project (no black-box feature-selection claims):

1. Remove near-constant / zero-variance features.
2. Remove one column from every highly-correlated pair (|corr| >= threshold).
3. Optionally rank remaining features by Random Forest importance and keep
   the top-K (importance-based selection is OPTIONAL, off by default).

This does NOT invent an "optimal" feature list — it applies transparent,
inspectable rules and reports exactly what was dropped and why.

Required packages
------------------
pip install pandas numpy scikit-learn

How to run
----------
python -m preprocessing.feature_selection

(Run this AFTER preprocess.py has produced train.csv, since selection is
fit on the training split only.)

Expected output
---------------
- Prints dropped columns and the reason.
- Saves artifacts/preprocessing/selected_features.json
"""

import json
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import config


def remove_low_variance(df: pd.DataFrame, feature_cols: list, threshold: float = 1e-5) -> list:
    kept = []
    dropped = []
    for c in feature_cols:
        if df[c].var() <= threshold:
            dropped.append(c)
        else:
            kept.append(c)
    print(f"[feature_selection] Dropped {len(dropped)} near-constant columns: {dropped or 'none'}")
    return kept


def remove_correlated(df: pd.DataFrame, feature_cols: list, threshold: float = 0.95) -> list:
    corr = df[feature_cols].corr(numeric_only=True).abs()
    to_drop = set()
    cols = list(corr.columns)
    for i, c1 in enumerate(cols):
        if c1 in to_drop:
            continue
        for c2 in cols[i + 1:]:
            if c2 in to_drop:
                continue
            if corr.loc[c1, c2] >= threshold:
                to_drop.add(c2)
    kept = [c for c in feature_cols if c not in to_drop]
    print(f"[feature_selection] Dropped {len(to_drop)} correlated columns (>= {threshold}): {sorted(to_drop) or 'none'}")
    return kept


def rank_by_importance(df: pd.DataFrame, feature_cols: list, label_col: str, top_k: int = None):
    """OPTIONAL — only run this if you explicitly want an importance-based
    cut. Returns features sorted by RF importance, most important first."""
    rf = RandomForestClassifier(
        n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    rf.fit(df[feature_cols], df[label_col])
    importances = sorted(
        zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True
    )
    ranked_cols = [c for c, _ in importances]
    if top_k:
        ranked_cols = ranked_cols[:top_k]
    return ranked_cols, importances


def run_feature_selection(use_importance_ranking: bool = False, top_k: int = None):
    with open(os.path.join(config.PREPROCESSING_DIR, "feature_list.json")) as f:
        meta = json.load(f)
    feature_cols = meta["feature_columns"]
    label_col = meta["label_column"]

    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "train.csv"))

    cols = remove_low_variance(train_df, feature_cols)
    cols = remove_correlated(train_df, cols)

    importances = None
    if use_importance_ranking:
        cols, importances = rank_by_importance(train_df, cols, label_col, top_k=top_k)

    out = {"selected_features": cols, "label_column": label_col}
    with open(os.path.join(config.PREPROCESSING_DIR, "selected_features.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"[feature_selection] Final feature count: {len(cols)} (from {len(feature_cols)})")
    print(f"[feature_selection] Saved to {config.PREPROCESSING_DIR}/selected_features.json")
    return cols


if __name__ == "__main__":
    run_feature_selection()
