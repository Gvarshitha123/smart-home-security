"""
models/data_utils.py

Small shared helper so every model script loads train/val/test the same way,
using the feature list produced by feature_selection.py (falls back to the
full feature_list.json if selection hasn't been run yet).
"""

import json
import os

import pandas as pd

import config


def load_splits():
    sel_path = os.path.join(config.PREPROCESSING_DIR, "selected_features.json")
    full_path = os.path.join(config.PREPROCESSING_DIR, "feature_list.json")

    if os.path.exists(sel_path):
        with open(sel_path) as f:
            meta = json.load(f)
    elif os.path.exists(full_path):
        with open(full_path) as f:
            meta = json.load(f)
            meta["selected_features"] = meta["feature_columns"]
    else:
        raise FileNotFoundError(
            "No feature list found. Run preprocessing/preprocess.py "
            "(and optionally preprocessing/feature_selection.py) first."
        )

    feature_cols = meta["selected_features"]
    label_col = meta["label_column"]

    train_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "test.csv"))

    X_train, y_train = train_df[feature_cols], train_df[label_col]
    X_val, y_val = val_df[feature_cols], val_df[label_col]
    X_test, y_test = test_df[feature_cols], test_df[label_col]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, label_col
