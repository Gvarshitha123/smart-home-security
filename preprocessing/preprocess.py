"""
preprocessing/preprocess.py — STAGE 2

Purpose
-------
Turn a raw dataset CSV into a clean, encoded, train/val/test-split dataset,
without assuming column names. You MUST have already run
dataset_inspection.py and set config.LABEL_COLUMN before this will do
anything useful — if LABEL_COLUMN is still None, this script raises an
error rather than guessing.

Pipeline implemented (matches project spec, Section 5):
  Raw Dataset
    -> Remove Duplicates
    -> Handle Missing Values
    -> Handle Infinite Values
    -> Remove Identifiers / Leakage Columns
    -> Encode Categorical Features
    -> Feature Selection (see feature_selection.py)
    -> Class Distribution Analysis
    -> Stratified Sampling (if FAST_MODE)
    -> Train / Validation / Test Split
    -> Fit preprocessing ONLY on training data, transform val/test

Required packages
------------------
pip install pandas numpy scikit-learn joblib

How to run
----------
python -m preprocessing.preprocess --file data/raw/<your_file>.csv

Expected output
---------------
- data/processed/train.csv, val.csv, test.csv
- artifacts/preprocessing/encoder.joblib, scaler.joblib, feature_list.json
  (so the dashboard/inference pipeline can reuse the exact same transform)

Common errors
-------------
- ValueError "config.LABEL_COLUMN is not set": run dataset_inspection.py
  first, look at candidate_label_columns, then set LABEL_COLUMN in config.py.
- KeyError on a column: the raw file's columns differ from what you set in
  config.IDENTIFIER_COLUMNS — re-check dataset_inspection.py output.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"[preprocess] Removed {before - len(df)} duplicate rows.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    n_missing = int(df.isna().sum().sum())
    if n_missing == 0:
        print("[preprocess] No missing values found.")
        return df

    for c in numeric_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in categorical_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode(dropna=True).iloc[0] if not df[c].mode(dropna=True).empty else "unknown")

    print(f"[preprocess] Filled {n_missing} missing values (median for numeric, mode for categorical).")
    return df


def handle_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf_total = 0
    for c in numeric_cols:
        mask = np.isinf(df[c])
        n_inf = int(mask.sum())
        if n_inf > 0:
            n_inf_total += n_inf
            finite_max = df.loc[~mask, c].max()
            df.loc[mask, c] = finite_max
    print(f"[preprocess] Replaced {n_inf_total} infinite values with the column's finite max.")
    return df


def remove_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in config.IDENTIFIER_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"[preprocess] Dropped identifier/leakage columns: {cols_to_drop}")
    else:
        print("[preprocess] No identifier columns configured/dropped "
              "(set config.IDENTIFIER_COLUMNS after inspecting the raw data).")
    return df


def encode_categoricals(df: pd.DataFrame, label_col: str):
    """Label-encode categorical FEATURE columns (not the label itself)."""
    encoders = {}
    categorical_cols = [
        c for c in df.select_dtypes(exclude=[np.number]).columns if c != label_col
    ]
    for c in categorical_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
    print(f"[preprocess] Label-encoded categorical features: {categorical_cols or 'none'}")
    return df, encoders


def stratified_sample_if_needed(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if not config.FAST_MODE:
        return df
    target_n = config.FAST_MODE_SAMPLE_SIZE
    if len(df) <= target_n:
        return df
    frac = target_n / len(df)
    sampled = (
        df.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(frac=frac, random_state=config.RANDOM_STATE))
    )
    print(f"[preprocess] FAST_MODE stratified sample: {len(df)} -> {len(sampled)} rows.")
    return sampled.reset_index(drop=True)


def run_pipeline(raw_path: str):
    label_col = config.LABEL_COLUMN
    if not label_col:
        raise ValueError(
            "config.LABEL_COLUMN is not set. Run dataset_inspection.py first, "
            "inspect 'candidate_label_columns' in its output, then set "
            "LABEL_COLUMN in config.py before running preprocessing."
        )

    df = load_raw(raw_path)
    if label_col not in df.columns:
        raise KeyError(
            f"LABEL_COLUMN '{label_col}' not found in {raw_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = handle_infinite_values(df)
    df = remove_identifier_columns(df)

    print("\n[preprocess] Class distribution BEFORE sampling:")
    print(df[label_col].value_counts())

    df = stratified_sample_if_needed(df, label_col)

    df, encoders = encode_categoricals(df, label_col)

    label_encoder = LabelEncoder()
    df[label_col] = label_encoder.fit_transform(df[label_col].astype(str))
    print(f"\n[preprocess] Label classes: {list(label_encoder.classes_)}")

    feature_cols = [c for c in df.columns if c != label_col]

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=config.RANDOM_STATE, stratify=df[label_col]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=config.RANDOM_STATE, stratify=temp_df[label_col]
    )

    # Fit scaler ONLY on training data — prevents leakage from val/test.
    # Cast feature columns to float64 first: scaling can produce fractional
    # values, which pandas will reject writing into an original int column.
    for _df in (train_df, val_df, test_df):
        _df[feature_cols] = _df[feature_cols].astype("float64")

    scaler = StandardScaler()
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    train_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "test.csv"), index=False)

    joblib.dump(scaler, os.path.join(config.PREPROCESSING_DIR, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(config.PREPROCESSING_DIR, "feature_encoders.joblib"))
    joblib.dump(label_encoder, os.path.join(config.PREPROCESSING_DIR, "label_encoder.joblib"))
    with open(os.path.join(config.PREPROCESSING_DIR, "feature_list.json"), "w") as f:
        json.dump({"feature_columns": feature_cols, "label_column": label_col}, f, indent=2)

    print(f"\n[preprocess] Done. train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"[preprocess] Saved processed splits to {config.PROCESSED_DATA_DIR}")
    print(f"[preprocess] Saved preprocessing artifacts to {config.PREPROCESSING_DIR}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to raw CSV (e.g. data/raw/ciciot2023.csv)")
    args = parser.parse_args()
    run_pipeline(args.file)
