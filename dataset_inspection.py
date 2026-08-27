"""
dataset_inspection.py — STAGE 1

Purpose
-------
Inspect whatever CSV file(s) you place in data/raw/ (primary: CICIoT2023
stratified) and data/external/ (secondary: Edge-IIoTset) WITHOUT assuming
any column names, label column, or class list.

Run this BEFORE writing any preprocessing code. The output tells you what
the real dataset looks like so config.LABEL_COLUMN and the preprocessing
pipeline can be set correctly.

Required packages
------------------
pip install pandas numpy

How to run
----------
python dataset_inspection.py --file data/raw/<your_file>.csv
python dataset_inspection.py --file data/external/<your_file>.csv

Expected output
---------------
A printed report + a saved JSON report in artifacts/metrics/, containing:
shape, columns, dtypes, missing values, duplicate rows, candidate label
columns, numeric/categorical columns, class distribution (if a label-like
column is found), constant columns, high-cardinality columns, and possible
identifier/leakage columns.

Common errors
-------------
- FileNotFoundError: the path you passed does not exist yet. Download the
  dataset from Kaggle and place the CSV under data/raw/ or data/external/.
- MemoryError on very large files: re-run with --sample 200000 to inspect
  a random sample of rows instead of the full file.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import config


def _find_candidate_label_columns(df: pd.DataFrame) -> list:
    """Heuristic only — this does NOT decide the label column for you.
    It just surfaces likely candidates by name and low cardinality."""
    name_hints = ["label", "class", "attack", "category", "target", "type"]
    candidates = []
    for col in df.columns:
        lower = col.lower()
        if any(hint in lower for hint in name_hints):
            candidates.append(col)
    # Also flag any low-cardinality object/int column as a possible label
    for col in df.columns:
        if col in candidates:
            continue
        nunique = df[col].nunique(dropna=True)
        if df[col].dtype == object and 1 < nunique <= 50:
            candidates.append(col)
    return candidates


def _find_identifier_like_columns(df: pd.DataFrame) -> list:
    """Heuristic surfacing of possible ID / leakage columns (e.g. IP, MAC,
    flow ID, timestamp). You must manually confirm these before dropping."""
    name_hints = ["id", "ip", "mac", "flow", "timestamp", "time", "index"]
    return [c for c in df.columns if any(h in c.lower() for h in name_hints)]


def inspect_file(path: str, sample: int | None = None) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Download the dataset from Kaggle and place "
            f"the CSV there before running inspection."
        )

    print(f"\nReading: {path}")
    if sample:
        # Fast, memory-safe peek: read a random sample of rows via chunking.
        total_rows = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
        skip = sorted(
            np.random.RandomState(config.RANDOM_STATE).choice(
                range(1, total_rows + 1),
                size=max(total_rows - sample, 0),
                replace=False,
            )
        ) if total_rows > sample else []
        df = pd.read_csv(path, skiprows=skip)
    else:
        df = pd.read_csv(path)

    report = {}
    report["file_name"] = os.path.basename(path)
    report["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    report["column_names"] = list(df.columns)
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

    report["missing_values"] = {
        c: int(v) for c, v in df.isna().sum().items() if v > 0
    }
    report["infinite_values"] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        n_inf = int(np.isinf(df[c].to_numpy(dtype="float64", na_value=0.0)).sum())
        if n_inf > 0:
            report["infinite_values"][c] = n_inf

    report["duplicate_rows"] = int(df.duplicated().sum())

    report["numerical_columns"] = numeric_cols
    report["categorical_columns"] = df.select_dtypes(exclude=[np.number]).columns.tolist()

    candidate_labels = _find_candidate_label_columns(df)
    report["candidate_label_columns"] = candidate_labels

    if candidate_labels:
        report["class_distribution"] = {}
        for col in candidate_labels:
            report["class_distribution"][col] = (
                df[col].value_counts(dropna=False).to_dict()
            )

    report["feature_cardinality"] = {
        c: int(df[c].nunique(dropna=True)) for c in df.columns
    }

    report["constant_columns"] = [
        c for c in df.columns if df[c].nunique(dropna=True) <= 1
    ]

    report["possible_identifier_columns"] = _find_identifier_like_columns(df)

    # High correlation among numeric features (possible redundancy/leakage)
    high_corr_pairs = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1:]:
                val = corr.loc[c1, c2]
                if pd.notna(val) and val >= 0.95:
                    high_corr_pairs.append({"col_1": c1, "col_2": c2, "correlation": round(float(val), 4)})
    report["highly_correlated_feature_pairs"] = high_corr_pairs

    report["memory_usage_mb"] = round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)

    return report


def _print_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print(f"DATASET INSPECTION REPORT — {report['file_name']}")
    print("=" * 70)
    print(f"Shape: {report['shape']['rows']} rows x {report['shape']['columns']} columns")
    print(f"Memory usage: {report['memory_usage_mb']} MB")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"\nColumns ({len(report['column_names'])}):")
    for c in report["column_names"]:
        print(f"  - {c} ({report['dtypes'][c]})")
    print(f"\nMissing values: {report['missing_values'] or 'None found'}")
    print(f"Infinite values: {report['infinite_values'] or 'None found'}")
    print(f"Constant columns: {report['constant_columns'] or 'None found'}")
    print(f"Possible identifier/leakage columns: {report['possible_identifier_columns'] or 'None found'}")
    print(f"Highly correlated feature pairs (>=0.95): {len(report['highly_correlated_feature_pairs'])}")
    print(f"\nCandidate label columns (heuristic — confirm manually!): "
          f"{report['candidate_label_columns'] or 'None found'}")
    if report.get("class_distribution"):
        for col, dist in report["class_distribution"].items():
            print(f"\n  Class distribution for candidate column '{col}':")
            for cls, count in dist.items():
                print(f"    {cls}: {count}")
    print("\n" + "=" * 70)
    print("STOP: Confirm the real label column and class list from the")
    print("output above before writing/running any preprocessing code.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect a raw dataset file.")
    parser.add_argument("--file", required=True, help="Path to the CSV file to inspect")
    parser.add_argument("--sample", type=int, default=None,
                         help="Optional: inspect a random sample of N rows (memory-safe for huge files)")
    args = parser.parse_args()

    report = inspect_file(args.file, sample=args.sample)
    _print_report(report)

    out_name = os.path.splitext(os.path.basename(args.file))[0] + "_inspection.json"
    out_path = os.path.join(config.METRICS_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Full JSON report saved to: {out_path}")


if __name__ == "__main__":
    main()
