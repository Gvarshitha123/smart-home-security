"""
main.py

Purpose
-------
Convenience CLI to run the pipeline stages in order (after you've inspected
your real dataset and set config.LABEL_COLUMN / config.IDENTIFIER_COLUMNS).
This does NOT replace running dataset_inspection.py first — that step
requires you to read the output and manually confirm settings in config.py.

How to run
----------
python main.py --stage preprocess --file data/raw/your_file.csv
python main.py --stage detectors
python main.py --stage federated
python main.py --stage evaluate
python main.py --stage dashboard   (equivalent to: streamlit run dashboard/app.py)
"""

import argparse
import subprocess
import sys


def run_preprocess(file_path: str):
    from preprocessing.preprocess import run_pipeline
    run_pipeline(file_path)
    from preprocessing.feature_selection import run_feature_selection
    run_feature_selection()


def run_detectors():
    from models.random_forest import train_and_evaluate as rf_train
    from models.xgboost_model import train_and_evaluate as xgb_train
    from models.lightgbm_model import train_and_evaluate as lgbm_train
    rf_train()
    xgb_train()
    lgbm_train()
    from evaluation.metrics import _print_comparison
    _print_comparison()


def run_federated():
    from federated.aggregation import run_hierarchical_fedprox, run_centralized_baseline
    run_hierarchical_fedprox()
    run_centralized_baseline()


def run_evaluate():
    from evaluation.plots import run_all
    run_all()


def run_dashboard():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                         choices=["preprocess", "detectors", "federated", "evaluate", "dashboard"])
    parser.add_argument("--file", help="Raw CSV path (only needed for --stage preprocess)")
    args = parser.parse_args()

    if args.stage == "preprocess":
        if not args.file:
            raise SystemExit("--file is required for --stage preprocess")
        run_preprocess(args.file)
    elif args.stage == "detectors":
        run_detectors()
    elif args.stage == "federated":
        run_federated()
    elif args.stage == "evaluate":
        run_evaluate()
    elif args.stage == "dashboard":
        run_dashboard()
