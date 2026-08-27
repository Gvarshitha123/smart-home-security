"""
config.py — Central configuration for the Smart Home Security project.

Nothing in this project should hard-code paths, thresholds, or FL settings
directly in module files. Everything tunable lives here.
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Performance mode
# ---------------------------------------------------------------------------
# FAST_MODE=True  -> small stratified sample, fewer FL rounds, lighter GRU.
#                    Use this for development and for viva demonstrations.
# FAST_MODE=False -> full pipeline for final experiments (needs more RAM/time).
FAST_MODE = True

FAST_MODE_SAMPLE_SIZE = 20000       # rows sampled per class-balanced draw
FULL_MODE_SAMPLE_SIZE = None        # None = use all available rows

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_DIR = os.path.join(ARTIFACT_DIR, "models")
PREPROCESSING_DIR = os.path.join(ARTIFACT_DIR, "preprocessing")
METRICS_DIR = os.path.join(ARTIFACT_DIR, "metrics")
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots")

EVAL_RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")
EVAL_CONFUSION_DIR = os.path.join(BASE_DIR, "evaluation", "confusion_matrices")
EVAL_PLOTS_DIR = os.path.join(BASE_DIR, "evaluation", "plots")

for _d in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
    MODEL_DIR, PREPROCESSING_DIR, METRICS_DIR, PLOTS_DIR,
    EVAL_RESULTS_DIR, EVAL_CONFUSION_DIR, EVAL_PLOTS_DIR,
]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset placeholders — DO NOT assume column names.
# These are filled in only after dataset_inspection.py has been run on the
# real downloaded file. Until then they stay None / empty.
# ---------------------------------------------------------------------------
PRIMARY_DATASET_FILENAME = None     # e.g. "CICIoT2023_stratified.csv"
LABEL_COLUMN = None                 # e.g. "label" — set after inspection
IDENTIFIER_COLUMNS = []             # columns to drop (IDs, timestamps used as leakage, etc.)

# ---------------------------------------------------------------------------
# Federated learning topology (fixed by project architecture — do not change)
# ---------------------------------------------------------------------------
NUM_CLIENTS = 9
NUM_EDGE_SERVERS = 3
CLIENTS_PER_EDGE = 3

EDGE_SERVER_MAP = {
    "edge_server_1": ["home_1", "home_2", "home_3"],
    "edge_server_2": ["home_4", "home_5", "home_6"],
    "edge_server_3": ["home_7", "home_8", "home_9"],
}

# FedProx
FEDPROX_MU = 0.01                 # proximal term weight
FED_ROUNDS = 5 if FAST_MODE else 20
LOCAL_EPOCHS = 1 if FAST_MODE else 5
LOCAL_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Risk assessment thresholds (configurable; NOT a certified risk standard)
# ---------------------------------------------------------------------------
RISK_LOW_THRESHOLD = 0.40      # below this -> LOW
RISK_MEDIUM_THRESHOLD = 0.70   # below this (and >= low) -> MEDIUM; above -> HIGH

# ---------------------------------------------------------------------------
# Decision engine thresholds
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.60    # minimum model confidence to act on a prediction

# ---------------------------------------------------------------------------
# Threat progression (GRU) settings
# ---------------------------------------------------------------------------
GRU_SEQUENCE_LENGTH = 10 if FAST_MODE else 20
GRU_HIDDEN_SIZE = 16 if FAST_MODE else 32
GRU_NUM_LAYERS = 1 if FAST_MODE else 2
GRU_EPOCHS = 5 if FAST_MODE else 20

# ---------------------------------------------------------------------------
# Simulated devices (software layer only — no physical IoT hardware)
# ---------------------------------------------------------------------------
SIMULATED_DEVICES = [
    "CCTV Camera",
    "Smart TV",
    "Smart Lock",
    "Smart Light",
    "Smart Thermostat",
    "Smart Speaker",
    "IoT Sensor",
]

# ---------------------------------------------------------------------------
# Claim labels — use these constants in reports/UI instead of free text,
# to keep IMPLEMENTED / MEASURED / PROPOSED / EXPECTED usage consistent.
# ---------------------------------------------------------------------------
class ClaimStatus:
    IMPLEMENTED = "IMPLEMENTED"
    MEASURED = "MEASURED"
    PROPOSED = "PROPOSED"
    EXPECTED = "EXPECTED"
