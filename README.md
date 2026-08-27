# AI-Based Smart Home Security Using Hierarchical Federated Learning, Risk Assessment, Threat Progression Prediction and Intelligent Real-Time Alerting

BTech mini-project (team of 4). This repository is a **working code
scaffold**: every module runs and produces real, measured output on
whatever real dataset you provide — nothing here fabricates accuracy,
column names, or class lists.

## What's implemented vs what's a placeholder

| Status | Meaning |
|---|---|
| **IMPLEMENTED** | Code exists and runs end-to-end. |
| **MEASURED** | A number was actually produced by running the code on real data. |
| **PROPOSED** | Design/architecture exists in code but needs your real dataset to produce results. |

This scaffold ships with **no trained models and no measured metrics**,
because it has not been run against your actual downloaded CICIoT2023 /
Edge-IIoTset data yet. `config.LABEL_COLUMN` starts as `None` on purpose —
`preprocessing/preprocess.py` will refuse to run until you set it, so no
step can silently guess your dataset's structure.

## Architecture (fixed — do not remove modules)

```
Smart Home → IoT Devices (simulated) → Preprocessing → Feature Engineering
→ Local Threat Detection → Risk Assessment → Threat Progression Prediction
→ Local Federated Training → 3 Edge Servers → Regional Aggregation
→ Global Model → Intelligent Decision Engine → Risk-Based Real-Time Alert
→ Streamlit Security Dashboard
```

## Project structure

```
smart_home_security/
├── data/{raw,processed,external}/
├── preprocessing/{preprocess.py, feature_selection.py}
├── models/{random_forest.py, xgboost_model.py, lightgbm_model.py, gru_model.py, data_utils.py}
├── federated/{client.py, edge_server.py, fedprox.py, aggregation.py, model_arch.py}
├── risk/{risk_assessment.py, threat_progression.py}
├── alerts/alert_manager.py
├── dashboard/app.py
├── evaluation/{metrics.py, plots.py, results/, confusion_matrices/, plots/}
├── simulation/device_simulator.py
├── artifacts/{models, preprocessing, metrics, plots}
├── config.py
├── main.py
├── dataset_inspection.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Step-by-step run order

### 1. Get the datasets
- Primary: [CICIoT2023 Stratified Dataset](https://www.kaggle.com/datasets/raqeeb24/ciciot-2023-stratified-dataset) → place CSV(s) in `data/raw/`
- Secondary (external validation only): [Edge-IIoTset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) → place a selected CSV in `data/external/` (don't download the full 11+ GB package)

### 2. Inspect the real dataset (mandatory — do not skip)
```bash
python dataset_inspection.py --file data/raw/<your_file>.csv
```
Read the printed report. Note the real label column and class names, then
edit `config.py`:
```python
PRIMARY_DATASET_FILENAME = "<your_file>.csv"
LABEL_COLUMN = "<the real label column name>"
IDENTIFIER_COLUMNS = ["<any ID/leakage columns to drop>"]
```

### 3. Preprocess + feature selection
```bash
python main.py --stage preprocess --file data/raw/<your_file>.csv
```

### 4. Train threat detectors (Random Forest, XGBoost, LightGBM)
```bash
python main.py --stage detectors
```
This prints a ranked comparison — the "best" model is whichever actually
scores highest, not assumed in advance.

### 5. Threat progression (GRU)
```bash
python -m models.gru_model
```
Ships with a synthetic demo risk-score walk by default (clearly logged as
such). Feed it real historical risk scores from `risk/risk_assessment.py`
output for a real measurement.

### 6. Hierarchical federated learning (FedProx, 3 edge servers)
```bash
python main.py --stage federated
```

### 7. Generate evaluation plots
```bash
python main.py --stage evaluate
```

### 8. Run the dashboard
```bash
python main.py --stage dashboard
# or: streamlit run dashboard/app.py
```
Use **Start Simulation** / **Generate Test Event** on the Dashboard page —
this pulls real rows from your processed test split and assigns them to
simulated devices (clearly labeled **SIMULATION MODE**, never presented as
live physical traffic).

## FAST_MODE

`config.FAST_MODE = True` (default) uses a stratified sample, fewer FL
rounds, and a lighter GRU so everything runs on a normal student laptop.
Set `FAST_MODE = False` for final, fuller experiments once the pipeline
works end to end.

## Honesty guardrails baked into the code

- No column names, label columns, or class lists are assumed anywhere —
  everything is read from your actual dataset after inspection.
- Every metrics file is tagged `MEASURED`, `PROPOSED`, or `EXPECTED`.
- The risk score is explicitly labeled a project-level demo score, not a
  certified cybersecurity risk standard (e.g. not CVSS).
- Threat progression predicts a *trend*, never an exact future attack.
- Simulated homes/devices are explicitly documented as software
  simulations — no physical IoT deployment is claimed.
- Device "isolation" is only ever a *recommended action* string, never a
  claim that a physical device was actually isolated.
- Local alerts work fully offline; remote notification (SMS/email) is kept
  separate and optional, not a core dependency.

## Still to do (not yet built in this scaffold)

- Academic write-up (abstract, methodology, results write-up) — Stage 19
- Slide-by-slide PPT content — Stage 20
- Viva Q&A prep sheet — Stage 21

These are documentation deliverables best generated *after* you have real
MEASURED numbers from steps 1–7 above, so they don't need placeholder
results. Ask for these once you have run the pipeline on your real data.
