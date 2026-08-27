"""
dashboard/app.py — STAGES 15, 21, 22, 23, 37, 38

Purpose
-------
The main Streamlit Security Dashboard tying together every module:
threat detection, risk assessment, threat progression, decision engine,
alert manager, and the simulated device layer.

All numbers shown are computed live from actual saved model
artifacts/metrics — nothing on this page is hard-coded (Section 21/38).
If a required artifact (trained model, processed data) doesn't exist yet,
the relevant section says so instead of faking a value.

Required packages
------------------
pip install streamlit pandas numpy joblib torch plotly

How to run
----------
streamlit run dashboard/app.py

Expected output
---------------
A local web app (usually http://localhost:8501) with pages: Dashboard,
Threat Detection, Risk Assessment, Threat Progression, Federated Learning,
Alerts, Device Monitoring, Evaluation, About Project.

Common errors
-------------
- "No trained detector found": run models/random_forest.py (or xgboost/
  lightgbm) at least once first.
- "No processed data found": run preprocessing/preprocess.py first.
"""

import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from alerts.alert_manager import AlertManager, DecisionEngine
from evaluation.metrics import compare_all_models
from risk.risk_assessment import RiskAssessor
from risk.threat_progression import ThreatProgressionTracker
from simulation.device_simulator import DeviceEventSimulator

st.set_page_config(page_title="Smart Home Security Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------
@st.cache_resource
def get_alert_manager():
    return AlertManager()


@st.cache_resource
def get_decision_engine():
    return DecisionEngine()


@st.cache_resource
def get_risk_assessor():
    return RiskAssessor()


@st.cache_resource
def get_progression_tracker():
    return ThreatProgressionTracker()


@st.cache_resource
def get_simulator():
    return DeviceEventSimulator()


def load_best_detector():
    """Loads whichever detector currently has the best MEASURED F1 score.
    Returns (model, model_name, label_encoder, feature_cols) or None."""
    rows = compare_all_models()
    if not rows:
        return None
    best_name = rows[0]["model"]
    model_file_map = {
        "RandomForest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
        "LightGBM": "lightgbm.joblib",
    }
    path = os.path.join(config.MODEL_DIR, model_file_map.get(best_name, ""))
    if not os.path.exists(path):
        return None
    model = joblib.load(path)

    label_encoder_path = os.path.join(config.PREPROCESSING_DIR, "label_encoder.joblib")
    label_encoder = joblib.load(label_encoder_path) if os.path.exists(label_encoder_path) else None

    sel_path = os.path.join(config.PREPROCESSING_DIR, "selected_features.json")
    full_path = os.path.join(config.PREPROCESSING_DIR, "feature_list.json")
    feat_path = sel_path if os.path.exists(sel_path) else full_path
    with open(feat_path) as f:
        meta = json.load(f)
    feature_cols = meta.get("selected_features") or meta.get("feature_columns")
    label_col = meta["label_column"]

    return model, best_name, label_encoder, feature_cols, label_col


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "Dashboard", "Threat Detection", "Risk Assessment", "Threat Progression",
    "Federated Learning", "Alerts", "Device Monitoring", "Evaluation", "About Project",
]
page = st.sidebar.radio("Navigate", PAGES)

alert_manager = get_alert_manager()
decision_engine = get_decision_engine()
risk_assessor = get_risk_assessor()
progression_tracker = get_progression_tracker()

if "sim_running" not in st.session_state:
    st.session_state.sim_running = False


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
if page == "Dashboard":
    st.title("Smart Home Security — Live Dashboard")
    st.caption("SIMULATION MODE — events are drawn from real processed dataset rows "
               "assigned to simulated devices, not live physical network traffic.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Security Score", f"{alert_manager.current_security_score()}%")
    col2.metric("Active Threats", alert_manager.active_threat_count())
    recent = alert_manager.recent_alerts(limit=1)
    current_level = recent[0]["risk_level"] if recent else "N/A"
    col3.metric("Current Risk Level", current_level)
    col4.metric("Most Vulnerable Device", alert_manager.most_vulnerable_device())

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Start / Stop Simulation")
        b1, b2, b3 = st.columns(3)
        if b1.button("Start Simulation"):
            st.session_state.sim_running = True
        if b2.button("Stop Simulation"):
            st.session_state.sim_running = False
        generate = b3.button("Generate Test Event")

        detector = load_best_detector()
        if detector is None:
            st.warning("No trained detector found yet — run models/random_forest.py "
                       "(or xgboost_model.py / lightgbm_model.py) first.")
        elif generate or st.session_state.sim_running:
            sim = get_simulator()
            try:
                event = sim.next_event()
            except FileNotFoundError as e:
                st.error(str(e))
                event = None

            if event:
                model, model_name, label_encoder, feature_cols, label_col = detector
                row = event["row"]
                X = pd.DataFrame([{c: row[c] for c in feature_cols}])
                pred_idx = model.predict(X)[0]
                proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
                confidence = float(max(proba)) if proba is not None else 1.0
                predicted_class = (
                    label_encoder.inverse_transform([pred_idx])[0] if label_encoder is not None else str(pred_idx)
                )

                recent_count = alert_manager.active_threat_count()
                progression_tracker.update(event["device"], confidence)
                trend_info = progression_tracker.get_trend(event["device"])

                risk = risk_assessor.assess(
                    confidence=confidence, predicted_class=predicted_class,
                    recent_alert_count=recent_count, threat_trend=trend_info["trend"],
                )
                decision = decision_engine.decide(confidence=confidence, risk_level=risk["risk_level"])

                st.info(f"**Device:** {event['device']} | **Detected:** {predicted_class} | "
                        f"**Confidence:** {confidence:.2f} | **Risk:** {risk['risk_score']} "
                        f"({risk['risk_level']}) | **Trend:** {trend_info['trend']} ({trend_info['status']})")

                if decision["action"] != "MONITOR":
                    alert_manager.raise_alert(
                        device=event["device"], threat=predicted_class, confidence=confidence,
                        risk_score=risk["risk_score"], risk_level=risk["risk_level"],
                        trend=trend_info["trend"], recommended_action=decision["recommended_action"],
                    )
                    st.warning(f"Recommended action: {decision['recommended_action']}")

    with c2:
        st.subheader("Recent Alerts")
        recents = alert_manager.recent_alerts(limit=8)
        if recents:
            st.dataframe(pd.DataFrame(recents), use_container_width=True)
        else:
            st.write("No alerts logged yet — generate a test event to populate this.")


# ---------------------------------------------------------------------------
# Page: Threat Detection
# ---------------------------------------------------------------------------
elif page == "Threat Detection":
    st.title("Threat Detection")
    rows = compare_all_models()
    if not rows:
        st.warning("No detector metrics found yet. Train Random Forest / XGBoost / "
                   "LightGBM first (models/*.py).")
    else:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(f"Best detector by measured F1-score: **{rows[0]['model']}**")


# ---------------------------------------------------------------------------
# Page: Risk Assessment
# ---------------------------------------------------------------------------
elif page == "Risk Assessment":
    st.title("Risk Assessment")
    st.write("Project-level cybersecurity risk score — not a certified risk measurement.")
    conf = st.slider("Model confidence", 0.0, 1.0, 0.9)
    attack = st.text_input("Predicted attack class", "DDoS")
    freq = st.slider("Recent alert count", 0, 10, 3)
    trend = st.selectbox("Threat trend", ["STABLE", "INCREASING", "CRITICAL"])
    result = risk_assessor.assess(confidence=conf, predicted_class=attack,
                                   recent_alert_count=freq, threat_trend=trend)
    st.json(result)


# ---------------------------------------------------------------------------
# Page: Threat Progression
# ---------------------------------------------------------------------------
elif page == "Threat Progression":
    st.title("Threat Progression Prediction")
    st.caption("Estimates the likely progression of an observed risk pattern — "
               "not an exact future attack prediction.")
    device = st.selectbox("Device", config.SIMULATED_DEVICES)
    trend_info = progression_tracker.get_trend(device)
    st.json(trend_info)
    st.caption("Trend history builds up as you generate simulation events on the Dashboard page.")


# ---------------------------------------------------------------------------
# Page: Federated Learning
# ---------------------------------------------------------------------------
elif page == "Federated Learning":
    st.title("Hierarchical Federated Learning (FedProx)")
    hist_path = os.path.join(config.METRICS_DIR, "fl_round_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        st.line_chart(pd.DataFrame(history).set_index("round")[["global_accuracy", "global_f1_weighted"]])
        st.dataframe(pd.DataFrame(history), use_container_width=True)
    else:
        st.warning("No FL round history yet — run federated/aggregation.py first.")

    cvf_path = os.path.join(config.METRICS_DIR, "centralized_vs_federated.json")
    if os.path.exists(cvf_path):
        with open(cvf_path) as f:
            cvf = json.load(f)
        st.subheader("Centralized vs Hierarchical Federated (measured)")
        st.json(cvf)


# ---------------------------------------------------------------------------
# Page: Alerts
# ---------------------------------------------------------------------------
elif page == "Alerts":
    st.title("Alert History")
    history = alert_manager.alert_history()
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True)
    else:
        st.write("No alerts logged yet.")


# ---------------------------------------------------------------------------
# Page: Device Monitoring
# ---------------------------------------------------------------------------
elif page == "Device Monitoring":
    st.title("Device Monitoring (Simulated Devices)")
    st.caption("These are software simulations representing heterogeneous smart-home "
               "devices — no physical IoT hardware is connected.")
    sim = get_simulator()
    st.json(sim.device_status())


# ---------------------------------------------------------------------------
# Page: Evaluation
# ---------------------------------------------------------------------------
elif page == "Evaluation":
    st.title("Evaluation")
    st.write("Generated plots (evaluation/plots.py) appear here once produced:")
    plots_dir = config.EVAL_PLOTS_DIR
    if os.path.exists(plots_dir) and os.listdir(plots_dir):
        for fname in sorted(os.listdir(plots_dir)):
            if fname.endswith(".png"):
                st.image(os.path.join(plots_dir, fname), caption=fname)
    else:
        st.warning("No plots generated yet — run: python -m evaluation.plots")


# ---------------------------------------------------------------------------
# Page: About Project
# ---------------------------------------------------------------------------
elif page == "About Project":
    st.title("About This Project")
    st.markdown("""
**AI-Based Smart Home Security Using Hierarchical Federated Learning,
Risk Assessment, Threat Progression Prediction and Intelligent
Real-Time Alerting** — BTech mini-project.

Architecture: Smart Home → IoT Devices (simulated) → Preprocessing →
Feature Engineering → Local Threat Detection → Risk Assessment →
Threat Progression Prediction → Local Federated Training → 3 Edge
Servers → Regional Aggregation → Global Model → Intelligent Decision
Engine → Risk-Based Real-Time Alert → Security Dashboard.

**Honesty notes:**
- Smart-home clients are software simulations, not physical deployments.
- The risk score is a project-level demo metric, not a certified standard.
- Threat progression predicts the likely trend of observed risk, not an
  exact future attack.
- Any metric shown elsewhere in this app is either MEASURED from your own
  trained models, or explicitly labeled PROPOSED/EXPECTED.
""")
