"""
alerts/alert_manager.py — STAGE 13 (Decision Engine) + STAGE 14 (Alert System)

Purpose
-------
Decision Engine: takes (prediction, confidence, risk_score, risk_level,
threat_progression, device) and decides an action:
  LOW    -> Monitor (no alert)
  MEDIUM -> User Notification
  HIGH   -> Immediate Alert + recommended action + logging

Alert Manager: builds a structured alert record, appends it to a local
JSON log (works fully offline — LOCAL RESPONSE), and exposes recent/active
alerts for the dashboard.

Important distinctions preserved here (Section 18/19/42):
- If device isolation is not physically implemented, the recommended
  action text is "Isolate device" — never a claim that isolation actually
  happened.
- LOCAL RESPONSE (this module) is separate from REMOTE NOTIFICATION, which
  is optional and not implemented as a hard dependency here.

Required packages
------------------
pip install (none beyond the standard library + config)

How to run
----------
from alerts.alert_manager import DecisionEngine, AlertManager
engine = DecisionEngine()
manager = AlertManager()

decision = engine.decide(confidence=0.94, risk_level="HIGH")
if decision["action"] != "MONITOR":
    alert = manager.raise_alert(device="CCTV Camera", threat="DDoS",
                                 confidence=0.94, risk_score=0.91,
                                 risk_level="HIGH", trend="INCREASING",
                                 recommended_action=decision["recommended_action"])

Expected output
---------------
Alerts are appended to artifacts/metrics/alert_log.json (used by
evaluation/plots.py for risk-level distribution and alert-frequency plots).
"""

import json
import os
from datetime import datetime, timezone

import config

RECOMMENDED_ACTIONS = {
    "LOW": "Monitor",
    "MEDIUM": "Notify user / review device activity",
    "HIGH": "Isolate device (recommended — not automatically enforced), investigate traffic",
}


class DecisionEngine:
    def __init__(self, confidence_threshold: float = config.CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def decide(self, confidence: float, risk_level: str) -> dict:
        if confidence < self.confidence_threshold:
            # Low-confidence predictions are not acted on, to reduce false alerts.
            return {"action": "MONITOR", "recommended_action": RECOMMENDED_ACTIONS["LOW"]}

        if risk_level == "LOW":
            action = "MONITOR"
        elif risk_level == "MEDIUM":
            action = "USER_NOTIFICATION"
        else:
            action = "IMMEDIATE_ALERT"

        return {"action": action, "recommended_action": RECOMMENDED_ACTIONS[risk_level]}


class AlertManager:
    def __init__(self, log_path: str = None):
        self.log_path = log_path or os.path.join(config.METRICS_DIR, "alert_log.json")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump([], f)

    def _load(self) -> list:
        with open(self.log_path) as f:
            return json.load(f)

    def _save(self, alerts: list):
        with open(self.log_path, "w") as f:
            json.dump(alerts, f, indent=2)

    def raise_alert(self, device: str, threat: str, confidence: float,
                     risk_score: float, risk_level: str, trend: str,
                     recommended_action: str) -> dict:
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "detected_threat": threat,
            "confidence": round(float(confidence), 4),
            "risk_score": round(float(risk_score), 4),
            "risk_level": risk_level,
            "threat_trend": trend,
            "recommended_action": recommended_action,
            "channel": "LOCAL_RESPONSE",  # see Section 19 — remote notification is separate/optional
        }
        alerts = self._load()
        alerts.append(alert)
        self._save(alerts)
        return alert

    def recent_alerts(self, limit: int = 10) -> list:
        return self._load()[-limit:][::-1]

    def alert_history(self) -> list:
        return self._load()

    def most_vulnerable_device(self) -> str:
        alerts = self._load()
        if not alerts:
            return "N/A"
        counts = {}
        for a in alerts:
            counts[a["device"]] = counts.get(a["device"], 0) + 1
        return max(counts, key=counts.get)

    def active_threat_count(self) -> int:
        # "Active" = HIGH or MEDIUM risk alerts in the most recent 20 events.
        recent = self._load()[-20:]
        return sum(1 for a in recent if a["risk_level"] in ("HIGH", "MEDIUM"))

    def current_security_score(self) -> int:
        """Simple, transparent formula: 100 minus a penalty per recent
        HIGH/MEDIUM alert (capped). This is a project-level demo metric,
        not a certified security score — computed from actual alert
        history, never hard-coded."""
        recent = self._load()[-20:]
        penalty = sum(15 if a["risk_level"] == "HIGH" else 5 for a in recent)
        return max(0, 100 - min(penalty, 100))


if __name__ == "__main__":
    engine = DecisionEngine()
    manager = AlertManager()
    decision = engine.decide(confidence=0.94, risk_level="HIGH")
    if decision["action"] != "MONITOR":
        alert = manager.raise_alert(
            device="CCTV Camera", threat="DDoS", confidence=0.94,
            risk_score=0.91, risk_level="HIGH", trend="INCREASING",
            recommended_action=decision["recommended_action"],
        )
        print(alert)
    print("Security score:", manager.current_security_score())
