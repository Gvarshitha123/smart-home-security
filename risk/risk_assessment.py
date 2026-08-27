"""
risk/risk_assessment.py — STAGE 7

Purpose
-------
Answer "how serious is this threat?" — separate from the detector's binary
"is this suspicious?" answer. Produces a project-level risk score in
[0.0, 1.0] and a LOW/MEDIUM/HIGH label.

This is explicitly a PROJECT-LEVEL cybersecurity risk score, not an
officially certified risk measurement (e.g. not CVSS).

Score composition (documented, not hidden):
  risk = w1 * model_confidence
       + w2 * attack_severity_weight(predicted_class)
       + w3 * frequency_factor(recent_alert_count)
       + w4 * progression_factor(threat_trend)

Weights are configurable and sum to 1.0.

Required packages
------------------
pip install numpy

How to run
----------
from risk.risk_assessment import RiskAssessor
assessor = RiskAssessor()
result = assessor.assess(confidence=0.94, predicted_class="DDoS",
                          recent_alert_count=3, threat_trend="INCREASING")

Expected output
---------------
{'risk_score': 0.83, 'risk_level': 'HIGH', 'breakdown': {...}}
"""

from dataclasses import dataclass, field

import config

# Severity weighting per attack category is a PROPOSED, configurable
# judgment call for this project — not a universal security standard.
# Update ATTACK_SEVERITY once the real dataset's class names are known
# (from dataset_inspection.py), and adjust with your team/mentor's input.
DEFAULT_ATTACK_SEVERITY = {
    "normal": 0.0,
    "benign": 0.0,
    "scan": 0.4,
    "reconnaissance": 0.4,
    "brute force": 0.6,
    "bruteforce": 0.6,
    "dos": 0.8,
    "ddos": 0.9,
    "mirai": 0.9,
    "backdoor": 0.95,
    "web attack": 0.7,
}

TREND_FACTOR = {
    "STABLE": 0.0,
    "INCREASING": 0.5,
    "CRITICAL": 1.0,
}


@dataclass
class RiskWeights:
    confidence: float = 0.35
    severity: float = 0.35
    frequency: float = 0.15
    progression: float = 0.15

    def __post_init__(self):
        total = self.confidence + self.severity + self.frequency + self.progression
        assert abs(total - 1.0) < 1e-6, f"RiskWeights must sum to 1.0, got {total}"


class RiskAssessor:
    def __init__(self, weights: RiskWeights = None,
                 attack_severity: dict = None,
                 low_threshold: float = config.RISK_LOW_THRESHOLD,
                 medium_threshold: float = config.RISK_MEDIUM_THRESHOLD):
        self.weights = weights or RiskWeights()
        self.attack_severity = attack_severity or DEFAULT_ATTACK_SEVERITY
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold

    def _severity_for(self, predicted_class: str) -> float:
        return self.attack_severity.get(str(predicted_class).strip().lower(), 0.5)

    def _frequency_factor(self, recent_alert_count: int, cap: int = 10) -> float:
        return min(recent_alert_count, cap) / cap

    def assess(self, confidence: float, predicted_class: str,
               recent_alert_count: int = 0, threat_trend: str = "STABLE") -> dict:
        confidence = max(0.0, min(1.0, confidence))
        severity = self._severity_for(predicted_class)
        frequency = self._frequency_factor(recent_alert_count)
        progression = TREND_FACTOR.get(threat_trend, 0.0)

        score = (
            self.weights.confidence * confidence
            + self.weights.severity * severity
            + self.weights.frequency * frequency
            + self.weights.progression * progression
        )
        score = round(min(max(score, 0.0), 1.0), 4)

        if score < self.low_threshold:
            level = "LOW"
        elif score < self.medium_threshold:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return {
            "risk_score": score,
            "risk_level": level,
            "breakdown": {
                "confidence_component": round(self.weights.confidence * confidence, 4),
                "severity_component": round(self.weights.severity * severity, 4),
                "frequency_component": round(self.weights.frequency * frequency, 4),
                "progression_component": round(self.weights.progression * progression, 4),
            },
            "disclaimer": "Project-level cybersecurity risk score for academic "
                           "demonstration — not a certified risk measurement.",
        }


if __name__ == "__main__":
    assessor = RiskAssessor()
    demo = assessor.assess(confidence=0.94, predicted_class="DDoS",
                            recent_alert_count=3, threat_trend="INCREASING")
    print(demo)
