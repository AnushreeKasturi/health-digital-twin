"""AI/ML layer of the twin.

* Risk models: calibrated gradient-boosted trees (scikit-learn
  ``HistGradientBoostingClassifier`` + isotonic ``CalibratedClassifierCV``)
  predicting 10-year type 2 diabetes and cardiovascular-event risk.
* Explanations: per-patient counterfactual attributions - each modifiable risk
  factor is moved to a healthy reference value and the change in predicted risk
  is reported, which is what a clinician / patient actually wants to know.
* Monitoring: a personalised anomaly detector combining an ``IsolationForest``
  fitted on the patient's own baseline window with z-scores against the twin's
  learned baseline and hard clinical thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import FEATURES, generate_cohort
from .patient import Patient

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "risk_models.joblib"
TARGETS = {"diabetes_10y": "Type 2 diabetes (10-yr)", "cvd_10y": "Cardiovascular event (10-yr)"}

# Healthy reference values used for counterfactual explanations (modifiable factors only).
HEALTHY_REFERENCE = {
    "bmi": ("BMI", 23.0),
    "systolic_bp": ("Systolic BP", 120.0),
    "fasting_glucose": ("Fasting glucose", 90.0),
    "hba1c": ("HbA1c", 5.4),
    "ldl": ("LDL cholesterol", 100.0),
    "hdl": ("HDL cholesterol", 55.0),
    "smoker": ("Smoking", 0.0),
    "exercise_min_per_week": ("Physical activity", 150.0),
    "sleep_hours": ("Sleep", 7.5),
    "resting_hr": ("Resting heart rate", 65.0),
    "sodium_g_per_day": ("Sodium intake", 2.0),
}


def train_models(n: int = 20000, seed: int = 42) -> dict:
    df = generate_cohort(n, seed)
    bundle = {"features": FEATURES, "models": {}, "metrics": {}}
    for target in TARGETS:
        X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[target], test_size=0.2, random_state=seed, stratify=df[target])
        base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
        model = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        bundle["models"][target] = model
        bundle["metrics"][target] = {
            "auc": float(roc_auc_score(y_te, p)),
            "brier": float(brier_score_loss(y_te, p)),
            "prevalence": float(y_te.mean()),
            "n_train": len(X_tr),
            "n_test": len(X_te),
        }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_models() -> dict:
    """Load the trained bundle, training it on first run (~30 s) or if it was built with another sklearn version."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return train_models()


def _frame(patients: list[Patient]) -> pd.DataFrame:
    return pd.DataFrame([p.features() for p in patients])[FEATURES]


def predict_risk(bundle: dict, patients: Patient | list[Patient]) -> dict[str, np.ndarray] | dict[str, float]:
    single = isinstance(patients, Patient)
    plist = [patients] if single else patients
    X = _frame(plist)
    quit_progress = np.array([p.smoking_cessation if p.smoker else 0.0 for p in plist])
    out = {t: bundle["models"][t].predict_proba(X)[:, 1] for t in TARGETS}
    if quit_progress.any():
        X_quit = X.assign(smoker=0.0)
        for t in TARGETS:
            out[t] = (1 - quit_progress) * out[t] + quit_progress * bundle["models"][t].predict_proba(X_quit)[:, 1]
    return {t: float(v[0]) for t, v in out.items()} if single else out


def explain_risk(bundle: dict, patient: Patient, target: str) -> pd.DataFrame:
    """Counterfactual attribution: risk reduction if one factor were at its healthy value."""
    base = predict_risk(bundle, patient)[target]
    rows, variants = [], []
    feats = patient.features()
    for key, (label, ref) in HEALTHY_REFERENCE.items():
        current = feats[key]
        worse = current > ref if key not in ("hdl", "exercise_min_per_week", "sleep_hours") else current < ref
        if key == "sleep_hours":
            worse = abs(current - ref) > 0.5
        if not worse:
            continue
        if key == "bmi":
            variant = replace(patient, weight_kg=ref * (patient.height_cm / 100) ** 2)
        elif key == "smoker":
            variant = replace(patient, smoker=False)
        else:
            variant = replace(patient, **{key: ref})
        rows.append((label, current, ref))
        variants.append(variant)
    if not variants:
        return pd.DataFrame(columns=["factor", "current", "target", "risk_reduction"])
    risks = predict_risk(bundle, variants)[target]
    df = pd.DataFrame(rows, columns=["factor", "current", "target"])
    df["risk_reduction"] = base - risks
    return df.sort_values("risk_reduction", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- monitoring

VITALS = ["heart_rate", "spo2", "skin_temp", "glucose"]

# Hard clinical limits (wearable context): (low, high)
CLINICAL_LIMITS = {
    "heart_rate": (40, 130),
    "spo2": (92, None),
    "skin_temp": (None, 36.0),
    "glucose": (70, 250),
}


@dataclass
class Alert:
    timestamp: pd.Timestamp
    vital: str
    value: float
    severity: str  # "warning" | "critical"
    reason: str


class VitalsMonitor:
    """Personalised anomaly detection learned from a baseline window of the patient's own data."""

    def __init__(self, contamination: float = 0.01, seed: int = 0):
        self.forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
        self.baseline: dict[str, tuple[float, float]] = {}

    @staticmethod
    def _windowed(df: pd.DataFrame) -> pd.DataFrame:
        w = df.set_index("timestamp")[["heart_rate", "spo2", "skin_temp", "glucose", "steps"]]
        w = w.resample("5min").agg({"heart_rate": "mean", "spo2": "min", "skin_temp": "mean", "glucose": "mean", "steps": "sum"})
        return w.interpolate(limit_direction="both")

    def fit(self, baseline_df: pd.DataFrame) -> "VitalsMonitor":
        w = self._windowed(baseline_df)
        self.forest.fit(w.values)
        for v in VITALS:
            # Activity-adjusted baseline for heart rate so exercise is not flagged.
            col = w[v] if v != "heart_rate" else w.loc[w["steps"] < 200, v]
            self.baseline[v] = (float(col.median()), float(col.std() + 1e-6))
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self._windowed(df)
        w["anomaly_score"] = -self.forest.score_samples(w.values)
        w["is_anomaly"] = self.forest.predict(w[["heart_rate", "spo2", "skin_temp", "glucose", "steps"]].values) == -1
        for v in VITALS:
            mu, sd = self.baseline[v]
            w[f"z_{v}"] = (w[v] - mu) / sd
        return w

    def alerts(self, scored: pd.DataFrame) -> list[Alert]:
        out: list[Alert] = []
        last: dict[str, pd.Timestamp] = {}
        for ts, row in scored.iterrows():
            for v in VITALS:
                lo, hi = CLINICAL_LIMITS[v]
                val = row[v]
                severity = reason = None
                if lo is not None and val < lo:
                    severity, reason = "critical", f"below clinical limit of {lo}"
                elif hi is not None and val > hi:
                    severity, reason = "critical", f"above clinical limit of {hi}"
                elif row["is_anomaly"] and abs(row[f"z_{v}"]) > 3 and not (v == "heart_rate" and row["steps"] > 200):
                    severity, reason = "warning", f"{row[f'z_{v}']:+.1f} SD from personal baseline (multivariate anomaly)"
                if severity and (v not in last or ts - last[v] > pd.Timedelta("30min")):
                    out.append(Alert(ts, v, float(val), severity, reason))
                    last[v] = ts
        return out
