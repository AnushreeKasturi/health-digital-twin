"""Synthetic data generators.

* ``generate_cohort`` - a population cohort with 10-year outcome labels used to
  train the risk models. Outcomes are sampled from logistic models whose
  coefficients are inspired by FINDRISC / ADA (diabetes) and Framingham
  (cardiovascular) risk factors, with non-linear terms so the ML model has to
  learn more than a linear score.
* ``generate_wearable_stream`` - minute-level wearable + 5-minute CGM data for a
  patient, with optional injected clinical events used to exercise the
  monitoring / anomaly-detection layer.

No real patient data is used anywhere in this repository.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .patient import Patient
from .physiology import DEFAULT_MEALS, ExerciseBout, GlucoseParams, simulate_glucose_day

FEATURES = [
    "age", "male", "bmi", "systolic_bp", "diastolic_bp", "resting_hr", "fasting_glucose",
    "hba1c", "ldl", "hdl", "smoker", "family_history_diabetes", "exercise_min_per_week",
    "sleep_hours", "sodium_g_per_day", "on_bp_meds",
]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_cohort(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 80, n)
    male = rng.integers(0, 2, n)
    exercise = np.clip(rng.gamma(2.0, 60, n), 0, 600)
    bmi = np.clip(rng.normal(26 + 0.04 * (age - 45) - 0.006 * (exercise - 120), 4.5, n), 16, 50)
    smoker = (rng.random(n) < 0.18 + 0.05 * male).astype(int)
    fam = (rng.random(n) < 0.3).astype(int)
    sleep = np.clip(rng.normal(6.9, 1.0, n), 3.5, 10)
    sodium = np.clip(rng.normal(3.6, 1.0, n), 1.0, 8.0)
    sbp = np.clip(rng.normal(108 + 0.45 * age + 0.9 * (bmi - 25) + 2.2 * (sodium - 3.5) + 3 * male, 13, n), 85, 210)
    on_bp_meds = ((sbp > 145) & (rng.random(n) < 0.45)).astype(int)
    sbp = sbp - on_bp_meds * rng.normal(10, 4, n)
    dbp = np.clip(0.55 * sbp + rng.normal(12, 6, n), 50, 130)
    rhr = np.clip(rng.normal(72 - 0.02 * (exercise - 120) + 0.4 * (bmi - 25) + 4 * smoker, 8, n), 45, 115)
    insulin_resist = 0.08 * (bmi - 25) + 0.025 * (age - 45) + 0.6 * fam - 0.003 * (exercise - 120) + rng.normal(0, 0.8, n)
    fasting = np.clip(92 + 9 * insulin_resist + rng.normal(0, 8, n), 65, 300)
    hba1c = np.clip(5.3 + 0.022 * (fasting - 95) + rng.normal(0, 0.25, n), 4.2, 13)
    ldl = np.clip(rng.normal(115 + 0.5 * (age - 45) + 1.2 * (bmi - 25), 28, n), 40, 260)
    hdl = np.clip(rng.normal(52 - 0.8 * (bmi - 25) - 6 * male + 0.015 * exercise, 11, n), 20, 110)

    # 10-year type 2 diabetes onset (logit), FINDRISC-like drivers + interactions.
    logit_dm = (
        -5.2
        + 0.035 * (age - 45)
        + 0.11 * (bmi - 25)
        + 0.045 * (fasting - 95)
        + 1.1 * np.clip(hba1c - 5.6, 0, None)
        + 0.7 * fam
        - 0.004 * (exercise - 120)
        + 0.25 * np.abs(sleep - 7.2)
        + 0.004 * (sbp - 120)
        + 0.6 * ((bmi > 30) & (fam == 1))
    )
    already_dm = (hba1c >= 6.5) | (fasting >= 126)
    p_dm = np.where(already_dm, np.clip(_sigmoid(logit_dm) + 0.6, 0, 0.98), _sigmoid(logit_dm))

    # 10-year cardiovascular event (logit), Framingham-like drivers.
    logit_cvd = (
        -4.6
        + 0.065 * (age - 50)
        + 0.45 * male
        + 0.022 * (sbp - 120)
        + 0.006 * (ldl - 120)
        - 0.02 * (hdl - 50)
        + 0.75 * smoker
        + 0.55 * already_dm
        + 0.18 * np.clip(hba1c - 5.7, 0, None)
        + 0.25 * on_bp_meds
        - 0.002 * (exercise - 120)
        + 0.012 * (rhr - 70)
        + 0.0004 * np.clip(sbp - 140, 0, None) ** 2
    )
    p_cvd = _sigmoid(logit_cvd)

    df = pd.DataFrame({
        "age": age, "male": male, "bmi": bmi, "systolic_bp": sbp, "diastolic_bp": dbp,
        "resting_hr": rhr, "fasting_glucose": fasting, "hba1c": hba1c, "ldl": ldl, "hdl": hdl,
        "smoker": smoker, "family_history_diabetes": fam, "exercise_min_per_week": exercise,
        "sleep_hours": sleep, "sodium_g_per_day": sodium, "on_bp_meds": on_bp_meds,
    })
    df["diabetes_10y"] = (rng.random(n) < p_dm).astype(int)
    df["cvd_10y"] = (rng.random(n) < p_cvd).astype(int)
    return df


EVENT_TYPES = ("tachycardia", "desaturation", "hyperglycemia", "hypoglycemia", "fever")


def generate_wearable_stream(
    patient: Patient,
    days: int = 2,
    events: list[tuple[str, float, float]] | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Minute-level wearable data.

    ``events`` is a list of ``(event_type, start_hour_from_stream_start, duration_hours)``.
    Columns: timestamp, heart_rate, spo2, skin_temp, steps, glucose (CGM, every 5 min,
    NaN otherwise), systolic_bp / diastolic_bp (cuff readings 3x a day), event (ground truth).
    """
    rng = np.random.default_rng(seed)
    n = days * 1440
    minute = np.arange(n)
    mod = minute % 1440
    hour = mod / 60

    asleep = (hour < 6.5) | (hour >= 23)
    activity = np.zeros(n)
    daily_ex = min(patient.exercise_min_per_week / 7, 120)
    for d in range(days):
        start = d * 1440 + 18 * 60
        activity[start:start + int(daily_ex)] = 0.6
        walk = rng.choice(np.arange(d * 1440 + 7 * 60, d * 1440 + 22 * 60), 12, replace=False)
        for w in walk:
            activity[w:w + 8] = np.maximum(activity[w:w + 8], 0.25)

    circadian = -6 * np.cos(2 * np.pi * (hour - 15) / 24)
    hr = patient.resting_hr + 4 + circadian * 0.5 - 7 * asleep + 55 * activity + rng.normal(0, 2.5, n)
    spo2 = np.clip(97.5 - 0.8 * asleep + rng.normal(0, 0.6, n), 88, 100)
    temp = 33.5 + 0.6 * np.sin(2 * np.pi * (hour - 4) / 24) + 0.4 * activity + rng.normal(0, 0.1, n)
    steps = np.where(activity > 0, rng.poisson(30 + 110 * activity), rng.poisson(0.3 * ~asleep))

    params = GlucoseParams.from_patient(patient)
    glucose = np.empty(n)
    for d in range(days):
        jitter = [type(m)(m.minute + int(rng.integers(-30, 30)), m.carbs_g * rng.uniform(0.8, 1.2)) for m in DEFAULT_MEALS]
        ex = [ExerciseBout(18 * 60, int(daily_ex), 0.6)] if daily_ex > 0 else []
        glucose[d * 1440:(d + 1) * 1440] = simulate_glucose_day(params, jitter, ex)["glucose"]
    glucose += rng.normal(0, 4, n)

    event = np.array([""] * n, dtype=object)
    for etype, start_h, dur_h in events or []:
        s, e = int(start_h * 60), min(int((start_h + dur_h) * 60), n)
        ramp = np.sin(np.linspace(0, np.pi, e - s))
        event[s:e] = etype
        if etype == "tachycardia":
            hr[s:e] += 55 * ramp
        elif etype == "desaturation":
            spo2[s:e] -= 9 * ramp
        elif etype == "hyperglycemia":
            glucose[s:e] += 170 * ramp
        elif etype == "hypoglycemia":
            glucose[s:e] = np.minimum(glucose[s:e], glucose[s:e] - (glucose[s:e] - 52) * ramp)
        elif etype == "fever":
            temp[s:e] += 2.2 * ramp
            hr[s:e] += 20 * ramp

    cgm = np.where(minute % 5 == 0, glucose, np.nan)
    sbp = np.full(n, np.nan)
    dbp = np.full(n, np.nan)
    cuff = np.isin(mod, [8 * 60, 14 * 60, 21 * 60])
    sbp[cuff] = patient.systolic_bp + rng.normal(0, 6, cuff.sum())
    dbp[cuff] = patient.diastolic_bp + rng.normal(0, 4, cuff.sum())

    return pd.DataFrame({
        "timestamp": pd.Timestamp("2026-01-05") + pd.to_timedelta(minute, unit="min"),
        "heart_rate": hr.round(1),
        "spo2": np.clip(spo2, 70, 100).round(1),
        "skin_temp": temp.round(2),
        "steps": steps,
        "glucose": np.round(cgm, 1),
        "systolic_bp": np.round(sbp),
        "diastolic_bp": np.round(dbp),
        "event": event,
    })


DEFAULT_EVENTS = [("tachycardia", 10.5, 0.75), ("hyperglycemia", 21.0, 3.0), ("desaturation", 27.0, 0.6), ("hypoglycemia", 39.0, 1.0)]
