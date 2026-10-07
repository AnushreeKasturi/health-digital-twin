"""Twin synchronisation: keep the virtual patient aligned with incoming real-world data.

* ``update_baselines`` - exponentially weighted estimates of the patient's
  resting heart rate, SpO2, blood pressure and fasting glucose from wearable data.
* ``calibrate_glucose`` - fits the minimal-model insulin sensitivity (SI) and
  basal glucose (Gb) to observed CGM readings by least squares, so the twin's
  simulated glucose curve matches *this* patient.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .patient import Patient
from .physiology import DEFAULT_MEALS, ExerciseBout, GlucoseParams, Meal, simulate_glucose_day


def update_baselines(patient: Patient, stream: pd.DataFrame, alpha: float = 0.3) -> tuple[Patient, dict]:
    """Blend the profile with what the sensors observed; returns the synced patient + observed values."""
    s = stream[stream["event"] == ""] if "event" in stream else stream
    hour = s["timestamp"].dt.hour
    resting = s[(s["steps"] == 0) & (hour >= 5) & (hour < 7)]
    fasting = s[(hour >= 6) & (hour < 7)]["glucose"].dropna()
    observed = {
        "resting_hr": float(resting["heart_rate"].median()) if len(resting) else patient.resting_hr,
        "systolic_bp": float(s["systolic_bp"].dropna().mean()) if s["systolic_bp"].notna().any() else patient.systolic_bp,
        "diastolic_bp": float(s["diastolic_bp"].dropna().mean()) if s["diastolic_bp"].notna().any() else patient.diastolic_bp,
        "fasting_glucose": float(fasting.median()) if len(fasting) else patient.fasting_glucose,
        "mean_spo2": float(s["spo2"].mean()),
        "daily_steps": float(s["steps"].sum() / max(len(s) / 1440, 1)),
    }
    synced = replace(
        patient,
        resting_hr=(1 - alpha) * patient.resting_hr + alpha * observed["resting_hr"],
        systolic_bp=(1 - alpha) * patient.systolic_bp + alpha * observed["systolic_bp"],
        diastolic_bp=(1 - alpha) * patient.diastolic_bp + alpha * observed["diastolic_bp"],
        fasting_glucose=(1 - alpha) * patient.fasting_glucose + alpha * observed["fasting_glucose"],
    )
    return synced, observed


def observe_cgm_day(patient: Patient, si_factor: float = 0.7, gb_offset: float = 6.0, seed: int = 3):
    """Stand-in for a real CGM upload: the 'real' patient has physiology that differs from the
    population prior (``si_factor``, ``gb_offset``). Returns (cgm readings, logged meals, true params)."""
    rng = np.random.default_rng(seed)
    meals = [Meal(m.minute + int(rng.integers(-40, 40)), round(m.carbs_g * rng.uniform(0.7, 1.3))) for m in DEFAULT_MEALS]
    prior = GlucoseParams.from_patient(patient)
    true = replace(prior, si=prior.si * si_factor, gb=prior.gb + gb_offset)
    glucose = simulate_glucose_day(true, meals)["glucose"] + rng.normal(0, 5, 1440)
    cgm = np.where(np.arange(1440) % 5 == 0, glucose, np.nan)
    return cgm, meals, true


def calibrate_glucose(
    patient: Patient,
    cgm: np.ndarray,
    meals: list[Meal] | None = None,
    exercise: list[ExerciseBout] | None = None,
) -> tuple[GlucoseParams, dict]:
    """Fit (SI, Gb) to one day of CGM readings (1440-length array, NaN where missing)."""
    prior = GlucoseParams.from_patient(patient)
    meals = DEFAULT_MEALS if meals is None else meals
    mask = ~np.isnan(cgm)

    def loss(theta):
        si, gb = np.exp(theta[0]), theta[1]
        sim = simulate_glucose_day(replace(prior, si=si, gb=gb), meals, exercise)["glucose"]
        # Weak prior keeps the fit physiologically plausible with sparse data.
        return np.mean((sim[mask] - cgm[mask]) ** 2) + 2.0 * (theta[0] - np.log(prior.si)) ** 2

    x0 = np.array([np.log(prior.si), prior.gb])
    res = minimize(loss, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 0.5, "maxiter": 120})
    fitted = replace(prior, si=float(np.exp(res.x[0])), gb=float(res.x[1]))
    sim_prior = simulate_glucose_day(prior, meals, exercise)["glucose"]
    sim_fit = simulate_glucose_day(fitted, meals, exercise)["glucose"]
    rmse = lambda sim: float(np.sqrt(np.mean((sim[mask] - cgm[mask]) ** 2)))  # noqa: E731
    return fitted, {"rmse_prior": rmse(sim_prior), "rmse_fitted": rmse(sim_fit), "si_prior": prior.si, "gb_prior": prior.gb}
