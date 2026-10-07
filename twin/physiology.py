"""Mechanistic physiology models that form the 'physics' layer of the twin.

Two models are provided:

* ``simulate_glucose_day`` - a 24 h glucose-insulin simulation based on the
  Bergman minimal model with a two-compartment gut absorption term for meals,
  exercise-driven glucose uptake and a metformin effect.
* ``project_lifestyle`` - a week-by-week projection of weight, blood pressure,
  resting heart rate, LDL and HbA1c under a lifestyle / medication plan. HbA1c
  is driven by the mean glucose of the minimal model, so both models are
  coupled.

Effect sizes are simplified approximations drawn from published meta-analyses
(see README, "Model assumptions"). They are suitable for a proof-of-concept and
are NOT validated for clinical decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .patient import Patient

MINUTES_PER_DAY = 1440


@dataclass
class Meal:
    minute: int  # minute of the day the meal starts (0-1439)
    carbs_g: float


@dataclass
class ExerciseBout:
    minute: int
    duration_min: int
    intensity: float = 0.5  # 0 (very light) .. 1 (vigorous)


DEFAULT_MEALS = [Meal(8 * 60, 50), Meal(13 * 60, 70), Meal(20 * 60, 65)]


@dataclass
class GlucoseParams:
    """Patient-specific minimal-model parameters (the calibrated 'twin state')."""

    gb: float  # basal (fasting) glucose, mg/dL
    si: float  # insulin sensitivity, (mL/uU)/min
    beta: float  # beta-cell secretion gain, uU/mL per mg/dL per min
    p1: float = 0.025  # glucose effectiveness, 1/min
    p2: float = 0.025  # remote insulin decay, 1/min
    n: float = 0.12  # insulin clearance, 1/min
    ib: float = 10.0  # basal insulin, uU/mL
    vg_dl_per_kg: float = 1.7  # glucose distribution volume
    weight_kg: float = 75.0

    @classmethod
    def from_patient(cls, p: Patient) -> "GlucoseParams":
        # Insulin sensitivity falls with HbA1c and BMI (insulin resistance).
        si = 6e-4 * np.exp(-0.55 * max(p.hba1c - 5.0, 0)) * np.exp(-0.03 * max(p.bmi - 22, 0))
        # Beta-cell function declines as glycaemia worsens.
        beta = 0.09 * np.exp(-0.35 * max(p.hba1c - 5.4, 0))
        return cls(gb=p.fasting_glucose, si=float(si), beta=float(beta), weight_kg=p.weight_kg)


def _meal_appearance(meals: list[Meal], weight_kg: float, vg: float) -> np.ndarray:
    """Rate of glucose appearance from meals, mg/dL/min, per minute of the day."""
    t = np.arange(MINUTES_PER_DAY, dtype=float)
    ra = np.zeros(MINUTES_PER_DAY)
    k = 0.022  # gut absorption rate, 1/min (peak ~45 min)
    f = 0.9  # bioavailability
    volume_dl = vg * weight_kg
    for meal in meals:
        dt = t - meal.minute
        mask = dt >= 0
        ra[mask] += f * meal.carbs_g * 1000 * k**2 * dt[mask] * np.exp(-k * dt[mask]) / volume_dl
    return ra


def simulate_glucose_day(
    params: GlucoseParams,
    meals: list[Meal] | None = None,
    exercise: list[ExerciseBout] | None = None,
    metformin: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate one day at 1-minute resolution. Returns glucose, insulin and time arrays."""
    meals = DEFAULT_MEALS if meals is None else meals
    exercise = exercise or []

    si, gb = params.si, params.gb
    if metformin:  # improves insulin sensitivity and suppresses hepatic output
        si *= 1.25
        gb = 100 + (gb - 100) * 0.8 if gb > 100 else gb

    p3 = si * params.p2
    ra = _meal_appearance(meals, params.weight_kg, params.vg_dl_per_kg)

    uptake = np.zeros(MINUTES_PER_DAY)  # extra insulin-independent uptake from exercise
    for bout in exercise:
        end = min(bout.minute + bout.duration_min, MINUTES_PER_DAY)
        uptake[bout.minute:end] += 0.012 * bout.intensity

    g = np.empty(MINUTES_PER_DAY)
    ins = np.empty(MINUTES_PER_DAY)
    G, X, I = gb, 0.0, params.ib
    for i in range(MINUTES_PER_DAY):
        dG = -(params.p1 + X + uptake[i]) * G + params.p1 * gb + uptake[i] * 0.6 * gb + ra[i]
        dX = -params.p2 * X + p3 * (I - params.ib)
        dI = -params.n * (I - params.ib) + params.beta * max(G - gb, 0.0)
        G = max(G + dG, 40.0)
        X = X + dX
        I = max(I + dI, 0.0)
        g[i], ins[i] = G, I
    return {"minute": np.arange(MINUTES_PER_DAY), "glucose": g, "insulin": ins}


def glucose_metrics(glucose: np.ndarray) -> dict[str, float]:
    """Standard CGM summary metrics (international consensus ranges)."""
    return {
        "mean": float(glucose.mean()),
        "peak": float(glucose.max()),
        "time_in_range_pct": float(np.mean((glucose >= 70) & (glucose <= 180)) * 100),
        "time_above_pct": float(np.mean(glucose > 180) * 100),
        "time_below_pct": float(np.mean(glucose < 70) * 100),
        "gmi": float(3.31 + 0.02392 * glucose.mean()),  # glucose management indicator (%)
    }


def hba1c_from_mean_glucose(mean_glucose: float) -> float:
    """ADAG equation: eAG = 28.7 * A1c - 46.7."""
    return (mean_glucose + 46.7) / 28.7


@dataclass
class Plan:
    """A lifestyle / treatment scenario to test on the twin."""

    name: str = "Current lifestyle"
    calorie_deficit_kcal: float = 0.0  # daily
    exercise_min_per_week: float | None = None  # None = keep current
    sodium_g_per_day: float | None = None
    sleep_hours: float | None = None
    quit_smoking: bool = False
    antihypertensive: bool = False
    metformin: bool = False
    statin: bool = False
    adherence: float = 1.0  # 0..1, scales all effects


@dataclass
class Projection:
    weeks: np.ndarray
    states: list[Patient] = field(default_factory=list)


def project_lifestyle(patient: Patient, plan: Plan, weeks: int = 24) -> Projection:
    """Project the patient's state week by week under ``plan``."""
    a = float(np.clip(plan.adherence, 0, 1))
    ex_target = patient.exercise_min_per_week if plan.exercise_min_per_week is None else plan.exercise_min_per_week
    na_target = patient.sodium_g_per_day if plan.sodium_g_per_day is None else plan.sodium_g_per_day
    sleep_target = patient.sleep_hours if plan.sleep_hours is None else plan.sleep_hours
    d_ex = (ex_target - patient.exercise_min_per_week) * a
    d_na = (na_target - patient.sodium_g_per_day) * a
    d_sleep = (sleep_target - patient.sleep_hours) * a

    base_params = GlucoseParams.from_patient(patient)
    full = (d_ex, d_na, d_sleep)

    states = [patient]
    weight = patient.weight_kg
    hba1c = patient.hba1c
    for w in range(1, weeks + 1):
        # New habits are adopted gradually (~3-week time constant), not overnight.
        adopt = 1 - np.exp(-w / 3)
        d_ex, d_na, d_sleep = (x * adopt for x in full)
        weekly_ex = patient.exercise_min_per_week + d_ex
        sessions = [ExerciseBout(18 * 60, int(min(weekly_ex / 7, 120)), 0.6)] if weekly_ex > 0 else []
        # Weight: 7700 kcal ~ 1 kg, with metabolic adaptation over ~6 months,
        # plus a small exercise-driven energy expenditure term.
        deficit = plan.calorie_deficit_kcal * a + d_ex / 7 * 5
        weight -= deficit * 7 / 7700 * np.exp(-w / 30)
        weight_lost = patient.weight_kg - weight

        ramp4 = 1 - np.exp(-w / 4)  # BP / medication response time constant
        ramp8 = 1 - np.exp(-w / 8)  # training adaptation time constant

        sbp_drop = (
            1.0 * weight_lost
            + 2.5 * (-d_na) * ramp4  # ~2.5 mmHg per g/day sodium reduction
            + 5.0 * np.clip(d_ex / 150, -1, 1.5) * ramp8
            + 10.0 * plan.antihypertensive * a * ramp4
            + 1.5 * np.clip(d_sleep, -2, 2) * ramp8
        )
        sbp = patient.systolic_bp - sbp_drop
        dbp = patient.diastolic_bp - 0.6 * sbp_drop
        rhr = patient.resting_hr - 5.0 * np.clip(d_ex / 150, -1, 1.5) * ramp8 - 2 * plan.quit_smoking * a * ramp8
        ldl = patient.ldl * (1 - 0.35 * plan.statin * a * ramp4) - 0.8 * weight_lost
        hdl = patient.hdl + 0.35 * weight_lost + 2.0 * np.clip(d_ex / 150, 0, 1.5) * ramp8

        # Insulin sensitivity improves with weight loss, exercise and sleep.
        si_gain = (1 + 0.04 * weight_lost) * (1 + 0.20 * np.clip(d_ex / 150, -0.5, 1.5) * ramp8) * (
            1 + 0.04 * np.clip(d_sleep, -2, 2)
        )
        params = replace(base_params, si=base_params.si * si_gain, weight_kg=weight)
        fasting = patient.fasting_glucose - 0.9 * weight_lost - 6 * np.clip(d_ex / 150, 0, 1.5) * ramp8
        params.gb = max(fasting, 80)
        day = simulate_glucose_day(params, exercise=sessions, metformin=plan.metformin and a > 0)
        target_a1c = hba1c_from_mean_glucose(day["glucose"].mean())
        # Calibrate so that at week 0 the model reproduces the measured HbA1c.
        target_a1c += patient.hba1c - _baseline_a1c(patient)
        hba1c += (target_a1c - hba1c) * (1 - np.exp(-1 / 8))  # RBC turnover lag (~8 weeks)

        states.append(
            replace(
                patient,
                weight_kg=float(weight),
                systolic_bp=float(sbp),
                diastolic_bp=float(dbp),
                resting_hr=float(rhr),
                ldl=float(ldl),
                hdl=float(hdl),
                fasting_glucose=float(params.gb if not plan.metformin else params.gb * 0.92),
                hba1c=float(hba1c),
                exercise_min_per_week=float(weekly_ex),
                sodium_g_per_day=float(patient.sodium_g_per_day + d_na),
                sleep_hours=float(patient.sleep_hours + d_sleep),
                smoker=patient.smoker,
                smoking_cessation=float(patient.smoker and plan.quit_smoking) * a * (1 - np.exp(-w / 12)),
                on_bp_meds=patient.on_bp_meds or plan.antihypertensive,
            )
        )
    return Projection(weeks=np.arange(weeks + 1), states=states)


_A1C_CACHE: dict[tuple, float] = {}


def _baseline_a1c(patient: Patient) -> float:
    key = (round(patient.fasting_glucose, 1), round(patient.hba1c, 2), round(patient.bmi, 1), patient.exercise_min_per_week)
    if key not in _A1C_CACHE:
        params = GlucoseParams.from_patient(patient)
        weekly_ex = patient.exercise_min_per_week
        sessions = [ExerciseBout(18 * 60, int(min(weekly_ex / 7, 120)), 0.6)] if weekly_ex > 0 else []
        day = simulate_glucose_day(params, exercise=sessions)
        _A1C_CACHE[key] = hba1c_from_mean_glucose(day["glucose"].mean())
    return _A1C_CACHE[key]
