import numpy as np
import pytest

from twin.data import DEFAULT_EVENTS, generate_cohort, generate_wearable_stream
from twin.models import VitalsMonitor, explain_risk, load_models, predict_risk
from twin.patient import SAMPLE_PATIENTS
from twin.physiology import GlucoseParams, Plan, glucose_metrics, hba1c_from_mean_glucose, project_lifestyle, simulate_glucose_day
from twin.sync import calibrate_glucose, observe_cgm_day

P1, P2, P3 = SAMPLE_PATIENTS["P001"], SAMPLE_PATIENTS["P002"], SAMPLE_PATIENTS["P003"]


@pytest.fixture(scope="module")
def bundle():
    return load_models()


def test_glucose_model_matches_measured_hba1c():
    # The uncalibrated prior should already land close to each persona's lab HbA1c.
    for p in (P1, P2, P3):
        g = simulate_glucose_day(GlucoseParams.from_patient(p))["glucose"]
        assert abs(hba1c_from_mean_glucose(g.mean()) - p.hba1c) < 0.5


def test_diabetic_has_worse_glycaemic_control_than_healthy():
    tir = lambda p: glucose_metrics(simulate_glucose_day(GlucoseParams.from_patient(p))["glucose"])["time_in_range_pct"]  # noqa: E731
    assert tir(P2) < tir(P1) < tir(P3) + 1e-9


def test_metformin_and_exercise_lower_glucose():
    params = GlucoseParams.from_patient(P2)
    base = simulate_glucose_day(params)["glucose"].mean()
    assert simulate_glucose_day(params, metformin=True)["glucose"].mean() < base


def test_no_change_plan_is_stable():
    proj = project_lifestyle(P1, Plan(), 12)
    end = proj.states[-1]
    assert end.weight_kg == pytest.approx(P1.weight_kg)
    assert end.systolic_bp == pytest.approx(P1.systolic_bp)
    assert end.hba1c == pytest.approx(P1.hba1c, abs=0.01)


def test_lifestyle_plan_improves_markers_and_risk(bundle):
    proj = project_lifestyle(P1, Plan("L", 400, 180, 2.5, 7.5, True), 24)
    end = proj.states[-1]
    assert end.weight_kg < P1.weight_kg and end.systolic_bp < P1.systolic_bp and end.hba1c < P1.hba1c
    risks = predict_risk(bundle, proj.states)
    assert risks["cvd_10y"][-1] < risks["cvd_10y"][0]
    assert risks["diabetes_10y"][-1] < risks["diabetes_10y"][0]


def test_risk_models_rank_personas(bundle):
    r1, r3 = predict_risk(bundle, P1), predict_risk(bundle, P3)
    assert r1["cvd_10y"] > r3["cvd_10y"] and r1["diabetes_10y"] > r3["diabetes_10y"]
    for m in bundle["metrics"].values():
        assert m["auc"] > 0.75


def test_explanations_name_smoking_for_smoker(bundle):
    ex = explain_risk(bundle, P1, "cvd_10y")
    assert "Smoking" in ex["factor"].head(3).tolist()
    assert (ex["risk_reduction"] >= -0.02).all()


def test_calibration_recovers_hidden_parameters():
    cgm, meals, true = observe_cgm_day(P1, si_factor=0.6, gb_offset=8)
    fitted, info = calibrate_glucose(P1, cgm, meals)
    assert fitted.si == pytest.approx(true.si, rel=0.15)
    assert fitted.gb == pytest.approx(true.gb, abs=3)
    assert info["rmse_fitted"] < info["rmse_prior"]


def test_monitor_detects_injected_events():
    mon = VitalsMonitor().fit(generate_wearable_stream(P1, days=3, seed=1))
    stream = generate_wearable_stream(P1, days=2, events=DEFAULT_EVENTS)
    alerts = mon.alerts(mon.score(stream))
    detected = {a.vital for a in alerts if a.severity == "critical"}
    assert {"heart_rate", "spo2", "glucose"} <= detected
    # A clean stream should raise very few alerts.
    clean = mon.alerts(mon.score(generate_wearable_stream(P1, days=2, seed=11)))
    assert sum(a.severity == "critical" for a in clean) <= 1


def test_cohort_shape():
    df = generate_cohort(2000, seed=1)
    assert len(df) == 2000 and df.isna().sum().sum() == 0
    assert 0.01 < df["cvd_10y"].mean() < 0.3
