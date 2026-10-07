"""VitalTwin - Streamlit dashboard for the chronic-patient digital twin.

Run:  streamlit run app.py
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from twin.data import DEFAULT_EVENTS, generate_wearable_stream
from twin.models import TARGETS, VitalsMonitor, explain_risk, load_models, predict_risk
from twin.patient import SAMPLE_PATIENTS, Patient
from twin.physiology import (
    DEFAULT_MEALS, ExerciseBout, GlucoseParams, Meal, Plan, glucose_metrics, project_lifestyle, simulate_glucose_day,
)
from twin.sync import calibrate_glucose, observe_cgm_day, update_baselines

# Categorical slots (fixed order) and reserved status colours.
C_BASE, C_SCEN, C_ALT = "#2a78d6", "#eb6834", "#1baf7a"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
VITAL_LABELS = {"heart_rate": "Heart rate (bpm)", "spo2": "SpO₂ (%)", "skin_temp": "Skin temp (°C)", "glucose": "CGM glucose (mg/dL)"}

st.set_page_config(page_title="VitalTwin - Patient Digital Twin", page_icon="🫀", layout="wide")


def style(fig: go.Figure, height: int = 320, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=70 if title else 40, b=10), title=dict(text=title, y=0.98, yanchor="top") if title else None,
        hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=0.5)
    fig.update_traces(cliponaxis=False, selector=dict(type="scatter"))
    return fig


def risk_band(p: float) -> tuple[str, str]:
    if p < 0.05:
        return "Low", "good"
    if p < 0.10:
        return "Borderline", "warning"
    if p < 0.20:
        return "Elevated", "serious"
    return "High", "critical"


@st.cache_resource
def models():
    return load_models()


@st.cache_data
def stream_for(patient: Patient, with_events: bool):
    return generate_wearable_stream(patient, days=2, events=DEFAULT_EVENTS if with_events else None)


@st.cache_resource
def monitor_for(patient: Patient):
    return VitalsMonitor().fit(generate_wearable_stream(patient, days=3, seed=1))


@st.cache_data
def projection(patient: Patient, plan: Plan, weeks: int):
    proj = project_lifestyle(patient, plan, weeks)
    risks = predict_risk(models(), proj.states)
    df = pd.DataFrame([{**s.features(), "weight_kg": s.weight_kg} for s in proj.states])
    df["week"] = proj.weeks
    for t in TARGETS:
        df[t] = risks[t]
    return df


@st.cache_data
def calibration(patient: Patient):
    cgm, meals, true = observe_cgm_day(patient)
    fitted, info = calibrate_glucose(patient, cgm, meals)
    return cgm, meals, true, fitted, info


bundle = models()

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🫀 VitalTwin")
st.sidebar.caption("Digital twin for chronic-disease patients · proof of concept")
pid = st.sidebar.selectbox("Patient", list(SAMPLE_PATIENTS), format_func=lambda k: SAMPLE_PATIENTS[k].name)
base = SAMPLE_PATIENTS[pid]

with st.sidebar.expander("Edit clinical profile"):
    c1, c2 = st.columns(2)
    base = replace(
        base,
        age=c1.number_input("Age", 18, 95, base.age),
        weight_kg=c2.number_input("Weight (kg)", 35.0, 200.0, float(base.weight_kg), 0.5),
        systolic_bp=c1.number_input("Systolic BP", 80.0, 220.0, float(base.systolic_bp)),
        diastolic_bp=c2.number_input("Diastolic BP", 40.0, 140.0, float(base.diastolic_bp)),
        fasting_glucose=c1.number_input("Fasting glucose", 60.0, 350.0, float(base.fasting_glucose)),
        hba1c=c2.number_input("HbA1c (%)", 4.0, 14.0, float(base.hba1c), 0.1),
        ldl=c1.number_input("LDL", 30.0, 300.0, float(base.ldl)),
        hdl=c2.number_input("HDL", 15.0, 120.0, float(base.hdl)),
        exercise_min_per_week=c1.number_input("Exercise min/wk", 0.0, 900.0, float(base.exercise_min_per_week), 10.0),
        sleep_hours=c2.number_input("Sleep (h)", 3.0, 11.0, float(base.sleep_hours), 0.5),
        sodium_g_per_day=c1.number_input("Sodium g/day", 0.5, 10.0, float(base.sodium_g_per_day), 0.1),
        smoker=c2.checkbox("Smoker", base.smoker),
    )

sync_on = st.sidebar.toggle("Sync twin with wearable data", value=True, help="Blend profile values with what the sensors observed over the last 48 h.")
stream = stream_for(base, with_events=True)
patient, observed = update_baselines(base, stream) if sync_on else (base, {})

st.sidebar.divider()
st.sidebar.markdown("**Data sources (simulated)**  \n⌚ Smartwatch: HR, SpO₂, skin temp, steps  \n🩸 CGM: glucose every 5 min  \n🩺 BP cuff: 3×/day  \n📋 EHR: labs & history")
st.sidebar.caption("⚠️ Research prototype on synthetic data - not a medical device.")

# ----------------------------------------------------------------------------- header
risk = predict_risk(bundle, patient)
st.title(f"Digital twin · {patient.name}")
cols = st.columns(6)
cols[0].metric("BMI", f"{patient.bmi:.1f}")
cols[1].metric("Blood pressure", f"{patient.systolic_bp:.0f}/{patient.diastolic_bp:.0f}")
cols[2].metric("HbA1c", f"{patient.hba1c:.1f}%")
cols[3].metric("Resting HR", f"{patient.resting_hr:.0f} bpm")
for i, (t, label) in enumerate(TARGETS.items()):
    band, _ = risk_band(risk[t])
    cols[4 + i].metric(label.replace(" (10-yr)", " risk"), f"{risk[t]:.0%}", band, delta_color="off")

tabs = st.tabs(["🧬 Twin state", "📡 Live monitoring", "⚠️ Risk drivers", "🔮 What-if simulator", "🩸 Glucose twin"])

# ----------------------------------------------------------------------------- 1. twin state
with tabs[0]:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("How the twin stays in sync")
        st.markdown(
            "1. **Ingest** wearable, CGM, BP-cuff and EHR data.\n"
            "2. **Estimate** the patient's current baselines (EWMA state estimation).\n"
            "3. **Calibrate** the physiological model (insulin sensitivity, basal glucose) to this patient's CGM trace.\n"
            "4. **Predict** 10-year risks with calibrated ML models and **explain** the drivers.\n"
            "5. **Simulate** interventions on the twin before trying them on the patient."
        )
        if sync_on:
            rows = [
                ("Resting HR (bpm)", base.resting_hr, observed["resting_hr"], patient.resting_hr),
                ("Systolic BP", base.systolic_bp, observed["systolic_bp"], patient.systolic_bp),
                ("Diastolic BP", base.diastolic_bp, observed["diastolic_bp"], patient.diastolic_bp),
                ("Fasting glucose", base.fasting_glucose, observed["fasting_glucose"], patient.fasting_glucose),
            ]
            st.dataframe(
                pd.DataFrame(rows, columns=["Parameter", "EHR profile", "Sensors (48 h)", "Twin state"]).round(1),
                hide_index=True, width="stretch",
            )
            st.caption(f"Average daily steps observed: {observed['daily_steps']:,.0f} · mean SpO₂ {observed['mean_spo2']:.1f}%")
        else:
            st.info("Sync is off - the twin uses the EHR profile only.")
    with right:
        st.subheader("Physiological calibration from CGM")
        cgm, meals, true, fitted, info = calibration(patient)
        prior = GlucoseParams.from_patient(patient)
        t_h = np.arange(1440) / 60
        fig = go.Figure()
        fig.add_scatter(x=t_h, y=cgm, mode="markers", name="CGM readings", marker=dict(size=4, color="#898781"))
        fig.add_scatter(x=t_h, y=simulate_glucose_day(prior, meals)["glucose"], name="Population prior", line=dict(color=C_SCEN, width=2, dash="dot"))
        fig.add_scatter(x=t_h, y=simulate_glucose_day(fitted, meals)["glucose"], name="Calibrated twin", line=dict(color=C_BASE, width=2))
        fig.update_xaxes(title="Hour of day", dtick=3)
        fig.update_yaxes(title="mg/dL")
        st.plotly_chart(style(fig), width="stretch")
        c = st.columns(3)
        c[0].metric("Insulin sensitivity", f"{fitted.si * 1e4:.2f}", f"{(fitted.si / prior.si - 1):+.0%} vs prior", delta_color="off")
        c[1].metric("Basal glucose (mg/dL)", f"{fitted.gb:.0f}", f"{fitted.gb - prior.gb:+.0f} vs prior", delta_color="off")
        c[2].metric("Fit error, RMSE (mg/dL)", f"{info['rmse_fitted']:.1f}", f"{info['rmse_fitted'] - info['rmse_prior']:+.1f}", delta_color="inverse")
        st.caption("SI in 10⁻⁴ mL/µU/min. The twin is fitted by Nelder-Mead least squares on the Bergman minimal model using the logged meals.")

# ----------------------------------------------------------------------------- 2. monitoring
with tabs[1]:
    mon = monitor_for(base)
    scored = mon.score(stream)
    alerts = mon.alerts(scored)
    hours = len(scored) * 5 / 60
    upto = st.slider("Replay stream up to hour", 1.0, hours, hours, 0.5, help="Scrub through the 48-hour wearable stream as if it were arriving live.")
    cutoff = scored.index[0] + pd.Timedelta(hours=upto)
    view = scored[scored.index <= cutoff]
    shown = [a for a in alerts if a.timestamp <= cutoff]

    c = st.columns(4)
    latest = view.iloc[-1]
    for col, v in zip(c, ["heart_rate", "spo2", "skin_temp", "glucose"]):
        col.metric(VITAL_LABELS[v], f"{latest[v]:.1f}", f"{latest[f'z_{v}']:+.1f} SD vs baseline", delta_color="off")

    left, right = st.columns([2, 1])
    with left:
        for v in ["heart_rate", "spo2", "glucose"]:
            fig = go.Figure()
            fig.add_scatter(x=view.index, y=view[v], name=VITAL_LABELS[v], line=dict(color=C_BASE, width=2))
            mu, sd = mon.baseline[v]
            fig.add_hrect(y0=mu - 3 * sd, y1=mu + 3 * sd, fillcolor=C_BASE, opacity=0.08, line_width=0)
            for sev in ("warning", "critical"):
                pts = [a for a in shown if a.vital == v and a.severity == sev]
                if pts:
                    fig.add_scatter(
                        x=[a.timestamp for a in pts], y=[a.value for a in pts], mode="markers", name=sev.title(),
                        marker=dict(size=11, color=STATUS[sev], symbol="triangle-up" if sev == "warning" else "x", line=dict(width=2, color="white")),
                    )
            st.plotly_chart(style(fig, 210, VITAL_LABELS[v]), width="stretch")
        st.caption("Shaded band = personal baseline ±3 SD learned by the twin from the patient's own 3-day history.")
    with right:
        st.subheader(f"Alerts ({len(shown)})")
        icon = {"warning": "🟠", "critical": "🔴"}
        if not shown:
            st.success("No alerts in this window.")
        for a in reversed(shown):
            st.markdown(f"{icon[a.severity]} **{a.severity.upper()}** · {a.timestamp:%a %H:%M}  \n{VITAL_LABELS[a.vital]} = **{a.value:.1f}** - {a.reason}")
        truth = stream[stream["event"] != ""].groupby("event")["timestamp"].min().sort_values()
        with st.expander("Ground-truth injected events"):
            st.dataframe(truth.rename("onset").reset_index(), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- 3. risk drivers
with tabs[2]:
    st.caption("Calibrated gradient-boosted models trained on a 20,000-person synthetic cohort. Each bar shows how much the 10-year risk would fall if that single factor reached its healthy target, holding everything else constant.")
    cols = st.columns(2)
    for col, (t, label) in zip(cols, TARGETS.items()):
        with col:
            band, status = risk_band(risk[t])
            st.subheader(label)
            st.markdown(f"<span style='font-size:2.4rem;font-weight:600'>{risk[t]:.1%}</span> &nbsp; <span style='color:{STATUS[status]}'>●</span> {band}", unsafe_allow_html=True)
            if t == "diabetes_10y" and (patient.hba1c >= 6.5 or patient.fasting_glucose >= 126):
                st.warning("Labs already meet diabetes criteria (HbA1c ≥ 6.5% or FPG ≥ 126) - focus on glycaemic control and CVD risk.")
            ex = explain_risk(bundle, patient, t)
            ex = ex[ex["risk_reduction"] > 0.001]
            if ex.empty:
                st.success("No modifiable factor is meaningfully raising this risk.")
                continue
            fig = go.Figure(go.Bar(
                y=ex["factor"], x=ex["risk_reduction"] * 100, orientation="h", marker_color=C_BASE,
                text=[f"−{v * 100:.1f} pts" for v in ex["risk_reduction"]], textposition="outside", cliponaxis=False,
                customdata=np.stack([ex["current"], ex["target"]], axis=1),
                hovertemplate="%{y}: %{customdata[0]:.1f} → %{customdata[1]:.1f}<br>risk −%{x:.1f} pts<extra></extra>",
            ))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="Risk reduction (percentage points)", range=[0, ex["risk_reduction"].max() * 100 * 1.25])
            st.plotly_chart(style(fig, 60 + 34 * len(ex)), width="stretch")
    with st.expander("Model performance (held-out synthetic test set)"):
        m = pd.DataFrame(bundle["metrics"]).T
        m.index = [TARGETS[i] for i in m.index]
        st.dataframe(m[["auc", "brier", "prevalence", "n_train", "n_test"]].round(3), width="stretch")

# ----------------------------------------------------------------------------- 4. what-if
with tabs[3]:
    st.caption("Design an intervention plan and let the twin project the next weeks. Compare it with doing nothing, before the patient commits to it.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Lifestyle**")
        deficit = st.slider("Calorie deficit (kcal/day)", 0, 800, 300, 50)
        exercise = st.slider("Exercise (min/week)", 0, 400, int(max(patient.exercise_min_per_week, 150)), 10)
        sodium = st.slider("Sodium (g/day)", 1.0, 6.0, float(min(patient.sodium_g_per_day, 2.5)), 0.1)
    with c2:
        st.markdown("**Habits**")
        sleep = st.slider("Sleep (hours)", 4.0, 9.0, float(max(patient.sleep_hours, 7.0)), 0.5)
        quit_smoking = st.checkbox("Quit smoking", value=patient.smoker, disabled=not patient.smoker)
        adherence = st.slider("Expected adherence", 0.2, 1.0, 0.8, 0.05)
    with c3:
        st.markdown("**Medication**")
        bp_med = st.checkbox("Start/intensify antihypertensive")
        metformin = st.checkbox("Metformin")
        statin = st.checkbox("Statin")
        weeks = st.select_slider("Horizon (weeks)", [12, 24, 36, 52], 24)

    plan = Plan("Intervention", deficit, exercise, sodium, sleep, quit_smoking and patient.smoker, bp_med, metformin, statin, adherence)
    df_base = projection(patient, Plan(), weeks)
    df_plan = projection(patient, plan, weeks)
    end_b, end_p = df_base.iloc[-1], df_plan.iloc[-1]

    k = st.columns(6)
    for col, (lab, key, fmt, unit) in zip(k, [
        ("Weight (kg)", "weight_kg", "{:.1f}", ""), ("Systolic BP", "systolic_bp", "{:.0f}", ""), ("HbA1c (%)", "hba1c", "{:.2f}", ""),
        ("LDL (mg/dL)", "ldl", "{:.0f}", ""), ("Diabetes risk", "diabetes_10y", "{:.1%}", ""), ("CVD risk", "cvd_10y", "{:.1%}", ""),
    ]):
        d = end_p[key] - end_b[key]
        dtxt = f"{d * 100:+.1f} pts" if key in TARGETS else f"{d:+.1f}{unit}"
        col.metric(f"{lab} @ wk {weeks}", fmt.format(end_p[key]) + unit, dtxt, delta_color="inverse")

    def compare(key, title, scale=1.0, fmt=".1f"):
        fig = go.Figure()
        fig.add_scatter(x=df_base["week"], y=df_base[key] * scale, name="No change", line=dict(color=C_SCEN, width=2, dash="dot"), hovertemplate=f"%{{y:{fmt}}}")
        fig.add_scatter(x=df_plan["week"], y=df_plan[key] * scale, name="With plan", line=dict(color=C_BASE, width=2), hovertemplate=f"%{{y:{fmt}}}")
        fig.update_xaxes(title="Week")
        return style(fig, 260, title)

    g = st.columns(2)
    g[0].plotly_chart(compare("diabetes_10y", "10-yr diabetes risk (%)", 100), width="stretch")
    g[1].plotly_chart(compare("cvd_10y", "10-yr cardiovascular risk (%)", 100), width="stretch")
    g = st.columns(3)
    g[0].plotly_chart(compare("weight_kg", "Weight (kg)"), width="stretch")
    g[1].plotly_chart(compare("systolic_bp", "Systolic BP (mmHg)", fmt=".0f"), width="stretch")
    g[2].plotly_chart(compare("hba1c", "HbA1c (%)", fmt=".2f"), width="stretch")
    with st.expander("Projection table"):
        st.dataframe(df_plan[["week", "weight_kg", "systolic_bp", "diastolic_bp", "hba1c", "ldl", "resting_hr", "diabetes_10y", "cvd_10y"]].round(3), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- 5. glucose twin
with tabs[4]:
    st.caption("A 24-hour run of the calibrated glucose-insulin model. Change meals, exercise or medication and see the glucose response before it happens.")
    _, _, _, fitted, _ = calibration(patient)
    c1, c2 = st.columns([1, 2])
    with c1:
        meals = []
        for i, (lab, m) in enumerate(zip(["Breakfast", "Lunch", "Dinner"], DEFAULT_MEALS)):
            a, b = st.columns(2)
            hr = a.slider(f"{lab} time", 5.0, 23.0, m.minute / 60, 0.25, key=f"mt{i}")
            carbs = b.slider(f"{lab} carbs (g)", 0, 150, int(m.carbs_g), 5, key=f"mc{i}")
            meals.append(Meal(int(hr * 60), carbs))
        walk = st.checkbox("30-min walk after dinner", True)
        met = st.checkbox("Take metformin", key="met_day")
    ex = [ExerciseBout(meals[2].minute + 20, 30, 0.5)] if walk else []
    sim = simulate_glucose_day(fitted, meals, ex, met)
    ref = simulate_glucose_day(fitted)
    metrics = glucose_metrics(sim["glucose"])
    with c2:
        fig = go.Figure()
        fig.add_hrect(y0=70, y1=180, fillcolor=STATUS["good"], opacity=0.07, line_width=0)
        fig.add_scatter(x=t_h, y=ref["glucose"], name="Usual day", line=dict(color=C_SCEN, width=2, dash="dot"))
        fig.add_scatter(x=t_h, y=sim["glucose"], name="Planned day", line=dict(color=C_BASE, width=2))
        fig.update_xaxes(title="Hour of day", dtick=3)
        fig.update_yaxes(title="Glucose (mg/dL)")
        st.plotly_chart(style(fig, 360), width="stretch")
        k = st.columns(4)
        k[0].metric("Time in range 70-180", f"{metrics['time_in_range_pct']:.0f}%")
        k[1].metric("Mean glucose (mg/dL)", f"{metrics['mean']:.0f}")
        k[2].metric("Peak (mg/dL)", f"{metrics['peak']:.0f}")
        k[3].metric("GMI (est. A1c)", f"{metrics['gmi']:.1f}%")
        st.caption("Green band = target range 70-180 mg/dL (international CGM consensus).")
