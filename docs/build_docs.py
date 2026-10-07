"""Builds docs/Architecture_Diagram.pptx and docs/VitalTwin_Presentation.pptx.

Usage:  python docs/build_docs.py            (then export to PDF from PowerPoint / LibreOffice)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT))

INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
LINE = RGBColor(0xC3, 0xC2, 0xB7)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
BLUE_BG = RGBColor(0xE8, 0xF1, 0xFC)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
ORANGE_BG = RGBColor(0xFD, 0xEE, 0xE7)
AQUA = RGBColor(0x1B, 0xAF, 0x7A)
AQUA_BG = RGBColor(0xE6, 0xF6, 0xEF)
VIOLET = RGBColor(0x4A, 0x3A, 0xA7)
VIOLET_BG = RGBColor(0xED, 0xEB, 0xF7)
GRAY_BG = RGBColor(0xF3, 0xF2, 0xEF)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Segoe UI"


def box(slide, x, y, w, h, title, body=None, fill=GRAY_BG, edge=LINE, title_color=INK, size=11, body_size=9, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.08
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = edge
    shp.line.width = Pt(1.25)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.1)
    tf.margin_top = tf.margin_bottom = Inches(0.06)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = title
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), True, title_color, FONT
    for line in body or []:
        q = tf.add_paragraph()
        q.alignment = align
        rr = q.add_run()
        rr.text = line
        rr.font.size, rr.font.color.rgb, rr.font.name = Pt(body_size), INK2, FONT
    return shp


def text(slide, x, y, w, h, s, size=12, bold=False, color=INK, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, FONT
    return tb


def bullets(slide, x, y, w, h, items, size=14, color=INK2, gap=6):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        bold, rest = (item.split("|", 1) + [""])[:2] if "|" in item else ("", item)
        r0 = p.add_run()
        r0.text = "•  "
        r0.font.size, r0.font.color.rgb, r0.font.name = Pt(size), BLUE, FONT
        if bold:
            r1 = p.add_run()
            r1.text = bold
            r1.font.size, r1.font.bold, r1.font.color.rgb, r1.font.name = Pt(size), True, INK, FONT
        r2 = p.add_run()
        r2.text = rest
        r2.font.size, r2.font.color.rgb, r2.font.name = Pt(size), color, FONT
    return tb


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.75, dashed=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dashed:
        c.line.dash_style = 7  # dash
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd", {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def new_deck():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    return prs


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = WHITE
    return s


def header(slide, title, kicker=None):
    if kicker:
        text(slide, 0.6, 0.35, 12, 0.35, kicker.upper(), 11, True, BLUE)
    text(slide, 0.6, 0.6, 12.1, 0.8, title, 28, True, INK)


def footer(slide, n):
    text(slide, 0.6, 7.0, 9, 0.3, "VitalTwin · Patient Digital Twin for Chronic Disease · Happiest Health Digital Twin Challenge", 9, False, MUTED)
    text(slide, 12.0, 7.0, 0.8, 0.3, str(n), 9, False, MUTED, PP_ALIGN.RIGHT)


# ----------------------------------------------------------------------------- architecture
def architecture_slide(slide, top=1.45):
    y0 = top
    # Column 1: data sources
    text(slide, 0.4, y0, 2.3, 0.3, "1 · PHYSICAL PATIENT & DATA", 10, True, MUTED)
    srcs = [("⌚ Smartwatch", "HR · SpO₂ · skin temp · steps (1 min)"), ("🩸 CGM sensor", "Glucose every 5 min"),
            ("🩺 BP cuff", "Systolic / diastolic, 3×/day"), ("📋 EHR & labs", "HbA1c · lipids · history · meds"),
            ("📝 Patient log", "Meals · exercise · sleep")]
    for i, (t, b) in enumerate(srcs):
        box(slide, 0.4, y0 + 0.35 + i * 0.98, 2.3, 0.85, t, [b], GRAY_BG, LINE, INK, 11, 9)

    # Column 2: ingestion & sync
    text(slide, 3.05, y0, 2.6, 0.3, "2 · INGESTION & TWIN SYNC", 10, True, MUTED)
    box(slide, 3.05, y0 + 0.35, 2.6, 1.35, "Stream ingestion", ["Resample to 5-min windows", "Interpolate gaps, de-noise", "pandas pipeline (twin/data.py)"], AQUA_BG, AQUA)
    box(slide, 3.05, y0 + 1.85, 2.6, 1.35, "State estimation", ["EWMA blend of EHR + sensors", "Resting HR · BP · fasting glucose", "(twin/sync.py)"], AQUA_BG, AQUA)
    box(slide, 3.05, y0 + 3.35, 2.6, 1.55, "Physiological calibration", ["Fit insulin sensitivity SI and", "basal glucose Gb to CGM", "SciPy Nelder-Mead least squares", "RMSE → sensor-noise level"], AQUA_BG, AQUA)

    # Column 3: twin core
    text(slide, 6.0, y0, 3.6, 0.3, "3 · DIGITAL TWIN CORE", 10, True, MUTED)
    core = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.0), Inches(y0 + 0.35), Inches(3.6), Inches(4.55))
    core.adjustments[0] = 0.04
    core.fill.solid()
    core.fill.fore_color.rgb = WHITE
    core.line.color.rgb = BLUE
    core.line.width = Pt(2)
    core.line.dash_style = 7
    box(slide, 6.15, y0 + 0.5, 3.3, 0.7, "Twin state store", ["Patient profile + calibrated parameters"], BLUE_BG, BLUE)
    box(slide, 6.15, y0 + 1.35, 3.3, 1.55, "Physiology engine (mechanistic)", ["Bergman minimal model: glucose-insulin,", "meals, exercise, metformin (1-min ODE)", "Lifestyle model: weight · BP · HR · LDL", "· HbA1c (ADAG) week-by-week"], BLUE_BG, BLUE)
    box(slide, 6.15, y0 + 3.05, 3.3, 1.7, "AI / ML engine (data-driven)", ["Calibrated HistGradientBoosting:", "10-yr T2D & CVD risk (AUC 0.87 / 0.82)", "Counterfactual risk-driver explainer", "IsolationForest + personal z-score", "+ clinical limits → alerts"], VIOLET_BG, VIOLET)

    # Column 4: applications
    text(slide, 9.95, y0, 3.0, 0.3, "4 · APPLICATIONS (STREAMLIT)", 10, True, MUTED)
    apps = [("🧬 Twin state", "Sync view & calibration fit"), ("📡 Live monitoring", "Vitals replay + alert feed"),
            ("⚠️ Risk drivers", "Explainable 10-yr risk"), ("🔮 What-if simulator", "Compare plans over 12-52 wks"), ("🩸 Glucose twin", "Plan a day: meals, walk, meds")]
    for i, (t, b) in enumerate(apps):
        box(slide, 9.95, y0 + 0.35 + i * 0.92, 3.0, 0.8, t, [b], ORANGE_BG, ORANGE, INK, 11, 9)

    # Arrows
    for i in range(5):
        arrow(slide, 2.7, y0 + 0.78 + i * 0.98, 3.05, y0 + 0.78 + min(i, 2) * 1.5 + 0.25)
    arrow(slide, 4.35, y0 + 1.7, 4.35, y0 + 1.85)
    arrow(slide, 4.35, y0 + 3.2, 4.35, y0 + 3.35)
    arrow(slide, 5.65, y0 + 2.5, 6.0, y0 + 2.5)
    arrow(slide, 5.65, y0 + 4.1, 6.0, y0 + 4.1)
    for i in range(5):
        arrow(slide, 9.6, y0 + 2.6, 9.95, y0 + 0.75 + i * 0.92)

    # Feedback loop
    fb = y0 + 5.15
    box(slide, 0.4, fb, 12.55, 0.55, "Closed loop:  clinician / patient chooses an intervention on the twin  →  applied to the real patient  →  new sensor data re-syncs and re-calibrates the twin",
        None, GRAY_BG, LINE, INK2, 11, 9, PP_ALIGN.CENTER)


def build_architecture():
    prs = new_deck()
    s = blank(prs)
    header(s, "VitalTwin - System Architecture", "Architecture diagram")
    architecture_slide(s)
    out = DOCS / "Architecture_Diagram.pptx"
    prs.save(out)
    return out


# ----------------------------------------------------------------------------- presentation
def build_presentation(team: dict):
    metrics = json.loads((ROOT / "models" / "metrics.json").read_text())
    shots = DOCS / "screenshots"
    prs = new_deck()
    n = 0

    def slide(title, kicker):
        nonlocal n
        n += 1
        s = blank(prs)
        header(s, title, kicker)
        footer(s, n)
        return s

    # 1. Title
    n += 1
    s = blank(prs)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.18))
    band.fill.solid(); band.fill.fore_color.rgb = BLUE; band.line.fill.background()
    text(s, 0.8, 1.6, 11.5, 0.5, "HAPPIEST HEALTH · DIGITAL TWIN CHALLENGE", 14, True, BLUE)
    text(s, 0.8, 2.1, 11.5, 1.2, "VitalTwin", 60, True, INK)
    text(s, 0.8, 3.3, 11.5, 0.8, "A personal digital twin for chronic-disease patients that monitors, predicts and simulates", 22, False, INK2)
    text(s, 0.8, 4.7, 11.5, 1.6, [f"Team: {team['team']}", f"Members: {team['members']}", f"Institution: {team['college']}", "", f"Live demo: {team['live']}"], 16, False, INK)

    # 2. Problem
    s = slide("The problem: chronic disease is managed in snapshots", "Problem statement")
    bullets(s, 0.6, 1.6, 6.6, 5, [
        "India has 100M+ people with diabetes|  and ~315M with hypertension; many are undiagnosed or poorly controlled (ICMR-INDIAB, 2023).",
        "Care is episodic| - a clinic visit every 3 months sees one BP reading and one HbA1c, while the disease evolves every day.",
        "Wearables & CGMs produce rich data|, but nobody turns it into a personal, predictive model of the patient.",
        "Interventions are trial-and-error| - doctors cannot test 'what if this patient walks 30 min and cuts salt?' before prescribing it.",
    ], 15)
    box(s, 7.7, 1.7, 5.0, 3.6, "Our question", ["", "Can we build a living virtual replica of each patient that",
        "stays synced with their real data, flags deterioration early,", "predicts long-term risk, and lets clinicians safely test", "interventions on the twin first?"], BLUE_BG, BLUE, INK, 16, 13)

    # 3. Use case
    s = slide("Healthcare use case", "Who uses it and how")
    cards = [
        ("👤 Patient", ["Sees their own twin", "Understands what drives risk", "Plans meals & activity on the", "glucose twin before eating"]),
        ("🩺 Clinician", ["Remote monitoring dashboard", "Prioritised critical alerts", "Tests therapy plans on the twin", "before prescribing"]),
        ("🏥 Care program", ["Population triage by risk", "Preventive outreach for", "pre-diabetic / hypertensive", "members (e.g. corporate wellness)"]),
    ]
    for i, (t, b) in enumerate(cards):
        box(s, 0.6 + i * 4.15, 1.6, 3.9, 2.4, t, b, [BLUE_BG, AQUA_BG, ORANGE_BG][i], [BLUE, AQUA, ORANGE][i], INK, 17, 13)
    text(s, 0.6, 4.3, 12, 0.4, "Example journey - Ravi, 52, pre-diabetic, hypertensive smoker", 15, True, INK)
    bullets(s, 0.6, 4.75, 12.2, 2.2, [
        "Twin syncs| 48 h of smartwatch, CGM and BP-cuff data and calibrates his insulin sensitivity (−29 % vs population prior).",
        "Monitoring| catches a tachycardia episode, a night-time SpO₂ dip and a hypoglycaemic event as critical alerts.",
        "Risk view| shows 15 % diabetes / 19 % CVD 10-yr risk; smoking and systolic BP are the top modifiable drivers.",
        "What-if| a realistic plan (300 kcal deficit, 150 min/wk exercise, 2.5 g salt, quit smoking, 80 % adherence) cuts CVD risk to ~6 % in 24 weeks.",
    ], 13, gap=4)

    # 4. Solution overview
    s = slide("Solution: a hybrid physics + AI digital twin", "What we built")
    cols = [
        ("Sync", AQUA_BG, AQUA, ["Ingest multi-sensor streams", "EWMA state estimation", "Calibrate physiology to the", "patient's CGM (least squares)"]),
        ("Monitor", VIOLET_BG, VIOLET, ["Personal baselines", "IsolationForest anomalies", "Clinical hard limits", "De-duplicated alert feed"]),
        ("Predict", BLUE_BG, BLUE, ["10-yr T2D & CVD risk", "Calibrated gradient boosting", "Counterfactual explanations", "of modifiable drivers"]),
        ("Simulate", ORANGE_BG, ORANGE, ["Glucose-insulin ODE (24 h)", "Lifestyle & medication", "projection (12-52 wks)", "Scenario vs. no-change"]),
    ]
    for i, (t, f, e, b) in enumerate(cols):
        box(s, 0.6 + i * 3.1, 1.7, 2.9, 3.0, t, b, f, e, INK, 20, 13)
    text(s, 0.6, 5.0, 12.2, 1.6, ["Why hybrid? Mechanistic models give physiologically plausible, explainable simulations with little data;",
        "ML models capture population risk patterns that equations miss. Together they make a twin that can both explain and predict."], 14, False, INK2)

    # 5. Architecture
    s = slide("System architecture", "Architecture")
    architecture_slide(s, 1.35)

    # 6. AI/ML
    s = slide("AI / ML models & frameworks", "Model details")
    dm, cv = metrics["diabetes_10y"], metrics["cvd_10y"]
    rows = [
        ("Component", "Method", "Framework", "Result"),
        ("10-yr T2D risk", "HistGradientBoosting + isotonic calibration", "scikit-learn", f"AUC {dm['auc']:.3f} · Brier {dm['brier']:.3f}"),
        ("10-yr CVD risk", "HistGradientBoosting + isotonic calibration", "scikit-learn", f"AUC {cv['auc']:.3f} · Brier {cv['brier']:.3f}"),
        ("Risk explanation", "Counterfactual: factor → healthy target", "custom", "Per-patient ranked drivers"),
        ("Anomaly detection", "IsolationForest + personal z-score + limits", "scikit-learn", "All injected events caught"),
        ("Glucose physiology", "Bergman minimal model + gut absorption", "NumPy (Euler, 1 min)", "Est. HbA1c within 0.3 % of labs"),
        ("Twin calibration", "Nelder-Mead least squares on SI, Gb", "SciPy", "SI recovered within ~2 %"),
        ("Lifestyle projection", "Effect-size model, coupled to ODE", "NumPy", "Weight · BP · LDL · HbA1c"),
    ]
    tbl = s.shapes.add_table(len(rows), 4, Inches(0.6), Inches(1.6), Inches(12.1), Inches(4.2)).table
    widths = [2.3, 4.3, 2.3, 3.2]
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = val
            para = cell.text_frame.paragraphs[0]
            para.runs[0].font.size = Pt(13 if i else 13)
            para.runs[0].font.bold = i == 0
            para.runs[0].font.name = FONT
            para.runs[0].font.color.rgb = WHITE if i == 0 else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = BLUE if i == 0 else (GRAY_BG if i % 2 else WHITE)
    text(s, 0.6, 6.0, 12.1, 0.8, f"Trained on a {dm['n_train'] + dm['n_test']:,}-person synthetic cohort with FINDRISC / Framingham-inspired risk structure (80/20 stratified split). No real patient data is used; the pipeline is ready to retrain on NHANES, UK Biobank or hospital EHR data.", 12, False, MUTED)

    # 7-10. Screens
    for title, kicker, img, notes in [
        ("Twin state & physiological calibration", "Demo · sync", "twin_state.png", "EHR values are blended with 48 h of sensor data; the glucose model is fitted to this patient's CGM trace."),
        ("Live monitoring & alerts", "Demo · monitor", "monitoring.png", "Personal ±3 SD baselines, multivariate anomalies and clinical limits produce a prioritised alert feed."),
        ("Explainable risk drivers", "Demo · predict", "risk.png", "Each bar = risk reduction if that one factor reached its healthy target - actionable, not just a score."),
        ("What-if intervention simulator", "Demo · simulate", "whatif_charts.png", "Compare a care plan against no change before the patient commits to it."),
        ("Glucose twin: plan a day", "Demo · simulate", "glucose.png", "Move meals, change carbs, add a walk or metformin - see time-in-range before it happens."),
    ]:
        s = slide(title, kicker)
        p = shots / img
        if p.exists():
            pic = s.shapes.add_picture(str(p), Inches(0.6), Inches(1.5), width=Inches(8.6))
            pic.line.color.rgb = LINE
        text(s, 9.5, 1.6, 3.3, 4.5, notes, 15, False, INK2)

    # 11. Outcomes
    s = slide("Outcomes", "Results of the proof of concept")
    stats = [(f"{dm['auc']:.2f}", "AUC · diabetes risk"), (f"{cv['auc']:.2f}", "AUC · CVD risk"), ("~2 %", "error recovering hidden insulin sensitivity"), ("4 / 4", "injected clinical events detected")]
    for i, (v, l) in enumerate(stats):
        text(s, 0.6 + i * 3.1, 1.7, 2.9, 0.9, v, 40, True, BLUE)
        text(s, 0.6 + i * 3.1, 2.6, 2.9, 0.7, l, 13, False, INK2)
    bullets(s, 0.6, 3.6, 12.2, 3.2, [
        "Working end-to-end twin| - sync, monitor, predict, explain and simulate in one interactive app.",
        "Physiologically grounded| - simulated HbA1c within 0.3 % of each persona's lab value before any calibration.",
        "Actionable| - for the demo patient a realistic plan lowers 10-yr CVD risk from 19 % to ~6 % and diabetes risk from 15 % to ~8 % in 24 weeks.",
        "Tested| - 10 automated tests cover the physiology, calibration, ML ranking and anomaly detection.",
    ], 14)

    # 12. Roadmap
    s = slide("Limitations & roadmap", "Next steps")
    box(s, 0.6, 1.6, 5.9, 4.8, "Current limitations", ["", "• Synthetic training cohort - risks are illustrative", "• Effect sizes simplified from literature", "• Simulated sensor streams, single-patient app",
        "• Not clinically validated; not a medical device"], GRAY_BG, LINE, INK, 18, 14, anchor=MSO_ANCHOR.TOP)
    box(s, 6.8, 1.6, 5.9, 4.8, "Roadmap", ["", "• Retrain on real cohorts (NHANES, ICMR-INDIAB, EHR)", "• Live device APIs: Health Connect, HealthKit, Libre", "• FHIR integration & clinician multi-patient view",
        "• Kalman / Bayesian twin updating with uncertainty bands", "• Cardiac (HRV, ECG) and renal sub-models", "• Prospective validation study with a partner hospital"], BLUE_BG, BLUE, INK, 18, 14, anchor=MSO_ANCHOR.TOP)

    # 13. Thanks
    s = slide("Thank you", "VitalTwin")
    text(s, 0.6, 1.8, 12, 2.5, [f"Team: {team['team']}", f"Members: {team['members']}", f"Institution: {team['college']}", "", f"Live demo: {team['live']}", f"Code: {team['repo']}", f"Demo video: {team['video']}"], 18, False, INK)
    text(s, 0.6, 5.6, 12, 0.6, "Research prototype on synthetic data. Not intended for diagnosis or treatment decisions.", 12, False, MUTED)

    out = DOCS / "VitalTwin_Presentation.pptx"
    prs.save(out)
    return out


if __name__ == "__main__":
    team = json.loads((DOCS / "team.json").read_text(encoding="utf-8"))
    print(build_architecture())
    print(build_presentation(team))
