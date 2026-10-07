# Demo video script (target 16-18 minutes)

Record the screen at 1080p with `streamlit run app.py` open in the browser. Upload to YouTube as **Unlisted** and paste the link into the README.

| Time | Section | What to show / say |
|---|---|---|
| 0:00-1:30 | Intro | Team, institution, project name. Title slide of `VitalTwin_Presentation.pdf`. |
| 1:30-3:30 | Problem & use case | Slides 2-3: chronic disease is managed in snapshots; patient / clinician / care-program users; Ravi's journey. |
| 3:30-5:30 | Architecture | Slide 5 / `Architecture_Diagram.pdf`: data → sync → hybrid twin core (physiology + ML) → applications → closed loop. |
| 5:30-6:30 | AI/ML details | Slide 6: models, frameworks, metrics; mention synthetic data honestly. |
| 6:30-8:30 | **Live: Twin state** | Select Ravi. Toggle *Sync* off/on and show the EHR vs sensor vs twin table. Explain the CGM calibration chart: the population prior (dotted) vs the calibrated twin (solid) fitted to the dots; insulin sensitivity −29 %, RMSE drops to ~4.5 mg/dL (the sensor-noise level). |
| 8:30-10:30 | **Live: Monitoring** | Drag the replay slider from hour 1 to 48. Point out the tachycardia (~10:45), hyperglycaemia (~21:00), night SpO₂ dip (~03:00) and hypoglycaemia (~15:00 day 2). Open the "Ground-truth injected events" expander to show every event was caught. |
| 10:30-12:00 | **Live: Risk drivers** | 15 % diabetes / 19 % CVD risk. Read the top drivers (fasting glucose & BMI; smoking & systolic BP). Open "Model performance". Edit the profile in the sidebar (e.g. set Smoker off) and watch the risk update. |
| 12:00-14:30 | **Live: What-if** | Default plan → CVD risk 19 % → ~6 % in 24 weeks. Lower adherence to 0.4 and show the smaller benefit. Add a statin / antihypertensive and compare. Switch horizon to 52 weeks. |
| 14:30-16:00 | **Live: Glucose twin** | Increase lunch carbs to 120 g → peak and time-in-range worsen; add the post-dinner walk and metformin → recover. Switch to Meera (type 2 diabetes) to show a very different response. |
| 16:00-17:30 | Outcomes, limitations, roadmap | Slides 12-13. Close with the thank-you slide. |

Tips: run the app once before recording, so the models are trained and cached. Zoom the browser to 90 % so the charts fit on screen.
