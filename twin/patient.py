"""Patient profile: the static + slowly varying state the digital twin mirrors."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Patient:
    patient_id: str
    name: str
    age: int
    sex: str  # "M" or "F"
    height_cm: float
    weight_kg: float
    systolic_bp: float
    diastolic_bp: float
    resting_hr: float
    fasting_glucose: float  # mg/dL
    hba1c: float  # %
    ldl: float  # mg/dL
    hdl: float  # mg/dL
    smoker: bool
    family_history_diabetes: bool
    exercise_min_per_week: float
    sleep_hours: float
    sodium_g_per_day: float
    on_bp_meds: bool = False
    # 0..1 progress of a quit attempt; risk benefit of quitting accrues gradually.
    smoking_cessation: float = 0.0

    @property
    def bmi(self) -> float:
        return self.weight_kg / (self.height_cm / 100) ** 2

    def features(self) -> dict[str, float]:
        """Feature vector consumed by the ML risk models."""
        return {
            "age": self.age,
            "male": 1.0 if self.sex == "M" else 0.0,
            "bmi": self.bmi,
            "systolic_bp": self.systolic_bp,
            "diastolic_bp": self.diastolic_bp,
            "resting_hr": self.resting_hr,
            "fasting_glucose": self.fasting_glucose,
            "hba1c": self.hba1c,
            "ldl": self.ldl,
            "hdl": self.hdl,
            "smoker": float(self.smoker),
            "family_history_diabetes": float(self.family_history_diabetes),
            "exercise_min_per_week": self.exercise_min_per_week,
            "sleep_hours": self.sleep_hours,
            "sodium_g_per_day": self.sodium_g_per_day,
            "on_bp_meds": float(self.on_bp_meds),
        }

    def to_dict(self) -> dict:
        return asdict(self)


# Fictional personas used for the demo. Any resemblance to real people is coincidental.
SAMPLE_PATIENTS: dict[str, Patient] = {
    "P001": Patient(
        "P001", "Ravi (52, pre-diabetic, hypertensive)", 52, "M", 170, 86, 146, 92, 80,
        112, 6.1, 145, 40, True, True, 40, 6.0, 4.5,
    ),
    "P002": Patient(
        "P002", "Meera (61, type 2 diabetes)", 61, "F", 158, 74, 138, 84, 76,
        156, 7.8, 128, 46, False, True, 60, 6.5, 3.8, True,
    ),
    "P003": Patient(
        "P003", "Arjun (34, healthy baseline)", 34, "M", 178, 72, 118, 76, 62,
        88, 5.2, 98, 55, False, False, 200, 7.5, 2.5,
    ),
}
