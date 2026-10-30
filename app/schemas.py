"""Column names and category codes match the CSV, so the dataset is the API contract."""
from typing import Literal
from pydantic import BaseModel, Field

class Patient(BaseModel):
    model_config = {'extra': 'forbid', 'json_schema_extra': {'examples': [{'Age': 54, 'Sex': 'M', 'ChestPainType': 'ASY', 'RestingBP': 140, 'Cholesterol': 239, 'FastingBS': 0, 'RestingECG': 'Normal', 'MaxHR': 160, 'ExerciseAngina': 'N', 'Oldpeak': 1.2, 'ST_Slope': 'Flat'}]}}
    age: int = Field(alias='Age', ge=1, le=120, description='Age in years')
    sex: Literal['M', 'F'] = Field(alias='Sex')
    chest_pain_type: Literal['TA', 'ATA', 'NAP', 'ASY'] = Field(alias='ChestPainType', description='TA typical angina, ATA atypical angina, NAP non-anginal pain, ASY no symptoms')
    resting_bp: int = Field(alias='RestingBP', ge=0, le=250, description='Resting blood pressure in mm Hg; 0 means not measured')
    cholesterol: int = Field(alias='Cholesterol', ge=0, le=700, description='Serum cholesterol in mg/dl; 0 means not measured')
    fasting_bs: Literal[0, 1] = Field(alias='FastingBS', description='1 if fasting blood sugar > 120 mg/dl, otherwise 0')
    resting_ecg: Literal['Normal', 'ST', 'LVH'] = Field(alias='RestingECG')
    max_hr: int = Field(alias='MaxHR', ge=40, le=250, description='Maximum heart rate reached')
    exercise_angina: Literal['Y', 'N'] = Field(alias='ExerciseAngina', description='Chest pain during exercise')
    oldpeak: float = Field(alias='Oldpeak', ge=-5, le=10, description='ST depression; can be negative')
    st_slope: Literal['Up', 'Flat', 'Down'] = Field(alias='ST_Slope')

class HeartPrediction(BaseModel):
    label: Literal['no heart disease', 'heart disease']
    heart_disease_probability: float = Field(ge=0, le=1)
    has_heart_disease: bool
    threshold: float = Field(ge=0, le=1)

class ModelInfo(BaseModel):
    name: str
    loaded: bool
    algorithm: str | None = None
    trained_at: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    threshold: float | None = None
    features: list[str] = Field(default_factory=list)
    sklearn_version: str | None = None
