"""Load a trusted local model once, then predict in feature order."""
from pathlib import Path
import warnings
import joblib
import pandas as pd
import sklearn

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


def load(name='heart'):
    path = MODELS_DIR / f'{name}.joblib'
    if not path.exists():
        return None
    bundle = joblib.load(path)
    if bundle['sklearn_version'] != sklearn.__version__:
        warnings.warn('Model scikit-learn version differs; retrain before serving.', RuntimeWarning)
    return bundle


def predict_heart(bundle, patients):
    frame = pd.DataFrame([item.model_dump(by_alias=True) for item in patients], columns=bundle['features'])
    model = bundle['model']
    probabilities = model.predict_proba(frame)[:, list(model.classes_).index(1)]
    return [dict(label='heart disease' if p >= bundle['threshold'] else 'no heart disease',
                 heart_disease_probability=float(p), has_heart_disease=bool(p >= bundle['threshold']),
                 threshold=bundle['threshold']) for p in probabilities]
