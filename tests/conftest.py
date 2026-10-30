import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app import ml
from app.main import app
from train_heart import train, DATA


@pytest.fixture(scope='session')
def bundle(tmp_path_factory):
    return train(models_dir=tmp_path_factory.mktemp('models'), verbose=False)


@pytest.fixture
def client(bundle, monkeypatch):
    monkeypatch.setattr(ml, 'load', lambda: bundle)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def examples():
    """A clear-cut patient without and one with heart disease, as plain JSON values.
    The model is not perfect, so borderline rows would make the label check flaky."""
    df = pd.read_csv(DATA)
    healthy = df.query("HeartDisease == 0 and ST_Slope == 'Up' and ExerciseAngina == 'N' and ChestPainType == 'ATA'")
    sick = df.query("HeartDisease == 1 and ST_Slope == 'Flat' and ExerciseAngina == 'Y' and ChestPainType == 'ASY'")
    rows = [group.drop(columns='HeartDisease').iloc[0] for group in [healthy, sick]]
    return [{key: value.item() if hasattr(value, 'item') else value for key, value in row.items()} for row in rows]
