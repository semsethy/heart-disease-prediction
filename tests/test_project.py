import joblib
import pytest
from fastapi.testclient import TestClient
from app import ml
from app.main import app
from app.schemas import Patient


def test_training_quality_and_round_trip(bundle, examples, tmp_path):
    assert bundle['metrics']['accuracy'] > bundle['baseline_accuracy'] + .2
    assert bundle['metrics']['recall'] > .85
    assert bundle['metrics']['roc_auc'] > .88
    path = tmp_path / 'model.joblib'
    joblib.dump(bundle, path)
    patients = [Patient(**x) for x in examples]
    assert ml.predict_heart(bundle, patients) == ml.predict_heart(joblib.load(path), patients)


def test_health_metadata_and_contract(client):
    assert client.get('/health').json() == {'status': 'ok', 'models': {'heart': True}}
    info = client.get('/models').json()[0]
    assert info['loaded'] and len(info['features']) == 11
    spec = client.get('/openapi.json').json()
    assert '503' in spec['paths']['/heart/predict']['post']['responses']
    assert client.get('/docs').status_code == 200


def test_single_batch_and_order(client, examples):
    single = [client.post('/heart/predict', json=x).json() for x in examples]
    response = client.post('/heart/predict/batch', json=examples)
    assert response.status_code == 200
    assert response.json() == single
    assert [x['label'] for x in single] == ['no heart disease', 'heart disease']
    for prediction in single:
        assert prediction['has_heart_disease'] == (prediction['heart_disease_probability'] >= prediction['threshold'])


@pytest.mark.parametrize('change', ['missing', 'extra', 'category', 'null', 'text', 'too_old', 'negative_bp'])
def test_invalid_inputs(client, examples, change):
    row = examples[0].copy()
    if change == 'missing': row.pop('ST_Slope')
    if change == 'extra': row['HeartDisease'] = 0
    if change == 'category': row['ChestPainType'] = 'invalid'
    if change == 'null': row['Cholesterol'] = None
    if change == 'text': row['Age'] = 'old'
    if change == 'too_old': row['Age'] = 500
    if change == 'negative_bp': row['RestingBP'] = -10
    assert client.post('/heart/predict', json=row).status_code == 422


def test_not_measured_zero_and_batch_bounds(client, examples):
    row = examples[0] | {'Cholesterol': 0}
    assert client.post('/heart/predict', json=row).status_code == 200
    for rows in [[], [row] * 1001, [row, row | {'Sex': 'X'}]]:
        assert client.post('/heart/predict/batch', json=rows).status_code == 422


def test_missing_model(monkeypatch, examples):
    monkeypatch.setattr(ml, 'load', lambda: None)
    with TestClient(app) as client:
        assert client.get('/health').json()['status'] == 'degraded'
        assert client.get('/models').json()[0]['loaded'] is False
        assert client.post('/heart/predict', json=examples[0]).status_code == 503
        assert client.post('/heart/predict/batch', json=examples).status_code == 503


def test_frontend_is_served(client):
    page = client.get('/')
    assert page.status_code == 200 and 'patient-form' in page.text
    for asset in ['/static/app.js', '/static/styles.css']:
        assert client.get(asset).status_code == 200
    assert '/' not in client.get('/openapi.json').json()['paths']
