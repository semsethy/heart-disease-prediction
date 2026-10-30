"""Run: python -m uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, HTTPException, Request, Body
from app import ml
from app.schemas import Patient, HeartPrediction, ModelInfo


@asynccontextmanager
async def lifespan(app):
    app.state.models = {'heart': ml.load()}
    yield
    app.state.models.clear()


app = FastAPI(title='Heart Disease Prediction API', version='1.0.0', lifespan=lifespan,
              description='Classroom classifier for the Kaggle heart failure dataset. Not medical advice.')
UNAVAILABLE = {503: {'description': 'Model missing. Train and restart the API.'}}


def require_model(request):
    bundle = request.app.state.models['heart']
    if bundle is None:
        raise HTTPException(503, 'Model not loaded. Run python train_heart.py and restart the API.')
    return bundle


@app.get('/health')
def health(request: Request):
    loaded = request.app.state.models['heart'] is not None
    return {'status': 'ok' if loaded else 'degraded', 'models': {'heart': loaded}}


@app.get('/models', response_model=list[ModelInfo])
def models(request: Request):
    bundle = request.app.state.models['heart']
    return [dict(name='heart', loaded=bundle is not None,
                 **({key: bundle[key] for key in ['algorithm', 'trained_at', 'metrics', 'threshold', 'features', 'sklearn_version']} if bundle else {}))]


@app.post('/heart/predict', response_model=HeartPrediction, responses=UNAVAILABLE)
def predict(item: Patient, request: Request):
    return ml.predict_heart(require_model(request), [item])[0]


@app.post('/heart/predict/batch', response_model=list[HeartPrediction], responses=UNAVAILABLE)
def predict_batch(items: Annotated[list[Patient], Body(min_length=1, max_length=1000)], request: Request):
    return ml.predict_heart(require_model(request), items)
