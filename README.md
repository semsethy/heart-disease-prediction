# Heart Disease Prediction API

A simple machine learning project that predicts whether a patient has **heart
disease** from 11 clinical features. It uses scikit-learn for training and
FastAPI for predictions.

> Educational use only. This is not medical advice and must not be used to
> diagnose anyone.

## Run with Docker

The easiest way to run the app. You only need
[Docker](https://www.docker.com/products/docker-desktop/) installed.

```bash
docker compose up -d --build
```

Open **http://localhost:8000**. The model is trained while the image builds,
so there is no separate training step.

Useful commands:

```bash
docker compose ps        # status (shows "healthy" when ready)
docker compose logs -f   # follow the server logs
docker compose down      # stop and remove the container
```

To use a different port, change the left number in `compose.yaml`, for
example `"8080:8000"`, then open http://localhost:8080.

Without Compose:

```bash
docker build -t heart-disease-prediction .
docker run -d -p 8000:8000 --name heart-api heart-disease-prediction
```

## Run without Docker

Use Python 3.11 or newer. Run these commands from the project folder.

### 1. Set up the environment

```bash
python -m venv .venv
```

Activate it on macOS / Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies (`requirements-dev.txt` adds the notebook and test tools
to the app's `requirements.txt`):

```bash
pip install -r requirements-dev.txt
```

### 2. Train the model

Make sure the dataset is at `data/heart/heart.csv`, then run:

```bash
python train_heart.py
```

Training compares logistic regression, decision tree, and random forest using
5-fold cross-validation. It selects a model, chooses a probability threshold
using training data, and evaluates on a separate 20% test set.

The preprocessing steps, model, feature names, threshold, and metadata are
saved together in `models/heart.joblib`.

To choose a model explicitly:

```bash
python train_heart.py --algorithm "logistic regression"
```

### 3. Start the API

```bash
python -m uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** to use the web page: fill in the patient's
details (or click an example) and press **Predict**.

Open **http://127.0.0.1:8000/docs** to try the API directly. Under
`POST /heart/predict`, click **Try it out**, use the example request, and
click **Execute**.

Train the model before starting the API. After retraining, restart the API to
load the new model.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Web page for single predictions |
| GET | `/health` | Check whether the model is loaded |
| GET | `/models` | View the model, features, metrics, and threshold |
| POST | `/heart/predict` | Predict one patient |
| POST | `/heart/predict/batch` | Predict 1–1,000 patients |

Requests use the 11 column names from the CSV:

| Field | Allowed values |
| --- | --- |
| `Age` | 1–120 years |
| `Sex` | `M`, `F` |
| `ChestPainType` | `TA` typical angina, `ATA` atypical angina, `NAP` non-anginal pain, `ASY` no symptoms |
| `RestingBP` | 0–250 mm Hg (`0` = not measured) |
| `Cholesterol` | 0–700 mg/dl (`0` = not measured) |
| `FastingBS` | `1` if fasting blood sugar > 120 mg/dl, otherwise `0` |
| `RestingECG` | `Normal`, `ST`, `LVH` |
| `MaxHR` | 40–250 beats per minute |
| `ExerciseAngina` | `Y`, `N` |
| `Oldpeak` | -5 to 10 (ST depression, can be negative) |
| `ST_Slope` | `Up`, `Flat`, `Down` |

Missing fields, extra fields, invalid categories, and out-of-range numbers
return **422**.

Example request:

```json
{
  "Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140,
  "Cholesterol": 239, "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 160,
  "ExerciseAngina": "N", "Oldpeak": 1.2, "ST_Slope": "Flat"
}
```

Example response:

```json
{
  "label": "heart disease",
  "heart_disease_probability": 0.83,
  "has_heart_disease": true,
  "threshold": 0.41
}
```

The label is `heart disease` when the probability is greater than or equal to
the saved threshold. The threshold is selected during training; check
`/models` for its actual value.

If the model file is missing, `/health` reports `degraded` and prediction
requests return **503**. Train the model and restart the API.

You can also send the example requests in `requests.http` using a compatible
HTTP client extension.

## How the data is prepared

The model pipeline handles three kinds of columns:

- **Categories** (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`,
  `ST_Slope`) are one-hot encoded, as in the mushroom project.
- **Hidden missing values**: 172 rows have `Cholesterol = 0` and one has
  `RestingBP = 0`. These mean "not measured", so they are replaced with the
  training median, and an extra yes/no column records that the value was
  missing. In this dataset, 88% of patients with missing cholesterol have heart
  disease, compared with 48% of the rest, so that flag is useful information.
- **Other numbers** (`Age`, `FastingBS`, `MaxHR`, `Oldpeak`) are scaled to a
  similar range.

All of this is inside the saved pipeline, so the API sends raw values and the
model prepares them the same way it did during training.

## Results

From one training run on macOS (random forest, threshold 0.41, 184 test patients):

| Metric | Score |
| --- | --- |
| Baseline accuracy (always "heart disease") | 0.554 |
| Accuracy | 0.891 |
| Precision | 0.866 |
| Recall | 0.951 |
| F1 | 0.907 |
| ROC AUC | 0.941 |

Training on Linux (inside Docker) can give slightly different numbers, for
example a threshold of 0.39 and accuracy of 0.886, because random forests can
vary a little between platforms.

The model found 97 of 102 heart disease patients in the test set (5 missed)
and wrongly flagged 15 of 82 healthy patients. `ST_Slope` is the most
important feature, followed by `ChestPainType`, `Sex`, and `Cholesterol`.

## Project structure

```text
app/
  main.py                 API endpoints and startup
  schemas.py              Request and response validation
  ml.py                   Model loading and prediction
  static/                 Web page (index.html, styles.css, app.js)
data/heart/
  heart.csv               Training dataset
models/
  heart.joblib            Generated by training (ignored by Git)
notebooks/
  heart_model.ipynb       Model training notebook
train_heart.py            Training script
requests.http             Example API requests
tests/                    Automated tests
requirements.txt          App dependencies (used by Docker)
requirements-dev.txt      Adds notebook and test tools
Dockerfile                Container image (trains the model during build)
compose.yaml              Runs the container on port 8000
```

## Dataset

The [Heart Failure Prediction dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
by fedesoriano contains 918 patients and 11 features, combined from five UCI
heart disease datasets. The target column is `HeartDisease`: `1` means heart
disease and `0` means no heart disease.

## Tests

```bash
python -m pytest -q
```
