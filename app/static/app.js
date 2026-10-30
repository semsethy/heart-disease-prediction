// Sends the form to /heart/predict and shows the result.
const form = document.getElementById('patient-form');
const submit = document.getElementById('submit');
const formError = document.getElementById('form-error');
const cholesterol = form.elements.Cholesterol;
const cholUnknown = document.getElementById('chol-unknown');

const EXAMPLES = {
  low: { Age: 40, Sex: 'M', ChestPainType: 'ATA', RestingBP: 140, Cholesterol: 289, FastingBS: 0,
         RestingECG: 'Normal', MaxHR: 172, ExerciseAngina: 'N', Oldpeak: 0, ST_Slope: 'Up' },
  high: { Age: 48, Sex: 'F', ChestPainType: 'ASY', RestingBP: 138, Cholesterol: 214, FastingBS: 0,
          RestingECG: 'Normal', MaxHR: 108, ExerciseAngina: 'Y', Oldpeak: 1.5, ST_Slope: 'Flat' },
};

const LABELS = {
  Age: 'Age', RestingBP: 'Resting blood pressure', Cholesterol: 'Cholesterol', MaxHR: 'Maximum heart rate',
  Oldpeak: 'ST depression', Sex: 'Sex', ChestPainType: 'Chest pain type', FastingBS: 'Fasting blood sugar',
  RestingECG: 'Resting ECG', ExerciseAngina: 'Chest pain during exercise', ST_Slope: 'ST slope',
};

function fill(example) {
  for (const [name, value] of Object.entries(example)) form.elements[name].value = value;
  cholUnknown.checked = false;
  cholesterol.disabled = false;
}

// The model treats Cholesterol = 0 as "not measured", like the dataset.
cholUnknown.addEventListener('change', () => {
  cholesterol.disabled = cholUnknown.checked;
  if (cholUnknown.checked) cholesterol.value = '';
});

document.querySelectorAll('[data-example]').forEach((button) =>
  button.addEventListener('click', () => fill(EXAMPLES[button.dataset.example])));

function readForm() {
  const missing = [...form.elements]
    .filter((el) => el.name && !el.disabled && !el.checkValidity())
    .map((el) => LABELS[el.name]);
  if (missing.length) throw new Error(`Check these fields: ${missing.join(', ')}.`);
  const value = (name) => form.elements[name].value;
  return {
    Age: parseInt(value('Age'), 10),
    Sex: value('Sex'),
    ChestPainType: value('ChestPainType'),
    RestingBP: parseInt(value('RestingBP'), 10),
    Cholesterol: cholUnknown.checked ? 0 : parseInt(value('Cholesterol'), 10),
    FastingBS: parseInt(value('FastingBS'), 10),
    RestingECG: value('RestingECG'),
    MaxHR: parseInt(value('MaxHR'), 10),
    ExerciseAngina: value('ExerciseAngina'),
    Oldpeak: parseFloat(value('Oldpeak')),
    ST_Slope: value('ST_Slope'),
  };
}

async function errorMessage(response) {
  const body = await response.json().catch(() => ({}));
  if (response.status === 422 && Array.isArray(body.detail)) {
    return body.detail.map((d) => `${LABELS[d.loc.at(-1)] ?? d.loc.at(-1)}: ${d.msg}`).join('. ');
  }
  if (response.status === 503) return 'The model is not loaded. Run python train_heart.py and restart the server.';
  return body.detail || `Request failed (${response.status}).`;
}

function showResult(prediction) {
  const sick = prediction.has_heart_disease;
  const percent = (x) => `${Math.round(x * 100)}%`;
  const label = document.getElementById('result-label');
  label.textContent = sick ? 'Heart disease likely' : 'Heart disease unlikely';
  label.className = `result-label ${sick ? 'bad' : 'good'}`;
  document.getElementById('result-prob').textContent = percent(prediction.heart_disease_probability);
  const fill = document.getElementById('meter-fill');
  fill.style.width = percent(prediction.heart_disease_probability);
  fill.style.background = sick ? 'var(--bad)' : 'var(--good)';
  document.getElementById('meter-threshold').style.left = percent(prediction.threshold);
  document.getElementById('result-note').textContent =
    `The model says "likely" when the chance is ${percent(prediction.threshold)} or higher (the marker on the bar). ` +
    'This is a model estimate, not a diagnosis.';
  const result = document.getElementById('result');
  result.hidden = false;
  result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  formError.hidden = true;
  try {
    const patient = readForm();
    submit.disabled = true;
    submit.textContent = 'Predicting…';
    const response = await fetch('/heart/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patient),
    });
    if (!response.ok) throw new Error(await errorMessage(response));
    showResult(await response.json());
  } catch (error) {
    formError.textContent = error.message === 'Failed to fetch' ? 'Cannot reach the server. Is it running?' : error.message;
    formError.hidden = false;
  } finally {
    submit.disabled = false;
    submit.textContent = 'Predict';
  }
});

async function loadStatus() {
  const status = document.getElementById('model-status');
  try {
    const [info] = await (await fetch('/models')).json();
    if (!info.loaded) throw new Error();
    status.textContent = `Model: ${info.algorithm} · accuracy ${Math.round(info.metrics.accuracy * 100)}% · ` +
      `finds ${Math.round(info.metrics.recall * 100)}% of heart disease cases in testing`;
  } catch {
    status.textContent = 'Model not loaded. Run python train_heart.py and restart the server.';
    status.classList.add('down');
  }
}

fill(EXAMPLES.low);
loadStatus();
