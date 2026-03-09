from flask import Flask, render_template, jsonify, request
import os
import sys
import socket
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.signal import welch, butter, filtfilt

warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='.')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
ALL_MODELS_FILENAME = os.path.join(SCRIPT_DIR, "all_models.joblib")

# Available models (synced with eeg_ml.py / predict_eeg.py)
AVAILABLE_MODELS = [
    'ExtraTreesClassifier', 'Random Forest', 'Gradient Boosting', 'Neural Network',
]
DEFAULT_MODEL = 'ExtraTreesClassifier'

# ============================================================================
# EEG Feature Extraction (mirrors predict_eeg.py)
# ============================================================================
FS = 128
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

def _butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return b, a

def _bandpass_filter(data, fs=FS):
    b, a = _butter_bandpass(1.0, 40.0, fs)
    return filtfilt(b, a, data, axis=0)

def _band_powers(signal, fs=FS):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    return np.array([
        np.trapezoid(psd[np.logical_and(freqs >= lo, freqs <= hi)],
                     freqs[np.logical_and(freqs >= lo, freqs <= hi)])
        for lo, hi in BANDS.values()
    ])

def _extract_features(data, window_sec=2, fs=FS):
    window_size = int(window_sec * fs)
    data_filt = _bandpass_filter(data, fs) if data.shape[0] > 128 else data.copy()
    X = []
    n_windows = data_filt.shape[0] // window_size
    if n_windows > 0:
        for w in range(n_windows):
            seg = data_filt[w * window_size:(w + 1) * window_size, :]
            feats = []
            for ch in range(seg.shape[1]):
                feats.extend(_band_powers(seg[:, ch], fs))
            X.append(feats)
    else:
        feats = []
        for ch in range(data_filt.shape[1]):
            feats.extend(_band_powers(data_filt[:, ch], fs))
        X.append(feats)
    return np.array(X)

# ============================================================================
# Load models once at startup
# ============================================================================
_trained_models = {}   # { model_name: (clf, scaler) }

def _load_models():
    global _trained_models
    if not os.path.exists(ALL_MODELS_FILENAME):
        print(f"Warning: {ALL_MODELS_FILENAME} not found. Run eeg_ml.py all first.",
              file=sys.stderr)
        return
    models_data = joblib.load(ALL_MODELS_FILENAME)
    all_models = models_data['models']
    _trained_models = {name: (clf, scaler)
                       for name, (clf, scaler) in all_models.items()
                       if clf is not None}
    # Put ExtraTreesClassifier first
    ordered = sorted(_trained_models.keys(),
                     key=lambda n: (0 if n == DEFAULT_MODEL else 1, n))
    _trained_models = {n: _trained_models[n] for n in ordered}
    print(f"Loaded models: {list(_trained_models.keys())}")

_load_models()

# ============================================================================
# Helpers
# ============================================================================
def get_test_files():
    if not os.path.exists(DATA_DIR):
        return []
    return sorted(f for f in os.listdir(DATA_DIR)
                  if f.endswith("_test_data.csv") and
                  os.path.isfile(os.path.join(DATA_DIR, f)))

def _predict_one(test_path, clf, scaler):
    df = pd.read_csv(test_path)
    data = df.values.astype(float)
    X = _extract_features(data)
    if scaler is not None:
        X = scaler.transform(X)
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)
    unique, counts = np.unique(y_pred, return_counts=True)
    majority = int(unique[np.argmax(counts)])
    avg_proba = np.mean(y_proba, axis=0)
    label = "NON-ATTENTIVE" if majority == 1 else "ATTENTIVE"
    return {
        'label': label,
        'confidence': float(avg_proba[majority] * 100),
        'n_windows': int(len(y_pred)),
        'attentive_windows': int(np.sum(y_pred == 0)),
        'non_attentive_windows': int(np.sum(y_pred == 1)),
        'avg_prob_attentive': float(avg_proba[0] * 100),
        'avg_prob_non_attentive': float(avg_proba[1] * 100),
        'max_confidence': float(np.max(y_proba[:, majority]) * 100),
        'min_confidence': float(np.min(y_proba[:, majority]) * 100),
        'std_confidence': float(np.std(y_proba[:, majority]) * 100),
    }

# ============================================================================
# Routes
# ============================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/test-files')
def api_test_files():
    return jsonify({'files': get_test_files()})

@app.route('/api/models')
def api_models():
    return jsonify({'models': list(_trained_models.keys()), 'default': DEFAULT_MODEL})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        test_file = data.get('file')
        if not test_file:
            return jsonify({'error': 'No file specified'}), 400

        file_path = os.path.join(DATA_DIR, test_file)
        if not os.path.exists(file_path):
            return jsonify({'error': f'File not found: {test_file}'}), 404

        if not _trained_models:
            return jsonify({'error': 'No models loaded. Run eeg_ml.py all first.'}), 500

        # Optional single-model filter
        selected_model = data.get('model')
        if selected_model:
            if selected_model not in _trained_models:
                return jsonify({'error': f'Model not found: {selected_model}'}), 400
            models_to_run = {selected_model: _trained_models[selected_model]}
            active_default = selected_model
        else:
            models_to_run = _trained_models
            active_default = DEFAULT_MODEL

        model_results = {}
        for model_name, (clf, scaler) in models_to_run.items():
            try:
                model_results[model_name] = _predict_one(file_path, clf, scaler)
            except Exception as e:
                model_results[model_name] = {'error': str(e)}

        # Majority vote across models for overall status
        labels = [r['label'] for r in model_results.values() if 'label' in r]
        overall = max(set(labels), key=labels.count) if labels else 'UNKNOWN'

        # Default model result for gradient bar
        default_result = model_results.get(active_default) or next(iter(model_results.values()))

        return jsonify({
            'success': True,
            'file': test_file,
            'default_model': active_default,
            'overall_status': overall,
            'default_result': default_result,
            'models': model_results,
        })
    except Exception as e:
        return jsonify({'error': f'Request error: {str(e)}'}), 400

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    port = 5004
    print(f"Starting EEG Prediction UI...")
    print(f"Open http://localhost:{port} in your browser")
    print(f"Available test files: {get_test_files()}")
    app.run(debug=False, port=port, host='127.0.0.1')
