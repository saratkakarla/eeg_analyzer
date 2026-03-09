import numpy as np
import pandas as pd
import os
from scipy.signal import welch, butter, filtfilt
import joblib

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "data")
ALL_MODELS_FILENAME = os.path.join(DATA_DIR, "all_models.joblib")
MODEL_FILENAME = os.path.join(DATA_DIR, "data", "reference_model.joblib")
REFERENCE_DATA_PATH = "data/reference_data.csv"
FULL_FILENAME = os.path.join(DATA_DIR, REFERENCE_DATA_PATH)
OUT_FILENAME_MODEL = "reference_model.joblib"

# Available models
AVAILABLE_MODELS = [
    'Random Forest', 'Gradient Boosting',
    'Neural Network', 'ExtraTreesClassifier',
]

# Model shortcuts
MODEL_SHORTCUTS = {
    'rf': 'Random Forest',
    'gb': 'Gradient Boosting',
    'nn': 'Neural Network',
    'et': 'ExtraTreesClassifier',
}

def print_usage():
    print("\nUsage: python predict_eeg.py [model_name]")
    print("Runs prediction on all *_test_data.csv files in the data directory.")
    print("\nAvailable models:")
    for model in AVAILABLE_MODELS:
        print(f"  - {model}")
    print("\nModel shortcuts:")
    for shortcut, full_name in MODEL_SHORTCUTS.items():
        print(f"  - {shortcut:3s} → {full_name}")
    print("\nExample:")
    print("  python predict_eeg.py gb    # Gradient Boosting on all test files")
    print("  python predict_eeg.py et    # ExtraTreesClassifier on all test files")
    print("  python predict_eeg.py       # Default: ExtraTreesClassifier listed first")
    print()

# PiEEG-16 channel order
pieeg_channels_order = [
    "Fp1", "Fp2",    # frontopolar
    "F7", "F3", "F4", "F8",   # frontal
    "C3", "C4",               # central
    "P3", "P4",               # parietal
    "T7", "T8",               # temporal
    "P7", "P8",               # parietal-temporal
    "O1", "O2"                # occipital
]

# Sampling frequency
FS = 128

# Band definitions
BANDS = {
    "delta": (1, 4),   # sleepy
    "theta": (4, 8),   # dreamy
    "alpha": (8, 13),  # relaxed
    "beta":  (13, 30), # thinking
}

# ============================================================================
# Feature Extraction Functions
# ============================================================================
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=FS, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)

def band_powers(signal, fs=FS):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    feats = []
    for low, high in BANDS.values():
        idx = np.logical_and(freqs >= low, freqs <= high)
        feats.append(np.trapezoid(psd[idx], freqs[idx]))
    return np.array(feats)

def extract_features(data, channel_cols, window_sec=2, fs=FS):
    """
    Extract features from EEG data using windowing and band power analysis.
    
    Args:
        data: DataFrame with EEG channels
        channel_cols: List of channel column names
        window_sec: Window size in seconds
        fs: Sampling frequency
    
    Returns:
        X: Feature matrix
    """
    window_size = int(window_sec * fs)
    
    X = []
    
    # Filter the data - use a smaller window for filtering if data is small
    data_copy = data.copy()
    if data.shape[0] > 128:  # Only filter if we have enough samples
        data_filt = bandpass_filter(data_copy, lowcut=1.0, highcut=40.0, fs=fs, order=4)
    else:
        data_filt = data_copy
    
    n_samples = data_filt.shape[0]
    
    # If we have enough samples for windowing, use windows
    if n_samples >= window_size:
        n_windows = n_samples // window_size
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            window = data_filt[start:end, :]
            
            feats = []
            for ch in range(window.shape[1]):
                bp = band_powers(window[:, ch], fs=fs)
                feats.extend(bp)
            X.append(feats)
    else:
        # If data is too small, use the entire data as a single window
        feats = []
        for ch in range(data_filt.shape[1]):
            bp = band_powers(data_filt[:, ch], fs=fs)
            feats.extend(bp)
        X.append(feats)
    
    return np.array(X)

# ============================================================================
# Load Models
# ============================================================================
import glob
import datetime

if not os.path.exists(ALL_MODELS_FILENAME):
    print(f"Error: Models file not found at {ALL_MODELS_FILENAME}")
    exit(1)

print(f"Loading models from: {ALL_MODELS_FILENAME}")
models_data = joblib.load(ALL_MODELS_FILENAME)
all_models = models_data['models']
sorted_results = models_data.get('sorted_results', [])

# Only use models that were actually trained; ExtraTreesClassifier listed first (default)
DEFAULT_MODEL = 'ExtraTreesClassifier'
trained_models = [(name, clf, scaler) for name, (clf, scaler) in all_models.items() if clf is not None]
trained_models.sort(key=lambda x: (0 if x[0] == DEFAULT_MODEL else 1, x[0]))
if not trained_models:
    print("Error: No trained models found in all_models.joblib")
    exit(1)

print(f"Default model: {DEFAULT_MODEL}")
print(f"Loaded {len(trained_models)} model(s): {[n for n, _, _ in trained_models]}\n")

# ============================================================================
# Find all test files
# ============================================================================
test_files = sorted(glob.glob(os.path.join(SAMPLE_DATA_DIR, "*_test_data.csv")))
if not test_files:
    print(f"Error: No *_test_data.csv files found in {SAMPLE_DATA_DIR}")
    exit(1)

print(f"Found {len(test_files)} test file(s): {[os.path.basename(f) for f in test_files]}\n")

# ============================================================================
# Predict on each test file with all models
# ============================================================================
def predict_subject(test_path, clf, scaler):
    df_test = pd.read_csv(test_path)
    test_channels = df_test.columns.tolist()
    data_test = df_test[test_channels].values.astype(float)
    X_feat = extract_features(data_test, test_channels, window_sec=2)
    if scaler is not None:
        X_feat = scaler.transform(X_feat)
    y_pred = clf.predict(X_feat)
    y_pred_proba = clf.predict_proba(X_feat)
    unique, counts = np.unique(y_pred, return_counts=True)
    majority = unique[np.argmax(counts)]
    avg_proba = np.mean(y_pred_proba, axis=0)
    label = "NON-ATTENTIVE" if majority == 1 else "ATTENTIVE"
    confidence = avg_proba[int(majority)] * 100
    return {
        'label': label,
        'confidence': confidence,
        'n_windows': len(y_pred),
        'attentive_windows': int(np.sum(y_pred == 0)),
        'non_attentive_windows': int(np.sum(y_pred == 1)),
        'avg_prob_attentive': avg_proba[0] * 100,
        'avg_prob_non_attentive': avg_proba[1] * 100,
        'max_confidence': np.max(y_pred_proba[:, int(majority)]) * 100,
        'min_confidence': np.min(y_pred_proba[:, int(majority)]) * 100,
        'std_confidence': np.std(y_pred_proba[:, int(majority)]) * 100,
    }

# subject_results: { subject: { model_name: result_dict } }
subject_results = {}
for test_path in test_files:
    fname = os.path.basename(test_path)
    subject = fname.replace('_test_data.csv', '')
    print(f"{'='*60}")
    print(f"Subject: {subject}  ({fname})")
    print(f"{'='*60}")
    subject_results[subject] = {}
    for model_name, clf, scaler in trained_models:
        try:
            res = predict_subject(test_path, clf, scaler)
            subject_results[subject][model_name] = res
            print(f"  [{model_name:<22}]  {res['label']:<16}  confidence: {res['confidence']:.2f}%")
        except Exception as e:
            subject_results[subject][model_name] = None
            print(f"  [{model_name:<22}]  ERROR: {e}")
    print()

# ============================================================================
# Write subject_prediction.results
# ============================================================================
results_path = os.path.join(DATA_DIR, "subject_prediction.results")
model_names = [n for n, _, _ in trained_models]

with open(results_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("EEG SUBJECT PREDICTION RESULTS\n")
    f.write(f"Generated:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Models:     {', '.join(model_names)}\n")
    f.write("=" * 70 + "\n\n")

    for subject, model_results in subject_results.items():
        f.write(f"Subject: {subject}\n")
        f.write("=" * 70 + "\n")
        for model_name in model_names:
            res = model_results.get(model_name)
            f.write(f"  Model: {model_name}\n")
            f.write(f"  {'-'*50}\n")
            if res is None:
                f.write("    ERROR: Failed to produce prediction.\n\n")
                continue
            f.write(f"    Prediction:             {res['label']}\n")
            f.write(f"    Confidence:             {res['confidence']:.2f}%\n")
            f.write(f"    Windows analyzed:       {res['n_windows']}\n")
            f.write(f"    Attentive windows:      {res['attentive_windows']}\n")
            f.write(f"    Non-attentive windows:  {res['non_attentive_windows']}\n")
            f.write(f"    Avg prob (attentive):   {res['avg_prob_attentive']:.2f}%\n")
            f.write(f"    Avg prob (non-attend):  {res['avg_prob_non_attentive']:.2f}%\n")
            f.write(f"    Conf max/min/std:       {res['max_confidence']:.2f}% / {res['min_confidence']:.2f}% / {res['std_confidence']:.2f}%\n\n")
        f.write("\n")

    # Summary table
    f.write("=" * 70 + "\n")
    f.write("SUMMARY\n")
    f.write("=" * 70 + "\n")
    col_w = 16
    header = f"{'Subject':<12}" + "".join(f"  {m[:col_w]:<{col_w}}" for m in model_names)
    f.write(header + "\n")
    f.write("-" * 70 + "\n")
    for subject, model_results in subject_results.items():
        row = f"{subject:<12}"
        for model_name in model_names:
            res = model_results.get(model_name)
            if res:
                cell = f"{res['label'][:9]} {res['confidence']:.0f}%"
            else:
                cell = "ERROR"
            row += f"  {cell:<{col_w}}"
        f.write(row + "\n")
    f.write("=" * 70 + "\n")

print(f"Results saved to: {results_path}")
print("Prediction complete!")
