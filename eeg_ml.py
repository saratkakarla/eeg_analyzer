import numpy as np
import pandas as pd
import os
import sys
from scipy.signal import welch, butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, recall_score

import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Command-line Arguments
# ============================================================================
AVAILABLE_MODELS = [
    'Random Forest', 'Gradient Boosting',
    'Neural Network', 'ExtraTreesClassifier',
    'all'  # Train all models
]

# Model shortcuts
MODEL_SHORTCUTS = {
    'rf': 'Random Forest',
    'gb': 'Gradient Boosting',
    'nn': 'Neural Network',
    'et': 'ExtraTreesClassifier',
}

def print_usage():
    print("Usage: python eeg_ml.py [model_name]")
    print("\nAvailable models:")
    for model in AVAILABLE_MODELS[:-1]:
        print(f"  - {model}")
    print(f"  - {AVAILABLE_MODELS[-1]} (trains all models)")
    print("\nModel shortcuts:")
    for shortcut, full_name in MODEL_SHORTCUTS.items():
        print(f"  - {shortcut:3s} → {full_name}")
    print("\nExample:")
    print("  python eeg_ml.py gb        # Train Gradient Boosting")
    print("  python eeg_ml.py et        # Train ExtraTreesClassifier")
    print("  python eeg_ml.py all       # Train all models")
    print("  python eeg_ml.py           # Default: ExtraTreesClassifier")

# Check for model argument
selected_model = None
if len(sys.argv) > 1:
    model_arg = sys.argv[1]
    # Check if it's a shortcut
    if model_arg in MODEL_SHORTCUTS:
        selected_model = MODEL_SHORTCUTS[model_arg]
    elif model_arg in AVAILABLE_MODELS:
        selected_model = model_arg
    else:
        print(f"Error: Unknown model '{model_arg}'")
        print_usage()
        exit(1)
    print(f"Training model: {selected_model}")
else:
    selected_model = 'ExtraTreesClassifier'
    print(f"No model specified. Using default: {selected_model}")
    print_usage()

REFERENCE_DATA_PATH = "data/reference_data.csv"
OUT_FILENAME_PIEEG = "data/cleaned_reference_data.csv"
OUT_FILENAME_MODEL = "data/reference_model.joblib"

DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # Current script directory
FULL_FILENAME = os.path.join(DATA_DIR, REFERENCE_DATA_PATH)  # Change to your actual filename
OUTPUT_DIR = DATA_DIR  # Output directory for processed files

# Load raw data
if not os.path.exists(FULL_FILENAME):
    print(f"Error: File not found at {FULL_FILENAME}")
    print(f"Current directory: {DATA_DIR}")
    exit(1)

df_full = pd.read_csv(FULL_FILENAME)
print(f"Loaded data shape: {df_full.shape}")
print(df_full.head())


#This code actually converts 19 Channel data into 16 Channel Data
# Channels actually present in the PiEEG-16 cap
pieeg_channels_order = [
    "Fp1", "Fp2",    # frontopolar
    "F7", "F3", "F4", "F8",   # frontal
    "C3", "C4",               # central
    "P3", "P4",               # parietal
    "T7", "T8",               # temporal
    "P7", "P8",               # parietal-temporal
    "O1", "O2"                # occipital
]

# Build new DataFrame in PiEEG-16 order + labels
df_pieeg16 = df_full[pieeg_channels_order + ["Class", "ID"]]

print("\nProcessed data (PiEEG-16 channels):")
print(df_pieeg16.head())

OUT_FILENAME = os.path.join(OUTPUT_DIR, OUT_FILENAME_PIEEG)
df_pieeg16.to_csv(OUT_FILENAME, index=False)
print(f"Saved: {OUT_FILENAME}")

# ============================================================================
# Feature Extraction Functions
# ============================================================================
FS = 128  # Sampling frequency in Hz

# Load processed dataset
PROCESSED_FILENAME = os.path.join(OUTPUT_DIR, OUT_FILENAME_PIEEG)
if not os.path.exists(PROCESSED_FILENAME):
    print(f"Error: Processed file not found at {PROCESSED_FILENAME}")
    print("Please ensure the data processing steps above have been run.")
    exit(1)

df = pd.read_csv(PROCESSED_FILENAME)

# Reorder columns to match PiEEG-16 layout
df = df[pieeg_channels_order + ["Class", "ID"]]
print(f"\nLoaded processed data shape: {df.shape}")
print(df.head())


#These lines of code filter out the unnecessary noises
#and make sure that only the important parts of the brainwaves are recorded.
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=FS, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)


#Think of BANDS as a power meter
BANDS = {
    "delta": (1, 4), #sleepy
    "theta": (4, 8), #dreamy
    "alpha": (8, 13), #relaxed
    "beta":  (13, 30), #thinking
}

def band_powers(signal, fs=FS):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    feats = []
    for low, high in BANDS.values():
        idx = np.logical_and(freqs >= low, freqs <= high)
        feats.append(np.trapezoid(psd[idx], freqs[idx]))
    return np.array(feats)


# ============================================================================
# Feature Matrix Construction
# ============================================================================
def extract_features(df, channel_cols, window_sec=2):
    """
    Extract features from EEG data using windowing and band power analysis.
    
    Args:
        df: DataFrame with EEG channels and metadata
        channel_cols: List of channel column names
        window_sec: Window size in seconds
    
    Returns:
        X, y, subject_ids: Feature matrix, labels, and subject identifiers
    """
    window_size = window_sec * FS
    
    X = []
    y = []
    subject_ids = []
    
    for subj_id, df_sub in df.groupby("ID"):
        label_str = df_sub["Class"].iloc[0]
        label = 1 if label_str == "non-attentive" else 0
    
        data = df_sub[channel_cols].values.astype(float)
    
        # filter; Clean all the low and high noise
        # filter super low rumbly noises and the super high squeaky noises
        data_filt = bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=FS, order=4)
    
        n_samples = data_filt.shape[0]
        n_windows = n_samples // window_size
    
        for w in range(n_windows):
            start = w * window_size
            end = start + window_size
            window = data_filt[start:end, :]
    
            feats = []
            for ch in range(window.shape[1]):
                bp = band_powers(window[:, ch], fs=FS)
                feats.extend(bp)
            X.append(feats)
            y.append(label)
            subject_ids.append(subj_id)
    
    return np.array(X), np.array(y), subject_ids


window_sec = 2
channel_cols = pieeg_channels_order

print("\nExtracting features from EEG data (this may take a few minutes)...")
print("Sampling data for faster processing (every 10th sample)...")

# Sample the data to speed up feature extraction
df_sampled = df.iloc[::10].copy()
print(f"Sampled data shape: {df_sampled.shape}")

X, y, subject_ids = extract_features(df_sampled, channel_cols, window_sec=window_sec)
print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")


# ============================================================================
# Model Training and Evaluation
# ============================================================================
print("\nSplitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features (needed for Neural Networks)
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store models and their results
models = {}
results = {}
recalls = {}

# ============================================================================
# Model 1: Random Forest (optimized)
# ============================================================================
if selected_model == 'all' or selected_model == 'Random Forest':
    print("\n" + "="*60)
    print("Training Random Forest Classifier (optimized)...")
    print("="*60)
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    models['Random Forest'] = (rf_clf, None)
    results['Random Forest'] = rf_accuracy
    recalls['Random Forest'] = recall_score(y_test, rf_pred)
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(classification_report(y_test, rf_pred, target_names=["attentive", "non-attentive"]))
else:
    models['Random Forest'] = (None, None)
    results['Random Forest'] = None

# ============================================================================
# Model 2: Gradient Boosting
# ============================================================================
if selected_model == 'all' or selected_model == 'Gradient Boosting':
    print("\n" + "="*60)
    print("Training Gradient Boosting Classifier...")
    print("="*60)
    gb_clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        verbose=1
    )
    gb_clf.fit(X_train, y_train)
    gb_pred = gb_clf.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    models['Gradient Boosting'] = (gb_clf, None)
    results['Gradient Boosting'] = gb_accuracy
    recalls['Gradient Boosting'] = recall_score(y_test, gb_pred)
    print(f"Accuracy: {gb_accuracy:.4f}")
    print(classification_report(y_test, gb_pred, target_names=["attentive", "non-attentive"]))
else:
    models['Gradient Boosting'] = (None, None)
    results['Gradient Boosting'] = None

# ============================================================================
# Model 3: Neural Network (MLP)
# ============================================================================
if selected_model == 'all' or selected_model == 'Neural Network':
    print("\n" + "="*60)
    print("Training Neural Network (MLP)...")
    print("="*60)
    nn_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        verbose=1,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn_clf.fit(X_train_scaled, y_train)
    nn_pred = nn_clf.predict(X_test_scaled)
    nn_accuracy = accuracy_score(y_test, nn_pred)
    models['Neural Network'] = (nn_clf, scaler)
    results['Neural Network'] = nn_accuracy
    recalls['Neural Network'] = recall_score(y_test, nn_pred)
    print(f"Accuracy: {nn_accuracy:.4f}")
    print(classification_report(y_test, nn_pred, target_names=["attentive", "non-attentive"]))
else:
    models['Neural Network'] = (None, None)
    results['Neural Network'] = None

# ============================================================================
# Model 4: Extra Trees
# ============================================================================
if selected_model == 'all' or selected_model == 'ExtraTreesClassifier':
    print("\n" + "="*60)
    print("Training Extremely Randomized Trees...")
    print("="*60)
    et_clf = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    et_clf.fit(X_train, y_train)
    et_pred = et_clf.predict(X_test)
    et_accuracy = accuracy_score(y_test, et_pred)
    models['ExtraTreesClassifier'] = (et_clf, None)
    results['ExtraTreesClassifier'] = et_accuracy
    recalls['ExtraTreesClassifier'] = recall_score(y_test, et_pred)
    print(f"Accuracy: {et_accuracy:.4f}")
    print(classification_report(y_test, et_pred, target_names=["attentive", "non-attentive"]))
else:
    models['ExtraTreesClassifier'] = (None, None)
    results['ExtraTreesClassifier'] = None

# ============================================================================
# Results Summary
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)
print(f"{'#':<3} {'Model':<22} {'Accuracy':>10} {'Recall':>10}  {'Status'}")
print("-"*70)
# Filter out None results (models that weren't trained)
valid_results = {k: v for k, v in results.items() if v is not None}
sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
for i, (model_name, accuracy) in enumerate(sorted_results, 1):
    recall = recalls.get(model_name, float('nan'))
    status = "✓ >80%" if accuracy > 0.80 else ""
    print(f"{i:<3} {model_name:<22} {accuracy:>10.4f} {recall:>10.4f}  {status}")
print("="*70)

# Select the best model
if len(sorted_results) == 0:
    print("Error: No models were trained. Please check your input.")
    exit(1)

best_model_name = sorted_results[0][0]
best_model, best_scaler = models[best_model_name]
best_accuracy = sorted_results[0][1]

print(f"\nBest Model: {best_model_name} with {best_accuracy:.4f} accuracy")


# ============================================================================
# Model Saving - Save All Models and the Best Model
# ============================================================================
# Save the best model
MODEL_FILENAME = os.path.join(OUTPUT_DIR, OUT_FILENAME_MODEL)
joblib.dump((best_model, best_scaler), MODEL_FILENAME)
print(f"\nBest model ({best_model_name}) and scaler saved to: {MODEL_FILENAME}")

# Save all models for comparison testing
all_models_filename = os.path.join(OUTPUT_DIR, "all_models.joblib")
models_data = {
    'models': models,
    'results': results,
    'recalls': recalls,
    'sorted_results': sorted_results
}
joblib.dump(models_data, all_models_filename)
print(f"All models saved to: {all_models_filename}")

# Gather dataset statistics for the report
class_counts = df['Class'].value_counts()
n_subjects = df['ID'].nunique()

# Save model comparison results to file
import datetime
results_filename = os.path.join(OUTPUT_DIR, "eeg_models_accuracy.results")
with open(results_filename, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("EEG ATTENTION CLASSIFICATION — MODEL REPORT\n")
    f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n")

    f.write("\nDATA SUMMARY\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Raw samples (19 ch):        {df_full.shape[0]:>10,}\n")
    f.write(f"  Processed samples (16 ch):  {df.shape[0]:>10,}\n")
    f.write(f"  Sampled (every 10th):       {df_sampled.shape[0]:>10,}\n")
    f.write(f"  EEG channels:               {len(channel_cols):>10}\n")
    f.write(f"  Sampling frequency:         {FS:>9} Hz\n")
    f.write(f"  Window size:                {window_sec:>9} sec\n")
    f.write(f"  Subjects examined:          {n_subjects:>10}\n")
    f.write("\n  Class distribution (raw data):\n")
    for cls, count in class_counts.items():
        pct = count / df.shape[0] * 100
        f.write(f"    {cls:<20} {count:>10,}  ({pct:.1f}%)\n")

    f.write("\nFEATURE EXTRACTION\n")
    f.write("-" * 70 + "\n")
    f.write(f"  Total windows extracted:    {X.shape[0]:>10}\n")
    f.write(f"  Features per window:        {X.shape[1]:>10}\n")
    f.write(f"  Training samples:           {X_train.shape[0]:>10}\n")
    f.write(f"  Test samples:               {X_test.shape[0]:>10}\n")
    f.write(f"  Test split:                 {'20%':>10}\n")

    f.write("\nMODEL COMPARISON RESULTS\n")
    f.write("=" * 70 + "\n")
    f.write(f"{'#':<3} {'Model':<22} {'Accuracy':>10} {'Recall':>10}  {'Status'}\n")
    f.write("-" * 70 + "\n")
    for i, (model_name, accuracy) in enumerate(sorted_results, 1):
        recall = recalls.get(model_name, float('nan'))
        status = "✓ >80%" if accuracy > 0.80 else ""
        f.write(f"{i:<3} {model_name:<22} {accuracy:>10.4f} {recall:>10.4f}  {status}\n")
    f.write("=" * 70 + "\n")
    f.write(f"\nBest Model: {best_model_name} with {best_accuracy:.4f} accuracy\n")
print(f"Model comparison results saved to: {results_filename}")

