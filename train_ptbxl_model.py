"""
PTB-XL Real Clinical Dataset - CNN Model Retraining
-----------------------------------------------------------------
PTB-XL: 10-second 12-lead ECG records from 21,837 patients.
Label: NORM (normal) or MI (Myocardial Infarction / STEMI / NSTEMI)

Data Source: https://physionet.org/content/ptb-xl/1.0.3/
"""

import os
import json
import numpy as np
import pandas as pd
import wfdb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ─── Settings ─────────────────────────────────────────────────────────────────
PTBXL_PN_DIR   = "ptb-xl"          # PhysioNet short name
SAMPLE_RATE    = 100               # 100Hz or 500Hz
SIGNAL_SECONDS = 7                 # First 7 seconds (700 samples @ 100Hz)
N_SAMPLES      = 700               # Model input length
N_NORM         = 300               # Normal records to download
N_MI           = 300               # MI records to download
MODEL_DIR      = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

def download_ptbxl_metadata():
    """Downloads PTB-XL metadata CSV from PhysioNet."""
    meta_path = "ptbxl_database.csv"
    if os.path.exists(meta_path):
        print("Metadata already exists, skipping download.")
        return pd.read_csv(meta_path, index_col="ecg_id")
    
    print("Downloading PTB-XL metadata CSV...")
    wfdb.dl_files('ptb-xl', '.', ['ptbxl_database.csv'])
    df = pd.read_csv(meta_path, index_col='ecg_id')
    print(f"Total records: {len(df)}")
    return df

def parse_scp_codes(row):
    """Parses scp_codes column to check if MI or NORM is present."""
    try:
        codes = json.loads(row.replace("'", '"'))
        has_mi   = any(k in codes for k in ['MI', 'AMI', 'IMI', 'ASMI', 'ILMI', 'ALMI', 'INJAS', 'INJIN', 'INJIL', 'INJLA', 'WPW'])
        has_norm = 'NORM' in codes
        return 'MI' if has_mi else ('NORM' if has_norm else 'OTHER')
    except Exception:
        return 'OTHER'

def load_record(filename_lr):
    """Downloads a single PTB-XL record and returns Lead II signal."""
    try:
        record_path = filename_lr.replace('.hea', '')
        record_path = record_path.lstrip('/')
        record = wfdb.rdrecord(record_path, sampto=N_SAMPLES, pn_dir=PTBXL_PN_DIR)
        sig = record.p_signal[:N_SAMPLES, 1].astype(np.float32) # Lead II is index 1
        if len(sig) < N_SAMPLES:
            sig = np.pad(sig, (0, N_SAMPLES - len(sig)))
        return sig
    except Exception as e:
        return None

def build_model(input_shape=(N_SAMPLES, 1)):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 7, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(64, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # 0 = NORM, 1 = MI
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def train_and_export():
    df = download_ptbxl_metadata()
    df['label'] = df['scp_codes'].apply(parse_scp_codes)

    norm_records = df[df['label'] == 'NORM'].head(N_NORM)
    mi_records   = df[df['label'] == 'MI'].head(N_MI)
    
    print(f"Normal records: {len(norm_records)}, MI records: {len(mi_records)}")

    X, y = [], []

    print("Downloading Normal records...")
    for i, (ecg_id, row) in enumerate(norm_records.iterrows()):
        sig = load_record(row['filename_lr'])
        if sig is not None:
            X.append(sig)
            y.append(0)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(norm_records)} normal completed")

    print("Downloading MI (STEMI/NSTEMI) records...")
    for i, (ecg_id, row) in enumerate(mi_records.iterrows()):
        sig = load_record(row['filename_lr'])
        if sig is not None:
            X.append(sig)
            y.append(1)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(mi_records)} MI completed")

    X = np.array(X).reshape(-1, N_SAMPLES, 1)
    y = np.array(y)

    print(f"\nTotal data: {len(X)} | Normal: {np.sum(y==0)} | MI: {np.sum(y==1)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nTraining model (real clinical dataset)...")
    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_auc', mode='max'),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss')
    ]

    model.fit(X_train, y_train,
              epochs=30,
              batch_size=32,
              validation_data=(X_val, y_val),
              callbacks=callbacks,
              verbose=1)

    keras_path = os.path.join(MODEL_DIR, "ecg_ptbxl_model.h5")
    model.save(keras_path)
    print(f"\nKeras model saved successfully: {keras_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = os.path.join(MODEL_DIR, "ecg_ptbxl_model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved successfully: {tflite_path}")
    print(f"\nTo plot: python plot_ecg.py --model {tflite_path}")

    return tflite_path

if __name__ == "__main__":
    train_and_export()
