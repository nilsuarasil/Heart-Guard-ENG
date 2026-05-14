import numpy as np
import pandas as pd
import time
import math

def generate_ecg_signal(duration_sec=30, sampling_rate=250, heart_rate=75, is_anomaly=False):
    """
    Generates a simple synthetic ECG signal.
    duration_sec: Length of the signal in seconds.
    sampling_rate: Number of samples per second (Hz).
    heart_rate: Number of beats per minute (BPM).
    is_anomaly: If True, adds a STEMI-like ST elevation.
    """
    t = np.linspace(0, duration_sec, int(duration_sec * sampling_rate))
    
    # Heart beat frequency
    beat_frequency = heart_rate / 60.0
    
    # Basic ECG components (Simplified signal for P, Q, R, S, T waves)
    # R wave (Sharp peak)
    r_wave = np.sin(2 * np.pi * beat_frequency * t) ** 40 
    
    # P wave (Small, before R)
    p_wave = 0.1 * np.sin(2 * np.pi * beat_frequency * (t + 0.15)) ** 10
    
    # T wave (After R)
    t_wave = 0.2 * np.sin(2 * np.pi * beat_frequency * (t - 0.25)) ** 10
    
    # Basic signal combination
    ecg_signal = r_wave + p_wave + t_wave
    
    if is_anomaly:
        # STEMI Anomaly Simulation: ST segment elevation
        # We elevate the segment right before the T wave
        st_elevation = 0.3 * np.sin(2 * np.pi * beat_frequency * (t - 0.1)) ** 8
        ecg_signal += st_elevation
        
    # Adding noise (Baseline wander and high-frequency tiny noise)
    noise = 0.05 * np.random.randn(len(t))
    baseline_wander = 0.1 * np.sin(2 * np.pi * 0.1 * t)
    
    ecg_signal = ecg_signal + noise + baseline_wander
    
    return t, ecg_signal

def generate_vitals(is_critical=False):
    """
    Generates pulse and blood pressure (Systolic/Diastolic) data.
    """
    if not is_critical:
        hr = np.random.randint(60, 100)
        sys_bp = np.random.randint(110, 130)
        dia_bp = np.random.randint(70, 85)
    else:
        # Data that will trigger an emergency (e.g., excessively high pulse or BP)
        hr = np.random.choice([np.random.randint(40, 50), np.random.randint(120, 160)])
        sys_bp = np.random.randint(160, 200)
        dia_bp = np.random.randint(90, 120)
        
    return hr, sys_bp, dia_bp

if __name__ == "__main__":
    print("--- Normal Data Generation ---")
    hr, sys, dia = generate_vitals(is_critical=False)
    print(f"Heart Rate: {hr} BPM, BP: {sys}/{dia} mmHg")
    t_norm, ecg_norm = generate_ecg_signal(duration_sec=3, is_anomaly=False)
    print(f"Normal ECG Signal Generated: {len(ecg_norm)} samples.")
    
    print("\n--- Critical (Emergency) Data Generation ---")
    hr_crit, sys_crit, dia_crit = generate_vitals(is_critical=True)
    print(f"Heart Rate: {hr_crit} BPM, BP: {sys_crit}/{dia_crit} mmHg")
    t_anom, ecg_anom = generate_ecg_signal(duration_sec=3, is_anomaly=True)
    print(f"Abnormal ECG Signal Generated: {len(ecg_anom)} samples.")
    
    # Option to save data as CSV can be added
    # pd.DataFrame({'Time': t_anom, 'ECG': ecg_anom}).to_csv('mock_abnormal_ecg.csv', index=False)
