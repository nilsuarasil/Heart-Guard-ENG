import numpy as np
import tensorflow as tf
import wfdb
import os

def test_with_mitbih():
    model_path = "models/ecg_model.tflite"
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Downloading real patient data from MIT-BIH Arrhythmia Database (Record: 100)...")
    
    # We download record 100 directly from PhysioNet via wfdb.rdrecord
    # Let's take only the first 3 seconds of data (Sampling frequency 360Hz)
    # 3 x 360 = 1080 samples.
    # But our model was trained with 750 samples (3 sec * 250Hz) (On Mock data)
    # In a real-world scenario 'resampling' should be done.
    # For now, we will just cut to 750 samples to fit what the model expects.
    
    # It will download from the internet if not already downloaded
    record = wfdb.rdrecord('100', sampto=1000, pn_dir='mitdb')
    
    # record.p_signal contains the signal (e.g.: [1000 samples, 2 channels])
    # Let's take the MLII channel (Usually index 0).
    ecg_signal = record.p_signal[:, 0]
    
    # Since the model input shape is (750, 1), we take the first 750 samples
    test_data = ecg_signal[:750].reshape(1, 750, 1).astype(np.float32)
    
    # Start prediction for the model
    interpreter.set_tensor(input_details[0]['index'], test_data)
    
    print("\nAI (CNN Model) is analyzing Real Patient (Record 100 - Normal Rhythm) data...")
    interpreter.invoke()
    
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    print("\n--- ANALYSIS RESULT ---")
    print(f"Risk Score (0.0 Normal - 1.0 Critical): {prediction:.4f}")
    
    if prediction > 0.5:
        print("STATUS: CAUTION! Abnormal ECG pattern (e.g. STEMI) detected. Emergency intervention may be required!")
    else:
        print("STATUS: Normal ECG pattern.")

    
    # Second Patient Example (Record: 200 - Patient with frequent Ventricular Arrhythmias)
    print("\n------------------------------------------------------")
    print("Downloading at-risk patient data from MIT-BIH Arrhythmia Database (Record: 200)...")
    record_risk = wfdb.rdrecord('200', sampto=1000, pn_dir='mitdb')
    ecg_signal_risk = record_risk.p_signal[:, 0]
    test_data_risk = ecg_signal_risk[:750].reshape(1, 750, 1).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_data_risk)
    
    print("AI (CNN Model) is analyzing Real Patient (Record 200 - Arrhythmic Rhythm) data...")
    interpreter.invoke()
    
    prediction_risk = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    print("\n--- ANALYSIS RESULT ---")
    print(f"Risk Score (0.0 Normal - 1.0 Critical): {prediction_risk:.4f}")
    
    if prediction_risk > 0.5:
        print("STATUS: CAUTION! Abnormal ECG pattern (e.g. STEMI) detected. Emergency intervention may be required!")
    else:
        print("STATUS: Normal ECG pattern.")

if __name__ == "__main__":
    test_with_mitbih()
