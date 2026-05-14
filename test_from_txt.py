import numpy as np
import tensorflow as tf
import os
import sys

def test_from_txt(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: '{file_path}' not found. Please ensure your TXT file is here.")
        sys.exit(1)

    print(f"Reading data from '{file_path}'...")
    
    try:
        # Read data from Txt file.
        # Assumption: Each line has an ECG voltage (number) value, e.g.: -0.15, 0.45, 1.2 etc.
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Clean and split if separated by commas etc., then convert to float array
        data = []
        for line in lines:
            line = line.strip()
            if not line or "sure" in line.lower() or "genlik" in line.lower() or "time" in line.lower() or "amplitude" in line.lower():
                continue # Skip header lines or empty lines
            
            # Standardize comma or dot
            parts = line.replace(',', ' ').split()
            
            if len(parts) >= 2:
                # Format: 0.000, 0.062 (Time, Amplitude)
                # Take the second column (Amplitude/mV)
                val_str = parts[1]
            else:
                # Format: only amplitude value is present
                val_str = parts[0]
                
            try:
                data.append(float(val_str))
            except ValueError:
                pass # Skip if there is text etc.
                    
        signal = np.array(data)
        
    except Exception as e:
        print(f"Error occurred while reading data: {e}")
        sys.exit(1)
        
    print(f"Found {len(signal)} numerical matches.")
    
    if len(signal) == 0:
         print("ERROR: No readable valid numerical ECG data found in the file.")
         sys.exit(1)

    # Our model expects 750 length data
    # Let's adjust the data to the appropriate size for the model (Pad or Cut)
    target_length = 750
    if len(signal) > target_length:
        print(f"Warning: The data you entered is longer than 750 samples. Only the first 750 will be considered by the AI.")
        signal = signal[:target_length]
    elif len(signal) < target_length:
         print(f"Warning: The data you entered is shorter than 750 samples. It will be padded with zeros for the AI.")
         pad_length = target_length - len(signal)
         signal = np.pad(signal, (0, pad_length), 'constant', constant_values=(0, 0))
         
    # Adapt to model shape: (1, 750, 1)
    test_data = signal.reshape(1, 750, 1).astype(np.float32)

    # -- Load Model and Predict --
    model_path = "models/ecg_model.tflite"
    if not os.path.exists(model_path):
        print("ERROR: Model file not found. train_cnn_model.py must be run first.")
        sys.exit(1)
        
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], test_data)

    print("\nAI (CNN Model) is analyzing your txt data...")
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    print("\n--- ANALYSIS RESULT ---")
    print(f"Risk Score (0.000 Normal - 1.000 Critical): {prediction:.4f}")

    if prediction > 0.5:
        print("STATUS: CAUTION! Abnormal ECG pattern (e.g. STEMI) detected. Emergency intervention may be required!")
    else:
        print("STATUS: Normal ECG pattern.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage Error.")
        print("Correct Usage: python test_from_txt.py your_file.txt")
    else:
        file_to_test = sys.argv[1]
        test_from_txt(file_to_test)
