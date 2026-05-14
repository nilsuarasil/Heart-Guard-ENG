import numpy as np
import tensorflow as tf

def test_single_ecg():
    # Model path
    model_path = "models/ecg_model.tflite"
    
    # Load TFLite Model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get Model Input and Output Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Let's generate a single "Critical (Abnormal)" ECG Data for testing.
    # In real life, the array data coming from the patient will be placed here.
    # 750 samples, 1 channel
    print("Simulating ECG data coming from a possible patient...")
    test_data = np.random.randn(1, 750, 1).astype(np.float32)
    
    # We add an ST elevation to simulate a risky situation
    test_data[0, 400:450, 0] += 1.5 
    
    # Feed data to the model
    interpreter.set_tensor(input_details[0]['index'], test_data)

    # Run the prediction process
    print("AI (CNN Model) is analyzing the situation...")
    interpreter.invoke()

    # Get the result
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    print("\n--- ANALYSIS RESULT ---")
    print(f"Risk Score (0.0 Normal - 1.0 Critical): {prediction:.4f}")
    
    if prediction > 0.5:
        print("STATUS: CAUTION! Abnormal ECG pattern (e.g. STEMI) detected. Emergency intervention may be required!")
    else:
        print("STATUS: Normal ECG pattern.")

if __name__ == "__main__":
    test_single_ecg()
