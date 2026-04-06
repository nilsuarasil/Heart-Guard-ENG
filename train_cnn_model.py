import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import os
import json

def build_ecg_cnn_model(input_shape=(750, 1)):
    """
    A simple 1D CNN Model (for ECG Signals)
    Input: (Number of Samples, Number of Channels) -> e.g.: 3 second 250Hz signal = 750 samples
    Output: Binary Classification (0: Normal, 1: Anomaly/STEMI)
    """
    model = Sequential([
        # 1. Convolution Block
        Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        
        # 2. Convolution Block
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # 3. Convolution Block
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        # Fully Connected (Dense) Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Prevent overfitting
        Dense(1, activation='sigmoid') # Binary classification
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def create_dummy_dataset(num_samples=1000, input_length=750):
    """
    Generates random synthetic ECG data for training.
    In a real scenario, this would be fed from the MIT-BIH dataset.
    """
    print(f"Generating {num_samples} synthetic training data samples...")
    X = np.random.randn(num_samples, input_length, 1)
    
    # Labels: Half normal(0), half anomalous(1)
    y = np.random.randint(0, 2, size=(num_samples, 1))
    
    for i in range(num_samples):
        if y[i] == 1:
            X[i, 400:450, 0] += np.random.uniform(0.5, 1.5) # ST Elevation effect
            
    return X, y

def train_and_export_model():
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Parameters
    input_length = 750 # 3 seconds * 250Hz
    
    # Prepare data
    X_train, y_train = create_dummy_dataset(num_samples=2000, input_length=input_length)
    X_val, y_val = create_dummy_dataset(num_samples=400, input_length=input_length)
    
    # Build model
    print("Building CNN Model...")
    model = build_ecg_cnn_model(input_shape=(input_length, 1))
    model.summary()
    
    # Train
    print("Model training begins...")
    history = model.fit(X_train, y_train, 
                        epochs=5, 
                        batch_size=32, 
                        validation_data=(X_val, y_val),
                        verbose=1)
    
    # Save as Keras format
    keras_model_path = os.path.join(model_dir, "ecg_cnn_model.h5")
    model.save(keras_model_path)
    print(f"Keras model saved to: {keras_model_path}")
    
    # Convert to TensorFlow Lite format for mobile/edge
    print("Converting model to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_model_path = os.path.join(model_dir, "ecg_model.tflite")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model saved to: {tflite_model_path}")
    
    # Save training history (optional)
    history_dict = history.history
    for k in history_dict:
        history_dict[k] = [float(val) for val in history_dict[k]]
        
    with open(os.path.join(model_dir, "training_history.json"), 'w') as f:
        json.dump(history_dict, f)
        
    return tflite_model_path

if __name__ == "__main__":
    train_and_export_model()
