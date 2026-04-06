import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def generate_vitals_dataset(num_samples=2000):
    """
    Generates a Random Vital Signs Dataset
    In a real-world scenario, datasets like MIMIC-III or device history are used.
    """
    np.random.seed(42)
    data = []
    labels = []
    
    for _ in range(num_samples):
        # 0: Normal, 1: Abnormal / Critical Condition
        is_critical = np.random.choice([0, 1], p=[0.7, 0.3])
        
        if is_critical == 0:
            hr = np.random.normal(75, 10)  # Pulse
            sys_bp = np.random.normal(120, 10) # Systolic
            dia_bp = np.random.normal(80, 8)   # Diastolic
            labels.append(0)
        else:
            # Anomalies in critical conditions
            anomaly_type = np.random.choice(['high_hr', 'low_hr', 'high_bp', 'low_bp'])
            if anomaly_type == 'high_hr':
                hr = np.random.normal(140, 15)
                sys_bp = np.random.normal(130, 15)
                dia_bp = np.random.normal(85, 10)
            elif anomaly_type == 'low_hr':
                hr = np.random.normal(45, 5)
                sys_bp = np.random.normal(110, 10)
                dia_bp = np.random.normal(70, 5)
            elif anomaly_type == 'high_bp':
                hr = np.random.normal(90, 15)
                sys_bp = np.random.normal(180, 20)
                dia_bp = np.random.normal(110, 15)
            else: # low_bp
                hr = np.random.normal(110, 10)
                sys_bp = np.random.normal(85, 10)
                dia_bp = np.random.normal(50, 5)
            labels.append(1)
            
        hr = np.clip(hr, 30, 220)
        sys_bp = np.clip(sys_bp, 70, 250)
        dia_bp = np.clip(dia_bp, 40, 150)
        
        data.append([hr, sys_bp, dia_bp])
        
    df = pd.DataFrame(data, columns=['HeartRate', 'SystolicBP', 'DiastolicBP'])
    df['Label'] = labels
    return df

def train_rf_model():
    print("Generating dataset...")
    df = generate_vitals_dataset(num_samples=5000)
    
    X = df[['HeartRate', 'SystolicBP', 'DiastolicBP']]
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "vitals_rf_model.pkl")
    joblib.dump(rf_model, model_path)
    print(f"Random Forest Sklearn Model saved successfully: {model_path}")
    
    importances = rf_model.feature_importances_
    features = X.columns
    print("\nFeature Importances:")
    for f, imp in zip(features, importances):
        print(f"{f}: {imp:.4f}")

if __name__ == "__main__":
    train_rf_model()
