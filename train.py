import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ultralytics import YOLO
import os

def train_xgboost():
    print("--- Training XGBoost Model for Sensor Pathway ---")
    df = pd.read_csv('smoke_detection_iot.csv')
    
    # Select features usually measured or derived by BME680
    features = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'Pressure[hPa]']
    target = 'Fire Alarm'
    
    X = df[features].copy()
    X.columns = ['Temperature', 'Humidity', 'TVOC', 'Pressure']
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = xgb.XGBClassifier(
        n_estimators=50, # Lightweight model
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    print("Fitting model...")
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"XGBoost Test Accuracy: {acc:.4f}")
    print("Validation Report:")
    print(classification_report(y_test, preds))
    
    model.save_model("xgb_model.json")
    print("XGBoost model saved as xgb_model.json")

def train_yolo():
    print("\n--- Training YOLO Model for Vision Pathway ---")
    data_yaml_path = os.path.abspath('data.yaml')
    
    # Load a pretrained yolov11n pattern (requires ultralytics^8.3)
    # If network is slow to download yolov11n.pt, ultralytics will handle it
    model = YOLO("yolov8n.pt") # Using v8 nano since 'yolo11n' is brand-new and sometimes labeled differently, but I will map to YOLO11 if available. Let's stick with yolov8n.pt for stability, ultralytics supports it flawlessly.
    # Wait, the user specifically mentioned yolo11n. Let's use it.
    model = YOLO("yolo11n.pt") 
    
    # Train the model
    # Note: Using 5 epochs for CPU training to finish in reasonable time, adjust higher for deployment!
    print(f"Training on dataset config: {data_yaml_path}")
    results = model.train(
        data=data_yaml_path,
        epochs=5,
        imgsz=320, # reduced size for faster cpu training
        batch=16,
        project="runs/detect",
        name="fire_detection",
        workers=0 # no multiprocess to prevent spawn issues on windows
    )
    
    print("YOLO training completed.")
    print("Best model weights saved in 'runs/detect/fire_detection/weights/best.pt'")

if __name__ == "__main__":
    train_xgboost()
    print("\n=============================================\n")
    train_yolo()
