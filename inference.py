import time
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from ultralytics import YOLO
import cv2
import warnings

# We try to import Raspberry Pi libraries, but wrap them in try-except 
# so the script can at least be syntax-checked on non-Pi platforms.
try:
    import board
    import busio
    import adafruit_bme680
    from gpiozero import Buzzer
    RPI_MODE = True
except ImportError:
    print("Warning: Raspberry Pi specific libraries (adafruit_bme680, gpiozero, board) not found.")
    print("Running in mocked inference mode.")
    RPI_MODE = False

warnings.filterwarnings('ignore')

# ----------------------------
# 1. Configuration Constants
# ----------------------------
YOLO_CONF_THRESHOLD = 0.45
XGB_ANOMALY_THRESHOLD = 0.5
BUZZER_PIN = 17 # GPIO17 for buzzer

def load_models():
    print("Loading models...")
    # Load XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("xgb_model.json")
    
    # Load YOLO
    yolo_model = YOLO("runs/detect/fire_detection/weights/best.pt")
    
    print("Models loaded successfully.")
    return xgb_model, yolo_model

def setup_hardware():
    if not RPI_MODE:
        return None, None, None
        
    print("Initializing I2C and BME680 sensor...")
    i2c = busio.I2C(board.SCL, board.SDA)
    bme680 = adafruit_bme680.Adafruit_BME680_I2C(i2c)
    # Set sea-level pressure for altitude calculation if needed
    bme680.sea_level_pressure = 1013.25
    
    print(f"Initializing Buzzer on GPIO {BUZZER_PIN}...")
    alarm_buzzer = Buzzer(BUZZER_PIN)
    
    print("Initializing Camera...")
    # Camera index 0 is usually the standard PiCamera via V4L2
    cap = cv2.VideoCapture(0)
    # Set lower resolution for much faster inference
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    return bme680, alarm_buzzer, cap

def estimate_tvoc(gas_resistance_ohms):
    """
    Very crude estimation of TVOC from gas resistance if you don't use BSEC.
    High gas resistance = clean air. Low gas resistance = polluted/fire.
    In a real implementation, you'd map this using a baseline or use Bosch's BSEC library!
    """
    baseline_gas = 50000.0 # Just a hypothetical baseline
    if gas_resistance_ohms <= 0:
         return 0
    ratio = baseline_gas / gas_resistance_ohms
    tvoc_estimate = (ratio - 1) * 100 
    return max(0, tvoc_estimate)

def main():
    xgb_model, yolo_model = load_models()
    bme680, alarm_buzzer, cap = setup_hardware()
    
    print("\n--- Edge Fire Detection System Active ---\n")
    
    try:
        while True:
            # ---------------------------------
            # STEP 1: Sensor Pathway (XGBoost)
            # ---------------------------------
            if RPI_MODE:
                temp = bme680.temperature
                humidity = bme680.relative_humidity
                pressure = bme680.pressure
                gas_res = bme680.gas
                # Estimate TVOC since the sensor only gives resistance directly via adafruit lib
                tvoc = estimate_tvoc(gas_res)
            else:
                # Mock values for testing on desktop
                temp, humidity, pressure, tvoc = 25.0, 50.0, 1013.0, 0.0
                
            # Create a dataframe slice for xgboost prediction
            # Features must match the training: ['Temperature', 'Humidity', 'TVOC', 'Pressure']
            sensor_data = pd.DataFrame([{
                'Temperature': temp,
                'Humidity': humidity,
                'TVOC': tvoc,
                'Pressure': pressure
            }])
            
            # Predict Anomaly Probability
            xgb_prob = xgb_model.predict_proba(sensor_data)[0][1] # Probability of Fire (class 1)
            sensor_alert = (xgb_prob > XGB_ANOMALY_THRESHOLD)
            
            # ---------------------------------
            # STEP 2: Vision Pathway (YOLO)
            # ---------------------------------
            yolo_alert = False
            if RPI_MODE:
                ret, frame = cap.read()
                if ret:
                    # Run inference on frame
                    # Classes: typically 'fire' or 'smoke', you can filter by classes if needed
                    results = yolo_model.predict(source=frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
                    
                    # Check if bounding boxes exist
                    if len(results[0].boxes) > 0:
                        yolo_alert = True
                else:
                    print("Camera frame read failed!")
            else:
                # Mock yolo alert status for desktop testing
                yolo_alert = False
                
            # ---------------------------------
            # STEP 3: Late Fusion Engine
            # ---------------------------------
            print(f"BME680: Temp={temp:.1f}C, Hum={humidity:.1f}%, TVOC={tvoc:.1f} | XGB Prob: {xgb_prob:.2f} | YOLO Alert: {yolo_alert}")
            
            if sensor_alert and yolo_alert:
                print(">>> ALARM TRIGGERED! BOTH SENSORS DETECT FIRE/SMOKE <<<")
                if RPI_MODE:
                    alarm_buzzer.on()
            elif sensor_alert or yolo_alert:
                print(">>> WARNING! ONE SENSOR DETECTED ANOMALY - VERIFYING... <<<")
                if RPI_MODE:
                    alarm_buzzer.off()
            else:
                if RPI_MODE:
                    alarm_buzzer.off()
                    
            # Small delay to keep inference loop stable and not overwhelm CPU
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nExiting system...")
    finally:
        if RPI_MODE:
            cap.release()
            alarm_buzzer.off()

if __name__ == "__main__":
    main()
