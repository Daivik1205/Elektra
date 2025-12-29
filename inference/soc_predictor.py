import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from utils.dvdq_features import extract_dvdq_features

# Safe loading
MODEL_PATH = "models/elektra_soc_lstm.keras"
SOC_MODEL = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ---- TUNING 1: STABLE SCALER ----
# We pre-fit the scaler on realistic bounds so it doesn't jump around
# based on the small window of data.
scaler = StandardScaler()
# Mock fit on expected range: Voltage(3.0-4.2), Current(-5, 5), Temp(20-60)
scaler.fit([
    [3.0, -5.0, 20.0],
    [4.2,  5.0, 60.0]
])

# Load Physics Bias
try:
    ANODE = extract_dvdq_features("dv_dq_anode.csv", "anode")
    CATHODE = extract_dvdq_features("dv_dq_cathode.csv", "cathode")
except:
    ANODE = {"anode_dvdq_mean": 0}
    CATHODE = {"cathode_dvdq_mean": 0}

# ---- TUNING 2: SMOOTHING STATE ----
# This variable remembers the previous value to prevent jumps
_prev_soc = None

def predict_soc(df):
    global _prev_soc
    
    # Get the latest sensor readings
    v = df["voltage"].iloc[-1]
    i = df["current"].iloc[-1]
    
    # 1. PREDICT RAW SOC
    if SOC_MODEL is None:
        # ---- FINE TUNE: PHYSICS NOISE CANCELLATION ----
        # Raw voltage dips when you accelerate (V = OCV - I*R).
        # We estimate the true chemical voltage (OCV) by adding back the drop.
        # We assume internal resistance R approx 0.05 Ohms.
        estimated_ocv = v + (i * 0.05)
        
        # Convert OCV to SOC (Linear approx for demo)
        soc = (estimated_ocv - 3.0) / 1.2 * 100
    else:
        # AI Prediction Path
        features = df[["voltage", "current", "temperature"]].values
        # Use transform, not fit_transform (keeps scale stable)
        features_scaled = scaler.transform(features)
        X = features_scaled.reshape(1, features_scaled.shape[0], features_scaled.shape[1])
        
        soc_raw = SOC_MODEL.predict(X, verbose=0)[0][0]

        # Physics bias
        bias = 1 + 0.03 * (ANODE["anode_dvdq_mean"] + CATHODE["cathode_dvdq_mean"])
        soc = soc_raw * bias
        if soc <= 1.0: soc = soc * 100.0

    # 2. TUNING 3: THE "SHOCK ABSORBER" (EMA FILTER)
    # This mathematically dampens the noise.
    # If this is the first run, or a huge reset happened, take the new value.
    if _prev_soc is None or abs(soc - _prev_soc) > 10.0:
        smooth_soc = soc 
    else:
        # 95% History, 5% New Data. 
        # This makes the needle move slowly and smoothly, ignoring noise.
        smooth_soc = (0.05 * soc) + (0.95 * _prev_soc)
        
    _prev_soc = smooth_soc

    return float(np.clip(smooth_soc, 0, 100))