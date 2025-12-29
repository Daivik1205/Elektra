import pandas as pd
import joblib
import numpy as np
import os
from utils.dvdq_features import extract_dvdq_features

# Load Model
MODEL_PATH = "models/elektra_soh_model.pkl"
SOH_MODEL = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Load Physics Baselines
try:
    ANODE_BASE = extract_dvdq_features("dv_dq_anode.csv", "anode")
    CATHODE_BASE = extract_dvdq_features("dv_dq_cathode.csv", "cathode")
except:
    ANODE_BASE = {"anode_dvdq_area": 10.0, "anode_dvdq_peak_count": 2, "anode_dvdq_mean": 5.0}
    CATHODE_BASE = {"cathode_dvdq_area": 10.0, "cathode_dvdq_peak_count": 2, "cathode_dvdq_mean": 5.0}

FEATURE_ORDER = [
    "cycle", "mean_voltage", "voltage_std", "min_voltage", "max_voltage",
    "capacity_ah", "capacity_ratio", 
    "anode_dvdq_area", "anode_dvdq_peak_count", "anode_dvdq_mean",
    "cathode_dvdq_area", "cathode_dvdq_peak_count", "cathode_dvdq_mean",
    "delta_capacity", "rolling_voltage_std"
]

def predict_soh(cycle, mean_voltage, voltage_std, min_voltage, max_voltage, 
                capacity_ah, capacity_ratio, delta_capacity, rolling_voltage_std):
    
    if SOH_MODEL is None:
        return max(0, 100 - (cycle * 0.1))

    # ---- 🚀 DEMO MODE AGING (AGGRESSIVE) ----
    # 1. Decay: Drop 0.1% per cycle (200x faster than reality)
    # This ensures the gauge visibly moves if cycle increases by 10
    decay = max(0.4, 1.0 - (cycle * 0.001)) 
    
    # 2. Add "Jitter": Real sensors fluctuate.
    # We add +/- 0.5% random noise to the features. 
    # This makes the SOH needle vibrate slightly like a real car dashboard.
    def add_jitter(val):
        noise = np.random.normal(0, 0.05) # 5% noise variance
        return val * decay * (1 + noise)

    current_anode = {
        "anode_dvdq_area": add_jitter(ANODE_BASE["anode_dvdq_area"]),
        "anode_dvdq_peak_count": ANODE_BASE["anode_dvdq_peak_count"],
        "anode_dvdq_mean": add_jitter(ANODE_BASE["anode_dvdq_mean"])
    }
    
    current_cathode = {
        "cathode_dvdq_area": add_jitter(CATHODE_BASE["cathode_dvdq_area"]),
        "cathode_dvdq_peak_count": CATHODE_BASE["cathode_dvdq_peak_count"],
        "cathode_dvdq_mean": add_jitter(CATHODE_BASE["cathode_dvdq_mean"])
    }

    # 3. Prepare Input
    row = {
        "cycle": cycle,
        "mean_voltage": mean_voltage,
        "voltage_std": voltage_std,
        "min_voltage": min_voltage,
        "max_voltage": max_voltage,
        "capacity_ah": capacity_ah,
        "capacity_ratio": capacity_ratio,
        "delta_capacity": delta_capacity,
        "rolling_voltage_std": rolling_voltage_std,
        **current_anode,
        **current_cathode
    }

    df = pd.DataFrame([row])
    df = df.reindex(columns=FEATURE_ORDER, fill_value=0)

    pred = float(SOH_MODEL.predict(df)[0])
    
    # Clip to realistic bounds
    return float(np.clip(pred, 0, 100))