import pandas as pd
import joblib
from utils.dvdq_features import extract_dvdq_features

SOH_MODEL = joblib.load("models/elektra_soh_model.pkl")

ANODE = extract_dvdq_features("data/dv_dq_anode.csv", "anode")
CATHODE = extract_dvdq_features("data/dv_dq_cathode.csv", "cathode")

# 🔥 EXACT TRAINING FEATURE ORDER
FEATURE_ORDER = [
    "cycle",
    "mean_voltage",
    "voltage_std",
    "min_voltage",
    "max_voltage",
    "capacity_ah",
    "capacity_ratio",
    "anode_dvdq_area",
    "anode_dvdq_peak_count",
    "anode_dvdq_mean",
    "cathode_dvdq_area",
    "cathode_dvdq_peak_count",
    "cathode_dvdq_mean",
    "delta_capacity",
    "rolling_voltage_std",
]

def predict_soh(
    cycle,
    mean_voltage,
    voltage_std,
    min_voltage,
    max_voltage,
    capacity_ah,
    capacity_ratio,
    delta_capacity,
    rolling_voltage_std
):
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
        **ANODE,
        **CATHODE
    }

    df = pd.DataFrame([row])

    # 🔒 FORCE FEATURE ORDER
    df = df.reindex(columns=FEATURE_ORDER)

    return float(SOH_MODEL.predict(df)[0])
