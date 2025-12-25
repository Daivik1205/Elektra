import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

from utils.dvdq_features import extract_dvdq_features

SOC_MODEL = load_model("models/elektra_soc_lstm.keras")

scaler = StandardScaler()

ANODE = extract_dvdq_features("data/dv_dq_anode.csv", "anode")
CATHODE = extract_dvdq_features("data/dv_dq_cathode.csv", "cathode")

def predict_soc(df):
    features = df[["voltage", "current", "temperature"]].values
    features = scaler.fit_transform(features)

    X = features.reshape(1, features.shape[0], features.shape[1])
    soc = SOC_MODEL.predict(X, verbose=0)[0][0]

    # physics-informed bias
    bias = 1 + 0.03 * (ANODE["anode_dvdq_mean"] + CATHODE["cathode_dvdq_mean"])
    soc *= bias

    return float(np.clip(soc, 0, 100))
