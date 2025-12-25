import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def extract_dvdq_features(csv_path, prefix):
    df = pd.read_csv(csv_path)

    voltage = df.iloc[:, 0].values
    dvdq = df.iloc[:, 1].values

    peaks, _ = find_peaks(dvdq)

    return {
        f"{prefix}_dvdq_area": float(np.trapezoid(dvdq, voltage)),
        f"{prefix}_dvdq_peak_count": int(len(peaks)),
        f"{prefix}_dvdq_mean": float(np.mean(dvdq))
    }
