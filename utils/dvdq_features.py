import pandas as pd
import numpy as np
import os

def load_dvdq_features(path, prefix):
    """
    Extracts static chemistry features from dV/dQ curves.
    """
    if not os.path.exists(path):
        # Return zeros if file not found to prevent crash
        return {
            f"{prefix}_dvdq_area": 0.0,
            f"{prefix}_dvdq_mean": 0.0,
            f"{prefix}_dvdq_std": 0.0,
            f"{prefix}_dvdq_max": 0.0
        }

    d = pd.read_csv(path)
    voltage = d.iloc[:,0].values
    dvdq = d.iloc[:,1].values
    
    # Your requested logic
    return {
        f"{prefix}_dvdq_area": np.trapz(dvdq, voltage),
        f"{prefix}_dvdq_mean": dvdq.mean(),
        f"{prefix}_dvdq_std": dvdq.std(),
        f"{prefix}_dvdq_max": dvdq.max()
    }