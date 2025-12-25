# simulation/ev_signal_generator.py
import numpy as np
import pandas as pd

def generate_ev_signal_row(t):
    return {
        "time": t,
        "voltage": np.random.uniform(3.6, 4.2),
        "current": np.random.uniform(-2.0, 2.0),
        "temperature": np.random.uniform(25, 45)
    }
