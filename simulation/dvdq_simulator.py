import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gaussian(x, amp, mu, sigma):
    """Creates a bell-curve peak (simulating a chemical reaction)."""
    return amp * np.exp(-((x - mu)**2) / (2 * sigma**2))

def generate_synthetic_profile(voltage_range, peaks, noise_level=0.05):
    """
    Generates a synthetic dV/dQ curve by combining multiple peaks.
    voltage_range: (min_v, max_v)
    peaks: list of tuples (amplitude, center_voltage, width)
    """
    voltage = np.linspace(voltage_range[0], voltage_range[1], 1000)
    dvdq = np.zeros_like(voltage)
    
    # Add physics-based peaks
    for amp, mu, sigma in peaks:
        dvdq += gaussian(voltage, amp, mu, sigma)
        
    # Add realistic sensor noise
    dvdq += np.random.normal(0, noise_level, size=len(voltage))
    
    return pd.DataFrame({"voltage": voltage, "dvdq": dvdq})

# ==========================================
# 1. GENERATE SYNTHETIC ANODE (Graphite/Silicon)
# ==========================================
# Anode dV/dQ usually has sharp peaks during phase transitions.
# We simulate discharge (negative peaks) to match your previous real data.
anode_peaks = [
    # (Amplitude, Voltage Location, Width)
    (-30.0, 0.10, 0.02), # Major Graphite Stage 1
    (-15.0, 0.21, 0.04), # Graphite Stage 2
    (-8.0,  0.50, 0.15)  # Silicon tail / SEI interaction
]

df_anode = generate_synthetic_profile((0.0, 1.5), anode_peaks)
# Sort by voltage for area calculation
df_anode = df_anode.sort_values(by="voltage")

# Invert signal for the AI (The app expects positive peaks for feature extraction)
# We save the file with POSITIVE peaks so find_peaks() works easily.
df_anode["dvdq"] = df_anode["dvdq"].abs()

df_anode.to_csv("dv_dq_anode.csv", index=False)
print("✅ Generated 'dv_dq_anode.csv' (Synthetic Graphite/Si Profile)")

# ==========================================
# 2. GENERATE SYNTHETIC CATHODE (NCA/NMC)
# ==========================================
# Cathode peaks are broader and at higher voltages (3.5V - 4.2V)
cathode_peaks = [
    (5.0,  3.6, 0.1),  # Transition 1
    (12.0, 3.8, 0.08), # Major Nickel Peak
    (8.0,  4.1, 0.09)  # High voltage phase
]

df_cathode = generate_synthetic_profile((3.0, 4.3), cathode_peaks)
df_cathode = df_cathode.sort_values(by="voltage")
df_cathode.to_csv("dv_dq_cathode.csv", index=False)
print("✅ Generated 'dv_dq_cathode.csv' (Synthetic NCA Profile)")

# Optional: Plot to verify
try:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df_anode["voltage"], df_anode["dvdq"], color="red")
    plt.title("Synthetic Anode (Inverted for AI)")
    
    plt.subplot(1, 2, 2)
    plt.plot(df_cathode["voltage"], df_cathode["dvdq"], color="blue")
    plt.title("Synthetic Cathode")
    plt.tight_layout()
    plt.show()
except:
    pass