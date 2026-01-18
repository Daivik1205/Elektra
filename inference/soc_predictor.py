import numpy as np

class SOCPredictor:
    def __init__(self, total_capacity_ah=100.0):
        self.capacity_as = total_capacity_ah * 3600 # Amp-seconds
        self.prev_time = None
        self.estimated_soc = 90.0 # Initial guess
        
    def predict(self, voltage, current, temperature, timestamp):
        if self.prev_time is None:
            self.prev_time = timestamp
            return self.estimated_soc
        
        dt = timestamp - self.prev_time
        self.prev_time = timestamp
        
        # --- STRATEGY 1: COULOMB COUNTING (The Integrator) ---
        # This is very smooth but drifts over time
        coulomb_change = (current * dt) / self.capacity_as * 100
        self.estimated_soc += coulomb_change
        
        # --- STRATEGY 2: OCV RESET (The Corrector) ---
        # If current is very low (Idle), voltage stabilizes to OCV.
        # We can use this to "reset" the SOC to the true value based on voltage.
        if abs(current) < 1.0: 
            # Inverse of the OCV curve used in generator
            # cell_v = 3.2 + (soc/100) -> soc = (cell_v - 3.2) * 100
            cell_v = voltage / 96.0
            voltage_soc = (cell_v - 3.2) * 100
            
            # Gradually pull estimate towards voltage-based SOC (Complementary Filter)
            # This prevents jumps but corrects drift
            self.estimated_soc = (0.98 * self.estimated_soc) + (0.02 * voltage_soc)
            
        return np.clip(self.estimated_soc, 0, 100)