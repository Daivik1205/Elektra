import numpy as np

class EVSignalGenerator:
    def __init__(self):
        self.soc_simulated = 0.80  # Start roughly 80%
        self.current_speed = 0
        self.temp_simulated = 30.0
    
    def step(self, t):
        # Simulate Driving Physics (Random Walk with Inertia)
        # Acceleration (positive current) or Regen (negative current)
        acceleration = np.random.normal(0, 0.5) 
        self.current_speed = np.clip(self.current_speed + acceleration, -10, 10)
        
        # Current depends on speed/load 
        current = (self.current_speed * 0.5) + np.random.normal(0, 0.1)
        
        # Temperature rises with high current, cools slowly
        self.temp_simulated += (abs(current) * 0.05) - 0.02
        self.temp_simulated = np.clip(self.temp_simulated, 20, 60)

        # Voltage Physics (OCV - IR Drop)
        self.soc_simulated -= (current * 0.0001) # Slow depletion
        self.soc_simulated = np.clip(self.soc_simulated, 0, 1)
        
        ocv = 3.0 + (self.soc_simulated * 1.2) 
        voltage = ocv - (current * 0.05)
        
        return {
            "time": t,
            "voltage": float(voltage),
            "current": float(current),
            "temperature": float(self.temp_simulated),
            "power": float(voltage * current)
        }

# Singleton instance
_sim = EVSignalGenerator()

def generate_ev_signal_row(t):
    return _sim.step(t)