import numpy as np
import time

class EVSignalGenerator:
    def __init__(self):
        # State
        self.soc = 50.0        # %
        self.temp = 25.0       # °C
        self.current = 0.0     # Amps
        self.voltage = 350.0   # Volts
        self.odometer = 0.0    # km
        
        # Physics Constants
        self.capacity_ah = 100.0 # 100 Ah Battery
        self.resistance = 0.05   # 50mOhm Internal Resistance
        
        # Drive Cycle State Machine
        self.phase_timer = 0
        self.phase = "IDLE"
        self.operation_mode = "STANDBY"  # STANDBY, DISCHARGE, CHARGE

    def set_mode(self, mode):
        """Set operation mode: STANDBY, DISCHARGE, CHARGE"""
        if mode in ["STANDBY", "DISCHARGE", "CHARGE"]:
            self.operation_mode = mode
            self.phase_timer = 0
            self.phase = "IDLE"

    def _get_ocv(self, soc_pct):
        """Standard Li-Ion OCV Curve"""
        s = soc_pct / 100.0
        # 3.0V empty -> 4.2V full per cell (96s pack = 288V -> 403V)
        return 288 + (70 * s) + (45 * s**3)

    def step(self, real_dt, speed_factor):
        """
        real_dt: Real world time elapsed (e.g. 0.1s)
        speed_factor: Time multiplier (e.g. 100x)
        """
        # 1. CALCULATE SIMULATED TIME
        sim_dt = real_dt * speed_factor
        
        # 2. UPDATE DRIVE CYCLE based on operation mode
        self.phase_timer += sim_dt
        
        # --- OPERATION MODE LOGIC ---
        if self.operation_mode == "STANDBY":
            # Minimal current, natural decay
            target_i = np.random.normal(-0.5, 0.2)
            
        elif self.operation_mode == "DISCHARGE":
            # Driving cycle: IDLE -> ACCEL -> CRUISE -> REGEN -> repeat
            if self.phase == "IDLE":
                target_i = -2.0
                if self.phase_timer > 30:
                    self.phase = "ACCEL"
                    self.phase_timer = 0
            
            elif self.phase == "ACCEL":
                target_i = -180.0 + np.random.normal(0, 10)  # Heavy discharge
                if self.phase_timer > 15:
                    self.phase = "CRUISE"
                    self.phase_timer = 0
                    
            elif self.phase == "CRUISE":
                target_i = -60.0 + np.random.normal(0, 5)  # Highway cruise
                if self.phase_timer > 60:
                    self.phase = "REGEN"
                    self.phase_timer = 0
                    
            elif self.phase == "REGEN":
                target_i = 80.0 + np.random.normal(0, 8)  # Regenerative braking
                if self.phase_timer > 10:
                    self.phase = "IDLE"
                    self.phase_timer = 0
        
        elif self.operation_mode == "CHARGE":
            # Charging profiles: CC (constant current) -> CV (constant voltage)
            if self.soc < 80.0:
                # Constant current phase - fast charging
                target_i = 80.0 + np.random.normal(0, 5)  # Charge current
                if self.phase_timer > 120:  # Charge for 2 min simulated
                    self.phase = "CV"
                    self.phase_timer = 0
            else:
                # Constant voltage phase - taper charging
                target_i = 20.0 + np.random.normal(0, 2)  # Lower current
                if self.soc >= 98.0:
                    target_i = 0.0
        
        # 3. PHYSICS UPDATE
        
        # Smooth current response
        if speed_factor > 10:
            self.current = target_i + np.random.normal(0, 5.0)
        else:
            self.current = (0.9 * self.current) + (0.1 * target_i) + np.random.normal(0, 2.0)

        # Coulomb Counting (The Drain/Charge)
        ah_used = self.current * (sim_dt / 3600.0)
        self.soc += (ah_used / self.capacity_ah) * 100.0
        self.soc = np.clip(self.soc, 0.0, 100.0)

        # Temperature model
        load_ratio = abs(self.current) / 200.0
        
        if self.operation_mode == "CHARGE":
            target_temp = 25.0 + (load_ratio * 35.0)  # Charging heats less than discharging
        else:
            target_temp = 25.0 + (load_ratio * 40.0)
        
        if speed_factor > 50:
            self.temp = target_temp
        else:
            self.temp = (0.99 * self.temp) + (0.01 * target_temp)
        
        self.temp += np.random.normal(0, 0.1)
        self.temp = np.clip(self.temp, -10, 70)

        # Voltage Sag (V = OCV - IR)
        ocv = self._get_ocv(self.soc)
        sag = self.current * self.resistance
        self.voltage = ocv + sag
        self.voltage += np.random.normal(0, 0.2)
        self.voltage = np.clip(self.voltage, 280, 410)

        return {
            "time": time.time(),
            "voltage": self.voltage,
            "current": self.current,
            "temperature": self.temp,
            "soc": self.soc,
            "power": self.voltage * self.current / 1000.0  # kW
        }
        ocv = self._get_ocv(self.soc)
        sag = self.current * self.resistance
        self.voltage = ocv + sag # Current is negative, so V drops
        self.voltage += np.random.normal(0, 0.2)

        return {
            "time": time.time(),
            "voltage": self.voltage,
            "current": self.current,
            "temperature": self.temp,
            "soc": self.soc
        }