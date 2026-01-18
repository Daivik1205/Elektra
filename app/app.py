import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import datetime

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.ev_signal_generator import EVSignalGenerator
from inference.soc_predictor import SOCPredictor
from inference.soh_predictor import SOHPredictor
from utils.dvdq_features import load_dvdq_features
from safety.health_rules import check_safety

st.set_page_config(page_title="Elektra BMS Pro", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
    <style>
        div[data-testid="stMetric"] {
            background-color: #121212;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #333;
            min-height: 140px; 
            overflow: hidden;
        }
        div[data-testid="stPlotlyChart"] {
            background-color: #121212;
            border-radius: 10px;
            border: 1px solid #333;
            padding: 5px;
            min-height: 380px;
        }
        .live-val-box {
            font-family: 'Courier New', monospace;
            font-size: 26px;
            font-weight: bold;
            text-align: center;
            padding: 12px;
            margin-top: 10px;
            border-radius: 8px;
            background-color: #1E1E1E;
            border: 1px solid #444;
        }
        .val-v { color: #00CCFF; }
        .val-i { color: #FF5500; }
        .clock-box { color: #888; font-size: 14px; text-align: center; }
        .stDeployButton {display:none;}
        header {display: none;}
        .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

NOMINAL_CAPACITY = 100.0

# --- STATE ---
if 'car' not in st.session_state:
    st.session_state.car = EVSignalGenerator()
    st.session_state.soc_ai = SOCPredictor()
    st.session_state.soh_ai = SOHPredictor()
    st.session_state.history = pd.DataFrame(columns=["time", "voltage", "current", "soc", "temperature", "voltage_std", "capacity_ratio"])
    st.session_state.cycle_count = 10.0
    st.session_state.simulation_running = False
    st.session_state.sim_clock = 0.0 # Track simulated seconds
    st.session_state.smoothed_soc = 50.0  # Smoothed SOC
    st.session_state.smoothed_soh = 100.0  # Smoothed SOH
    st.session_state.soc_alpha = 0.15  # EMA smoothing factor for SOC
    st.session_state.soh_alpha = 0.1   # EMA smoothing factor for SOH
    st.session_state.soh_buffer = []  # Buffer for SOH predictions to stabilize
    st.session_state.soh_buffer_size = 20  # Number of predictions to collect before smoothing
    st.session_state.last_cycle_count = 10.0  # Track cycle changes
    st.session_state.current_capacity_ah = 100.0  # Current capacity that degrades
    st.session_state.capacity_history = [100.0]  # Track capacity over time
    st.session_state.last_cycle_count = 10.0  # Track cycle changes

@st.cache_resource
def get_chemistry_features():
    anode = load_dvdq_features("data/dv_dq_anode.csv", "anode")
    cathode = load_dvdq_features("data/dv_dq_cathode.csv", "cathode")
    return {**anode, **cathode}

static_features = get_chemistry_features()

# --- UI ---
st.title("⚡ Elektra: Digital Twin")
st.sidebar.header("🎮 Controls")

# Initialize operation mode in session state
if "operation_mode" not in st.session_state:
    st.session_state.operation_mode = "STANDBY"

# Mode Selection with centered buttons
st.sidebar.subheader("Operation Mode")
mode_col1, mode_col2, mode_col3 = st.sidebar.columns(3)

with mode_col1:
    if st.sidebar.button("🔌 CHARGE", use_container_width=True):
        st.session_state.operation_mode = "CHARGE"
        st.session_state.car.set_mode("CHARGE")

with mode_col2:
    if st.sidebar.button("🚗 DRIVE", use_container_width=True):
        st.session_state.operation_mode = "DISCHARGE"
        st.session_state.car.set_mode("DISCHARGE")

with mode_col3:
    if st.sidebar.button("⏸️ IDLE", use_container_width=True):
        st.session_state.operation_mode = "STANDBY"
        st.session_state.car.set_mode("STANDBY")

st.sidebar.divider()

# Start/Stop simulation
sim_col1, sim_col2 = st.sidebar.columns([2, 1])
with sim_col1:
    if st.sidebar.button("▶️ START SIM", type="primary", use_container_width=True):
        st.session_state.simulation_running = True

with sim_col2:
    if st.sidebar.button("⏹️ STOP", use_container_width=True):
        st.session_state.simulation_running = False

st.sidebar.divider()

# Speed Slider (1x to 5000x)
speed_factor = st.sidebar.slider("⚡ Speed (x)", 1, 5000, 100)

# Cycle Slider
cycle_input = st.sidebar.slider(
    "🔄 Battery Age (Cycles)", 
    min_value=0.0, max_value=5000.0, 
    value=float(st.session_state.cycle_count), step=10.0
)
st.session_state.cycle_count = cycle_input

# Display current mode
mode_color = {"CHARGE": "🟢", "DISCHARGE": "🔴", "STANDBY": "⚪"}
st.sidebar.info(f"{mode_color.get(st.session_state.operation_mode, '⚪')} Mode: **{st.session_state.operation_mode}**")

m1, m2, m3, m4 = st.columns(4)
c1, c2 = st.columns(2)
clock_ph = st.empty()

# --- RUN LOOP ---
if st.session_state.simulation_running:
    
    # 1. PHYSICS STEP
    real_dt = 0.1 # We update screen every 0.1s
    
    # Pass 'real_dt' and 'speed_factor' to physics
    data = st.session_state.car.step(real_dt, speed_factor)
    
    # Update Simulated Clock
    st.session_state.sim_clock += (real_dt * speed_factor)
    sim_time_str = str(datetime.timedelta(seconds=int(st.session_state.sim_clock)))

    # 3. CALCULATE REALISTIC CAPACITY DEGRADATION
    # Capacity degrades with: cycles + temperature + current stress
    cycle_degradation = cycle_input * 0.00015  # 0.015% per cycle
    temp_stress = max(0, (data['temperature'] - 25) * 0.0001)  # Higher temp = more degradation
    current_stress = abs(data['current']) * 0.00000005  # High current = more stress
    
    # Total degradation rate
    total_degradation = cycle_degradation + temp_stress + current_stress
    st.session_state.current_capacity_ah = NOMINAL_CAPACITY * (1.0 - total_degradation)
    st.session_state.current_capacity_ah = np.clip(st.session_state.current_capacity_ah, NOMINAL_CAPACITY * 0.5, NOMINAL_CAPACITY)
    
    # Track capacity history
    st.session_state.capacity_history.append(st.session_state.current_capacity_ah)
    if len(st.session_state.capacity_history) > 100:
        st.session_state.capacity_history.pop(0)
    
    # Calculate capacity_ratio (current capacity / nominal)
    capacity_ratio = st.session_state.current_capacity_ah / NOMINAL_CAPACITY
    
    # Calculate delta_capacity (rate of change)
    if len(st.session_state.capacity_history) > 1:
        delta_cap = (st.session_state.capacity_history[-1] - st.session_state.capacity_history[-2]) / st.session_state.capacity_history[-1]
    else:
        delta_cap = 0.0
    
    # 4. HISTORY WITH REALISTIC VALUES
    new_row = pd.DataFrame([{
        "time": pd.to_datetime(data['time'], unit='s'),
        "voltage": data['voltage'],
        "current": data['current'],
        "soc": 0,  # Will be updated after prediction
        "temperature": data['temperature'],
        "voltage_std": 0, 
        "capacity_ratio": capacity_ratio,
        "power": data.get('power', 0)
    }])
    st.session_state.history = pd.concat([st.session_state.history, new_row]).tail(100)
    
    # 5. FEATURES
    hist = st.session_state.history
    current_std = hist['voltage'].rolling(5).std().fillna(0).iloc[-1]
    roll_std = hist['voltage'].rolling(10).std().mean()
    if pd.isna(roll_std): roll_std = 0.0
    if pd.isna(delta_cap): delta_cap = 0.0

    # 6. PREDICT (AI MODELS ONLY)
    # SOC Prediction from AI model
    pred_soc_raw = st.session_state.soc_ai.predict(data['voltage'], data['current'], data['temperature'], data['time'])
    
    # Smooth SOC using exponential moving average
    st.session_state.smoothed_soc = (st.session_state.soc_alpha * pred_soc_raw) + ((1 - st.session_state.soc_alpha) * st.session_state.smoothed_soc)
    pred_soc = st.session_state.smoothed_soc
    
    # SOH Prediction from AI model with REALISTIC degraded values
    pred_soh_raw = st.session_state.soh_ai.predict(
        dynamic_features={
            "cycle": int(cycle_input),
            "mean_voltage": hist['voltage'].mean(),
            "voltage_std": current_std,
            "min_voltage": hist['voltage'].min(),
            "max_voltage": hist['voltage'].max(),
            "capacity_ah": st.session_state.current_capacity_ah,  # Use degraded capacity
            "capacity_ratio": capacity_ratio,  # Use realistic ratio
            "delta_capacity": delta_cap,  # Use real degradation rate
            "rolling_voltage_std": roll_std
        },
        static_features=static_features
    )
    
    # Buffer SOH predictions for smoother initialization
    st.session_state.soh_buffer.append(pred_soh_raw)
    
    # Once buffer is full, use it for smoothing
    if len(st.session_state.soh_buffer) > st.session_state.soh_buffer_size:
        st.session_state.soh_buffer.pop(0)  # Remove oldest
        buffer_mean = np.mean(st.session_state.soh_buffer)
        st.session_state.smoothed_soh = (st.session_state.soh_alpha * buffer_mean) + ((1 - st.session_state.soh_alpha) * st.session_state.smoothed_soh)
    else:
        # During buffer filling phase, use simple average
        if len(st.session_state.soh_buffer) >= 5:
            buffer_mean = np.mean(st.session_state.soh_buffer)
            st.session_state.smoothed_soh = (st.session_state.soh_alpha * buffer_mean) + ((1 - st.session_state.soh_alpha) * st.session_state.smoothed_soh)
    
    pred_soh = np.clip(st.session_state.smoothed_soh, 0, 100)
    
    # Update history with predicted SOC
    st.session_state.history.iloc[-1, st.session_state.history.columns.get_loc('soc')] = pred_soc

    # 5. RENDER
    # Show SIMULATED TIME and MODE
    clock_ph.markdown(f"<div class='clock-box'>⏱️ Simulated: {sim_time_str} | 🔋 Mode: {st.session_state.operation_mode}</div>", unsafe_allow_html=True)

    m1.metric("⚡ SOC", f"{pred_soc:.1f}%")
    m2.metric("❤️ SOH", f"{pred_soh:.1f}%")
    m3.metric("🔄 Cycles", f"{int(cycle_input)}")
    m4.metric("🌡️ Temp", f"{data['temperature']:.1f}°C")
    
    with c1:
        st.markdown(f"<div class='live-val-box val-v'>{data['voltage']:.2f} V</div>", unsafe_allow_html=True)
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=hist['time'], y=hist['voltage'], mode='lines', line=dict(color='#00CCFF', width=2)))
        fig_v.update_layout(title="<b>Voltage</b>", height=350, margin=dict(t=30,b=10,l=10,r=10), template="plotly_dark", uirevision='const', xaxis=dict(showticklabels=False), yaxis=dict(showgrid=True, gridcolor='#333'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_v, use_container_width=True)

    with c2:
        power = data['voltage'] * abs(data['current']) / 1000.0
        st.markdown(f"<div class='live-val-box val-i'>{data['current']:.2f} A | {power:.1f} kW</div>", unsafe_allow_html=True)
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(x=hist['time'], y=hist['current'], fill='tozeroy', line=dict(color='#FF5500')))
        fig_i.add_trace(go.Scatter(x=hist['time'], y=hist['temperature'], name='Temp', yaxis='y2', line=dict(color='yellow', dash='dot')))
        fig_i.update_layout(title="<b>Current / Temp</b>", height=350, margin=dict(t=30,b=10,l=10,r=10), template="plotly_dark", uirevision='const', xaxis=dict(showticklabels=False), yaxis=dict(showgrid=True, gridcolor='#333'), yaxis2=dict(overlaying='y', side='right', showgrid=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_i, use_container_width=True)

    # Alerts
    alerts = check_safety(pred_soc, pred_soh, data['temperature'])
    if alerts:
        st.error(f"⚠️ {alerts[0]}")
    else:
        st.success("✅ System Nominal")

    time.sleep(0.05)
    st.rerun()

else:
    st.info(f"🛑 Stopped | Mode: **{st.session_state.operation_mode}** | Press ▶️ to start simulation")