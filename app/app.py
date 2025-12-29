import sys
import os
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Fix Import Path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from simulation.ev_signal_generator import generate_ev_signal_row
from inference.soc_predictor import predict_soc
from inference.soh_predictor import predict_soh
from safety.health_rules import check_safety

# ---- CONFIG ----
st.set_page_config(page_title="Elektra Dashboard", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-family: 'Courier New'; font-weight: bold; }
    .stApp { background-color: #0E1117; }
</style>
""", unsafe_allow_html=True)

# ---- STATE ----
MAX_HISTORY = 100 
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["time", "voltage", "current", "temperature", "power"])
if "t" not in st.session_state:
    st.session_state.t = 0
if 'run_sim' not in st.session_state:
    st.session_state.run_sim = False

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Controls")
    def toggle_run():
        st.session_state.run_sim = not st.session_state.run_sim
    
    st.button("Start/Stop Simulation", on_click=toggle_run)
    status_text = "Running 🟢" if st.session_state.run_sim else "Paused 🔴"
    
    speed = st.slider("Update Speed", 0.05, 1.0, 0.1)
    
    if st.button("Reset System"):
        st.session_state.data = pd.DataFrame(columns=["time", "voltage", "current", "temperature", "power"])
        st.session_state.t = 0
        st.rerun()

# ---- LAYOUT ----
c1, c2 = st.columns([3, 1])
c1.title("⚡ Elektra Intelligent BMS")
c2.markdown(f"### {status_text}")

# Create Fixed Placeholders
top = st.container()
with top:
    k1, k2, k3 = st.columns([1.5, 1.5, 1])
    soc_ph = k1.empty()
    soh_ph = k2.empty()
    stat_ph = k3.empty()

st.divider()
chart_row = st.container()
with chart_row:
    g1, g2 = st.columns(2)
    volt_ph = g1.empty()
    temp_ph = g2.empty()

alert_ph = st.empty()

def create_gauge(val, title, stops):
    return go.Figure(go.Indicator(
        mode="gauge+number", value=val, title={'text': title},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "white"}, 'steps': stops, 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 99}}
    )).update_layout(height=250, margin=dict(t=50, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})

# ---- MAIN LOOP ----
if st.session_state.run_sim:
    while st.session_state.run_sim:
        # 1. Generate Data
        row = generate_ev_signal_row(st.session_state.t)
        st.session_state.t += 1
        
        # Efficient DataFrame Update
        st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([row])], ignore_index=True).tail(MAX_HISTORY)
        
        # 2. Predict SOC
        window = st.session_state.data.tail(30)
        soc = predict_soc(window)

        # ---- 🚀 TIME TRAVEL LOGIC ----
        # In reality, SOH takes years to drop. 
        # Here, we add 5 CYCLES per step. 
        # So 10 seconds of simulation = 50 cycles (approx 3 months of aging).
        sim_cycle = st.session_state.t * 5.0 
        
        # We also degrade capacity_ah artificially so the model sees correlation
        current_capacity = 2.5 * max(0.5, (1.0 - (sim_cycle * 0.001)))

        soh = predict_soh(
            cycle=sim_cycle,
            mean_voltage=window["voltage"].mean(),
            voltage_std=window["voltage"].std(),
            min_voltage=window["voltage"].min(),
            max_voltage=window["voltage"].max(),
            capacity_ah=current_capacity, 
            capacity_ratio=0.95,
            delta_capacity=-0.002,
            rolling_voltage_std=window["voltage"].std()
        )
        
        range_est = (soc / 100) * 300 * (soh / 100) # Range drops as health drops!

        # 3. Update UI
        key_id = st.session_state.t # Unique key for this frame
        
        soc_ph.plotly_chart(create_gauge(soc, "SOC (%)", [{'range': [0, 20], 'color': "red"}, {'range': [20, 100], 'color': "#00CC96"}]), use_container_width=True, key=f"soc_{key_id}")
        
        soh_ph.plotly_chart(create_gauge(soh, "SOH (%)", [{'range': [0, 70], 'color': "red"}, {'range': [70, 100], 'color': "#636EFA"}]), use_container_width=True, key=f"soh_{key_id}")
        
        stat_ph.metric("🚗 Est. Range", f"{range_est:.1f} mi", delta=f"{row['power']:.1f} W")
        
        # Charts
        v_fig = px.line(st.session_state.data, x="time", y="voltage", title="Voltage (V)", height=300)
        v_fig.update_traces(line_color='#00CC96')
        volt_ph.plotly_chart(v_fig, use_container_width=True, key=f"v_{key_id}")
        
        t_fig = px.line(st.session_state.data, x="time", y=["current", "temperature"], title="Current & Temp", height=300)
        temp_ph.plotly_chart(t_fig, use_container_width=True, key=f"t_{key_id}")
        
        # Alerts
        alerts = check_safety(soc, soh, row['temperature'])
        with alert_ph.container():
            st.subheader("🛡️ Diagnostics")
            for a in alerts:
                if "normal" in a: st.success(a)
                elif "Low" in a or "degrading" in a: st.warning(a)
                else: st.error(a)
        
        time.sleep(speed)