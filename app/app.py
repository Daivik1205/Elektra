# ============================
# ELEKTRA – LIVE EV DASHBOARD
# ============================

import sys
import os
import time
import pandas as pd
import streamlit as st

# ---- FIX IMPORT PATH ONCE ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ---- IMPORT PROJECT MODULES ----
from simulation.ev_signal_generator import generate_ev_signal_row
from inference.soc_predictor import predict_soc
from inference.soh_predictor import predict_soh
from safety.health_rules import check_safety

# ---- STREAMLIT CONFIG ----
st.set_page_config(
    page_title="Elektra – EV Battery AI",
    layout="wide"
)

st.title("⚡ Elektra – Live EV Battery Monitor")
st.caption("Real-time SOC & SOH prediction using AI + Electrochemistry")

# ---- SESSION STATE INIT ----
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(
        columns=["time", "voltage", "current", "temperature"]
    )

if "t" not in st.session_state:
    st.session_state.t = 0

# ---- LIVE UPDATE TOGGLE ----
run = st.toggle("▶️ Run Live Simulation", value=True)

# ---- PLACEHOLDERS ----
metric_col1, metric_col2, metric_col3 = st.columns(3)
chart_placeholder = st.empty()
alert_placeholder = st.empty()

# ---- MAIN LOOP ----
if run:
    for _ in range(1000):  # soft infinite loop
        # generate new sensor row
        row = generate_ev_signal_row(st.session_state.t)
        st.session_state.t += 1

        st.session_state.data = pd.concat(
            [st.session_state.data, pd.DataFrame([row])],
            ignore_index=True
        )

        df = st.session_state.data.tail(30)  # sliding window

        # ---- PREDICTIONS ----
        soc = predict_soc(df)

        soh = predict_soh(
            cycle=120,
            mean_voltage=df["voltage"].mean(),
            voltage_std=df["voltage"].std(),
            min_voltage=df["voltage"].min(),
            max_voltage=df["voltage"].max(),
            capacity_ah=2.5,
            capacity_ratio=0.95,
            delta_capacity=-0.002,
            rolling_voltage_std=df["voltage"].rolling(10).std().mean()
        )

        alerts = check_safety(
            soc=soc,
            soh=soh,
            temp=df["temperature"].max()
        )

        # ---- METRICS ----
        metric_col1.metric("🔋 SOC (%)", f"{soc:.2f}")
        metric_col2.metric("❤️ SOH (%)", f"{soh:.2f}")
        metric_col3.metric("🌡️ Max Temp (°C)", f"{df['temperature'].max():.1f}")

        # ---- CHART ----
        chart_placeholder.line_chart(
            df.set_index("time")[["voltage", "current", "temperature"]]
        )

        # ---- ALERTS ----
        with alert_placeholder.container():
            st.subheader("🚨 Safety Status")
            for a in alerts:
                st.write(a)

        time.sleep(1)
        st.rerun()

else:
    st.info("Simulation paused ⏸️")
