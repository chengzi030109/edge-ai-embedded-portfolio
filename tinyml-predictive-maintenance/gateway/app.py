from __future__ import annotations

"""Streamlit gateway dashboard for local telemetry.

This dashboard plays the role of an embedded-Linux gateway UI. The simulated
node writes JSONL telemetry, and this app reads that file to show current state,
alarm count, and score trends. When MQTT is added later, this file can be
extended to subscribe to live messages instead of reading from disk.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="TinyML Maintenance Gateway", layout="wide")
st.title("TinyML Predictive Maintenance Gateway")

# The user can point the dashboard at any telemetry run. The default matches
# scripts/run_simulated_node.py.
telemetry_path = Path(st.sidebar.text_input("Telemetry JSONL", "runs/telemetry.jsonl"))

if not telemetry_path.exists():
    st.info("Run scripts/run_simulated_node.py first to generate telemetry.")
    st.stop()

rows = []
with telemetry_path.open(encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            # Each line is a standalone telemetry message emitted by the node.
            rows.append(json.loads(line))

if not rows:
    st.info("Telemetry file is empty.")
    st.stop()

df = pd.DataFrame(rows)
latest = df.iloc[-1]
alarm_col = "is_alarm" if "is_alarm" in df.columns else "is_anomaly"

# Top-level operational metrics: this is what an operator would glance at first.
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest State", latest["true_state"])
col2.metric("Anomaly Score", f"{latest['score']:.2f}")
col3.metric("Threshold", f"{latest['threshold']:.2f}")
col4.metric("Debounced Alarms", int(df[alarm_col].sum()))

# Plot score and threshold together so alarms are visually explainable.
st.line_chart(df.set_index("seq")[["score", "threshold"]])

if "is_alarm" in df.columns:
    st.line_chart(df.set_index("seq")[["is_anomaly_raw", "is_alarm"]])

with st.expander("Recent telemetry"):
    # Keep the full raw table accessible for debugging model behavior.
    st.dataframe(df.tail(50), use_container_width=True)
