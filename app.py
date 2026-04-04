import streamlit as st
import pandas as pd
from data_cleaning import load_data, clean_data
from risk_engine import apply_risk_logic


from phase1 import AlertConfig, build_phase1_operational_view

st.set_page_config(page_title="Delivery Risk Dashboard", layout="wide")

st.title("Delivery Risk Dashboard — Phase 1")
st.caption("Live operations view for continuous incoming jobs and driver monitoring.")

with st.sidebar:
    st.header("Alert Configuration")
    high_risk_threshold = st.slider("High-risk score threshold", min_value=50, max_value=95, value=70)
    urgent_delay_minutes = st.slider("Urgent delay threshold (minutes)", min_value=10, max_value=60, value=30)

    st.header("Filters")
    risk_filter = st.multiselect("Risk level", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    alert_filter = st.multiselect("Alert level", options=["urgent", "high", "normal"], default=["urgent", "high", "normal"])

# Phase-1 ingest from csv (replace with queue/db in production)
df = load_data("data/dirtyFile.csv")
df_clean = clean_data(df)
df_final = apply_risk_logic(df_clean)

ops_df = build_phase1_operational_view(
    df_final,
    config=AlertConfig(high_risk_threshold=high_risk_threshold, urgent_delay_minutes=urgent_delay_minutes),
)

ops_df = ops_df[ops_df["risk_level"].isin(risk_filter) & ops_df["alert_level"].isin(alert_filter)]
ops_df = ops_df.sort_values(by=["alert_level", "risk_score", "delay"], ascending=[True, False, False])

urgent_count = int((ops_df["alert_level"] == "urgent").sum())
high_count = int((ops_df["alert_level"] == "high").sum())
normal_count = int((ops_df["alert_level"] == "normal").sum())

c1, c2, c3 = st.columns(3)
c1.metric("Urgent alerts", urgent_count)
c2.metric("High alerts", high_count)
c3.metric("Normal", normal_count)

if urgent_count > 0:
    st.error(f"{urgent_count} urgent deliveries require immediate action.")
else:
    st.success("No urgent deliveries right now.")

st.subheader("Operational Queue")
st.dataframe(
    ops_df[
        [
            "job_id",
            "driver_id",
            "delay",
            "risk_score",
            "risk_level",
            "alert_level",
            "ops_action",
            "status",
            "traffic_level",
            "weather_severity",
            "expected_delivery_time",
            "eta_drift",
            "driver_lat",
            "driver_lon",
            "last_telemetry_utc",
        ]
    ],
    use_container_width=True,
)

st.subheader("Risk Distribution")
st.bar_chart(ops_df["risk_level"].value_counts())

st.subheader("Alert Distribution")
st.bar_chart(ops_df["alert_level"].value_counts())

# df = load_data("data/dirtyFile.csv")
# df_clean = clean_data(df)
# df_final = apply_risk_logic(df_clean)

# st.title("Delivery Risk Dashboard")
# st.dataframe(df_final[['job_id','driver_id','delay','risk_score','risk_level','recommended_action']])
# st.bar_chart(df_final['risk_level'].value_counts())