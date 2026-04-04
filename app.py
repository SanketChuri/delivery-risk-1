from pathlib import Path

import pydeck as pdk
import streamlit as st

from data_cleaning import load_data, clean_data
from risk_engine import apply_risk_logic


from phase1 import AlertConfig, build_phase1_operational_view

st.set_page_config(page_title="Delivery Risk Dashboard", layout="wide")

st.title("Delivery Risk Dashboard — Phase 1")
st.caption("Live operations view for continuous incoming jobs and driver monitoring.")

with st.sidebar:
    st.header("Data Sources")
    telemetry_path = st.text_input("Telemetry CSV path", value="data/driver_locations.csv")

    st.header("Map Controls")
    map_mode = st.radio(
        "Map mode",
        options=["Local (auto-center)", "World (global view)"],
        index=0,
        help="Switch to World mode to zoom out and see all drivers globally.",
    )
    map_style = st.selectbox("Map style", options=["Light", "Dark", "Satellite"], index=0)

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
    telemetry_path=telemetry_path,
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

if Path(telemetry_path).exists():
    st.info(f"Using driver telemetry from `{telemetry_path}` for map locations.")
else:
    st.warning(f"Telemetry file `{telemetry_path}` not found. Showing synthetic fallback coordinates.")

st.subheader("Live Driver Map")
st.caption("Use **Sidebar → Map Controls → Map mode → World (global view)** for world map.")
if ops_df.empty:
    st.info("No rows match current filters, so no drivers are displayed on the map.")
else:
    map_df = ops_df[["driver_id", "job_id", "alert_level", "risk_score", "driver_lat", "driver_lon"]].copy()
    map_df["color"] = map_df["alert_level"].map(
        {
            "urgent": [255, 59, 48, 200],
            "high": [255, 149, 0, 200],
            "normal": [52, 199, 89, 180],
        }
    )

    map_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[driver_lon, driver_lat]",
        get_fill_color="color",
        get_radius=300,
        pickable=True,
    )

    world_map_view = map_mode == "World (global view)"
    if world_map_view:
        center_lat, center_lon, zoom = 20.0, 0.0, 1
    else:
        center_lat = float(map_df["driver_lat"].mean())
        center_lon = float(map_df["driver_lon"].mean())
        zoom = 10

    style_lookup = {
    "Light": "light",
    "Dark": "dark",
    "Satellite": "road",
}

    deck = pdk.Deck(
        map_style=style_lookup[map_style],
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom, pitch=0),
        layers=[map_layer],
        tooltip={"text": "Driver: {driver_id}\nJob: {job_id}\nAlert: {alert_level}\nRisk: {risk_score}"},
    )
    st.pydeck_chart(deck, use_container_width=True)

    if world_map_view:
        st.caption("World view enabled. Marker colors: red=urgent, orange=high, green=normal.")
    else:
        st.caption("Marker colors: red=urgent, orange=high, green=normal.")

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