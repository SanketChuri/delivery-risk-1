from pathlib import Path
import math

import pandas as pd
import pydeck as pdk
import streamlit as st

from data_cleaning import load_data, clean_data
from risk_engine import apply_risk_logic
from phase1 import AlertConfig, build_phase1_operational_view


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two lat/lon points in kilometers."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return round(r * c, 2)


def normalize_fleet_telemetry(telemetry_path: str) -> pd.DataFrame:
    """Load and normalize fleet telemetry into driver_lat/driver_lon columns."""
    fleet_df = pd.read_csv(telemetry_path)
    fleet_df.columns = fleet_df.columns.str.strip().str.lower()

    fleet_df = fleet_df.rename(
        columns={
            "lat": "driver_lat",
            "lon": "driver_lon",
            "latitude": "driver_lat",
            "longitude": "driver_lon",
            "lng": "driver_lon",
        }
    )

    if "driver_id" not in fleet_df.columns:
        raise ValueError("Telemetry file must contain a driver_id column.")
    if "driver_lat" not in fleet_df.columns or "driver_lon" not in fleet_df.columns:
        raise ValueError("Telemetry file must contain lat/lon or equivalent columns.")

    fleet_df["driver_id"] = fleet_df["driver_id"].astype(str).str.strip().str.lower()
    fleet_df["driver_lat"] = pd.to_numeric(fleet_df["driver_lat"], errors="coerce")
    fleet_df["driver_lon"] = pd.to_numeric(fleet_df["driver_lon"], errors="coerce")
    fleet_df = fleet_df.dropna(subset=["driver_lat", "driver_lon"]).copy()
    return fleet_df


def classify_driver(row: pd.Series) -> str:
    if row["is_assigned"]:
        return "assigned"
    if row["distance_to_pickup_km"] <= 20:
        return "nearby_available"
    return "idle"


def get_driver_fill(row: pd.Series) -> list[int]:
    if row["is_closest_driver"]:
        return [255, 215, 0, 230]  # gold

    if row["is_assigned"]:
        if row.get("alert_level") == "urgent":
            return [255, 59, 48, 210]  # red
        if row.get("alert_level") == "high":
            return [255, 149, 0, 200]  # orange
        return [52, 199, 89, 190]  # green

    if row["driver_status"] == "nearby_available":
        return [0, 122, 255, 200]  # blue

    return [142, 142, 147, 120]  # grey


def build_route(row: pd.Series):
    """Return the active route for a job."""
    status = str(row["status"]).strip().lower()

    # Delivered jobs do not need an active route
    if status == "delivered":
        return None

    # If parcel is already moving, show driver -> drop
    if status in ["on_route", "delayed", "picked_up", "in_transit"]:
        if pd.notna(row["driver_lat"]) and pd.notna(row["driver_lon"]):
            return [
                [row["driver_lon"], row["driver_lat"]],
                [row["drop_lon"], row["drop_lat"]],
            ]

    # Otherwise show pickup -> drop
    return [
        [row["pickup_lon"], row["pickup_lat"]],
        [row["drop_lon"], row["drop_lat"]],
    ]


def compute_driver_distance(row: pd.Series) -> str:
    status = str(row.get("status", "")).strip().lower()

    if status == "delivered":
        return "Completed"

    if row["is_assigned"] and status in ["on_route", "delayed", "picked_up", "in_transit"]:
        if pd.notna(row.get("drop_lat")) and pd.notna(row.get("drop_lon")):
            dist = haversine_km(
                row["driver_lat"],
                row["driver_lon"],
                row["drop_lat"],
                row["drop_lon"],
            )
            return f"{dist} km to drop"

    return f"{row['distance_to_pickup_km']} km to pickup"


st.set_page_config(page_title="Delivery Risk Dashboard", layout="wide")

st.title("Delivery Risk Dashboard — Phase 1")
st.caption("Live operations view for continuous incoming jobs and driver monitoring.")

with st.sidebar:
    st.header("Data Sources")
    telemetry_path = st.text_input("Telemetry CSV path", value="data/driver_locations.csv")
    fallback_region = st.selectbox("Fallback location region", options=["UK", "US"], index=0)

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
    risk_filter = st.multiselect(
        "Risk level",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    alert_filter = st.multiselect(
        "Alert level",
        options=["urgent", "high", "normal"],
        default=["urgent", "high", "normal"],
    )

# Load and transform operations data
df = load_data("data/dirtyFile.csv")
df_clean = clean_data(df)
df_final = apply_risk_logic(df_clean)

ops_df = build_phase1_operational_view(
    df_final,
    config=AlertConfig(
        high_risk_threshold=high_risk_threshold,
        urgent_delay_minutes=urgent_delay_minutes,
    ),
    telemetry_path=telemetry_path,
    fallback_region=fallback_region.lower(),
)

ops_df = ops_df[
    ops_df["risk_level"].isin(risk_filter) & ops_df["alert_level"].isin(alert_filter)
].copy()

ops_df = ops_df.sort_values(
    by=["alert_level", "risk_score", "delay"],
    ascending=[True, False, False],
)

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
    st.warning(
        f"Telemetry file `{telemetry_path}` not found. "
        f"Showing synthetic fallback coordinates in {fallback_region}."
    )

st.subheader("Live Driver Map")
st.caption(
    "Gold + black border = closest available driver, red/orange/green = assigned driver risk, "
    "blue = nearby available, grey = idle, blue dots = pending pickup, purple dots = drop. "
    "Active jobs show route from driver to drop."
)

if ops_df.empty:
    st.info("No rows match current filters, so no drivers are displayed on the map.")
else:
    selected_job = ops_df.iloc[0]
    pickup_lat = float(selected_job["pickup_lat"])
    pickup_lon = float(selected_job["pickup_lon"])

    # Full fleet telemetry
    fleet_df = normalize_fleet_telemetry(telemetry_path)

    # Assigned driver/job info from visible jobs
    assigned_info = ops_df[
        ["driver_id", "job_id", "alert_level", "risk_score", "status", "drop_lat", "drop_lon"]
    ].copy()
    assigned_info["driver_id"] = assigned_info["driver_id"].astype(str).str.strip().str.lower()
    assigned_info = assigned_info.dropna(subset=["driver_id"])
    assigned_info = assigned_info[assigned_info["driver_id"] != ""]
    assigned_info = assigned_info[assigned_info["driver_id"] != "nan"]

    # Merge assignment info into full fleet
    fleet_df = fleet_df.merge(assigned_info, on="driver_id", how="left")
    fleet_df["is_assigned"] = fleet_df["job_id"].notna()

    # Distance to selected pickup
    fleet_df["distance_to_pickup_km"] = fleet_df.apply(
        lambda row: haversine_km(
            row["driver_lat"],
            row["driver_lon"],
            pickup_lat,
            pickup_lon,
        ),
        axis=1,
    )

    fleet_df["driver_status"] = fleet_df.apply(classify_driver, axis=1)

    # Closest available driver
    fleet_df["is_closest_driver"] = False
    available_mask = ~fleet_df["is_assigned"]

    if available_mask.any():
        closest_idx = fleet_df.loc[available_mask, "distance_to_pickup_km"].idxmin()
    else:
        closest_idx = fleet_df["distance_to_pickup_km"].idxmin()

    fleet_df.loc[closest_idx, "is_closest_driver"] = True

    # Styling
    fleet_df["fill_color"] = fleet_df.apply(get_driver_fill, axis=1)
    fleet_df["line_color"] = fleet_df["is_closest_driver"].apply(
        lambda x: [0, 0, 0, 255] if x else [0, 0, 0, 0]
    )
    fleet_df["line_width"] = fleet_df["is_closest_driver"].apply(lambda x: 4 if x else 0)
    fleet_df["radius"] = fleet_df.apply(
        lambda row: 260 if row["is_closest_driver"]
        else 200 if row["driver_status"] == "nearby_available"
        else 180,
        axis=1,
    )

    # Tooltip fields for drivers
    fleet_df["tooltip_title"] = fleet_df["driver_id"].fillna("Unknown driver")
    fleet_df["tooltip_type"] = fleet_df["driver_status"].fillna("unknown")
    fleet_df["tooltip_job"] = fleet_df["job_id"].fillna("Unassigned")
    fleet_df["tooltip_driver"] = fleet_df["driver_id"].fillna("Unknown driver")
    fleet_df["tooltip_status"] = fleet_df["status"].fillna("unknown")
    fleet_df["tooltip_alert"] = fleet_df["alert_level"].fillna("-")
    fleet_df["tooltip_distance"] = fleet_df.apply(compute_driver_distance, axis=1)

    # Tooltip fields for jobs
    ops_df["tooltip_title"] = ops_df["job_id"]
    ops_df["tooltip_job"] = ops_df["job_id"]
    ops_df["tooltip_driver"] = ops_df["driver_id"].astype(str).replace("nan", "").str.strip()
    ops_df["tooltip_driver"] = ops_df["tooltip_driver"].replace("", "Unassigned")
    ops_df["tooltip_status"] = ops_df["status"].fillna("unknown")
    ops_df["tooltip_alert"] = ops_df["alert_level"].fillna("unknown")
    ops_df["tooltip_distance"] = "-"

    # Pickup points only for jobs not yet actively moving or delivered
    pickup_df = ops_df[
        ~ops_df["status"].astype(str).str.strip().str.lower().isin(
            ["on_route", "delayed", "picked_up", "in_transit", "delivered"]
        )
    ].copy()
    pickup_df["tooltip_type"] = "pickup"

    drop_df = ops_df.copy()
    drop_df["tooltip_type"] = "drop"

    # Route data
    ops_df["route"] = ops_df.apply(build_route, axis=1)
    route_df = ops_df[ops_df["route"].notna()].copy()

    route_df["route_color"] = route_df["alert_level"].map(
        {
            "urgent": [255, 59, 48, 160], # red
            "high": [255, 149, 0, 150], # orange
            "normal": [90, 90, 90, 120], # grey
        }
    )

    # Layers
    pickup_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pickup_df,
        get_position="[pickup_lon, pickup_lat]",
        get_fill_color=[0, 122, 255, 200],
        get_radius=120,
        pickable=True,
    )

    drop_layer = pdk.Layer(
        "ScatterplotLayer",
        data=drop_df,
        get_position="[drop_lon, drop_lat]",
        get_fill_color=[175, 82, 222, 170],
        get_radius=120,
        pickable=True,
    )

    map_layer = pdk.Layer(
        "ScatterplotLayer",
        data=fleet_df,
        get_position="[driver_lon, driver_lat]",
        get_fill_color="fill_color",
        get_line_color="line_color",
        get_line_width="line_width",
        get_radius="radius",
        stroked=True,
        filled=True,
        pickable=True,
    )

    route_layer = pdk.Layer(
        "LineLayer",
        data=route_df,
        get_source_position="route[0]",
        get_target_position="route[1]",
        get_color="route_color",
        get_width=3,
    )

    world_map_view = map_mode == "World (global view)"
    if world_map_view:
        center_lat, center_lon, zoom = 20.0, 0.0, 1
    else:
        center_lat, center_lon, zoom = 54.5, -2.5, 5

    style_lookup = {
        "Light": "light",
        "Dark": "dark",
        "Satellite": "road",
    }

    deck = pdk.Deck(
        map_style=style_lookup[map_style],
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom,
            pitch=0,
        ),
        layers=[route_layer, pickup_layer, drop_layer, map_layer],
        tooltip={
            "text": "ID: {tooltip_title}\nType: {tooltip_type}\nJob: {tooltip_job}\nAssigned Driver: {tooltip_driver}\nStatus: {tooltip_status}\nAlert: {tooltip_alert}\nDistance: {tooltip_distance}"
        },
    )
    st.pydeck_chart(deck, use_container_width=True)

    if world_map_view:
        st.caption("World view enabled.")
    else:
        st.caption("UK view enabled.")

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