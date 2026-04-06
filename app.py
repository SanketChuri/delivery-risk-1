
from pathlib import Path
import math

import pandas as pd
import pydeck as pdk
import streamlit as st

from data_cleaning import load_data, clean_data
from risk_engine import apply_risk_logic
from phase1 import AlertConfig, build_phase1_operational_view

from llm_agent import generate_ai_brief


# ----------------------------
# Constants
# ----------------------------
STATUS_OPTIONS = [
    "pending", "assigned", "scheduled", "awaiting_pickup",
    "on_route", "picked_up", "in_transit", "delayed", "delivered"
]

PRE_PICKUP_STATUSES = ["pending", "assigned", "scheduled", "awaiting_pickup"]
ACTIVE_MOVE_STATUSES = ["on_route", "delayed", "picked_up", "in_transit"]

TRUCK_ICON = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f69a.png"
BOX_ICON = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f4e6.png"
FLAG_ICON = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f3c1.png"
STAR_ICON = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/2b50.png"

CITY_BOXES = {
    "London": (51.28, 51.70, -0.50, 0.20),
    "Manchester": (53.35, 53.60, -2.40, -2.10),
    "Birmingham": (52.35, 52.60, -2.05, -1.70),
    "Leeds": (53.70, 53.90, -1.70, -1.40),
    "Glasgow": (55.78, 55.95, -4.40, -4.10),
    "Edinburgh": (55.88, 56.02, -3.35, -3.05),
    "Newcastle": (54.90, 55.05, -1.75, -1.50),
    "Liverpool": (53.33, 53.47, -3.10, -2.85),
    "Bristol": (51.38, 51.52, -2.70, -2.50),
    "Sheffield": (53.30, 53.43, -1.60, -1.35),
    "Cardiff": (51.43, 51.55, -3.30, -3.05),
    "Nottingham": (52.90, 53.02, -1.25, -1.05),
}


# ----------------------------
# Helpers
# ----------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two lat/lon points in kilometers."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return round(r * c, 2)


@st.cache_data(show_spinner=False)
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
    return fleet_df.dropna(subset=["driver_lat", "driver_lon"]).copy()


def classify_driver(row: pd.Series) -> str:
    status = str(row.get("status", "")).strip().lower()

    if status == "delivered":
        return "nearby_available" if row["distance_to_pickup_km"] <= 20 else "idle"

    if row["is_assigned"]:
        return "assigned"

    return "nearby_available" if row["distance_to_pickup_km"] <= 20 else "idle"


def get_driver_fill(row: pd.Series) -> list[int]:
    if row["is_closest_driver"]:
        return [255, 215, 0, 230]
    if row["is_assigned"]:
        if row.get("alert_level") == "urgent":
            return [255, 59, 48, 210]
        if row.get("alert_level") == "high":
            return [255, 149, 0, 200]
        return [52, 199, 89, 190]
    if row["driver_status"] == "nearby_available":
        return [0, 122, 255, 200]
    return [142, 142, 147, 120]


def build_route(row: pd.Series):
    status = str(row["status"]).strip().lower()

    if status == "delivered":
        return None

    if status in ACTIVE_MOVE_STATUSES and pd.notna(row["driver_lat"]) and pd.notna(row["driver_lon"]):
        return [
            [row["driver_lon"], row["driver_lat"]],
            [row["drop_lon"], row["drop_lat"]],
        ]

    return [
        [row["pickup_lon"], row["pickup_lat"]],
        [row["drop_lon"], row["drop_lat"]],
    ]


def compute_driver_distance(row: pd.Series) -> str:
    status = str(row.get("status", "")).strip().lower()

    if status == "delivered":
        return "Completed"

    if row["is_assigned"] and status in ACTIVE_MOVE_STATUSES:
        if pd.notna(row.get("drop_lat")) and pd.notna(row.get("drop_lon")):
            dist = haversine_km(
                row["driver_lat"],
                row["driver_lon"],
                row["drop_lat"],
                row["drop_lon"],
            )
            return f"{dist} km to drop"

    return f"{row['distance_to_pickup_km']} km to pickup"


def safe_text(value, fallback: str = "-") -> str:
    if pd.isna(value):
        return fallback
    value = str(value).strip()
    return fallback if value == "" or value.lower() == "nan" else value


def make_icon(url: str, size: int = 72) -> dict:
    return {"url": url, "width": size, "height": size, "anchorY": size}


def contains_filter(series: pd.Series, text: str) -> pd.Series:
    text = text.strip().upper()
    if not text:
        return pd.Series([True] * len(series), index=series.index)
    return series.astype(str).str.strip().str.upper().str.contains(text, na=False)


def apply_region_filter(df: pd.DataFrame, region_filter: str) -> pd.DataFrame:
    if df.empty or region_filter == "All UK":
        return df
    min_lat, max_lat, min_lon, max_lon = CITY_BOXES[region_filter]
    return df[
        df["pickup_lat"].between(min_lat, max_lat) &
        df["pickup_lon"].between(min_lon, max_lon)
    ]


def get_map_center(region_filter: str, world_map_view: bool, job_df: pd.DataFrame, driver_df: pd.DataFrame):
    if world_map_view:
        return 20.0, 0.0, 1

    if not job_df.empty:
        first_row = job_df.iloc[0]
        lat = pd.to_numeric(first_row.get("pickup_lat"), errors="coerce")
        lon = pd.to_numeric(first_row.get("pickup_lon"), errors="coerce")
        if pd.notna(lat) and pd.notna(lon):
            if region_filter != "All UK":
                return lat, lon, 9
            return lat, lon, 6

    if not driver_df.empty:
        first_driver = driver_df.iloc[0]
        lat = pd.to_numeric(first_driver.get("driver_lat"), errors="coerce")
        lon = pd.to_numeric(first_driver.get("driver_lon"), errors="coerce")
        if pd.notna(lat) and pd.notna(lon):
            if region_filter != "All UK":
                return lat, lon, 9
            return lat, lon, 6

    if region_filter != "All UK":
        min_lat, max_lat, min_lon, max_lon = CITY_BOXES[region_filter]
        return (min_lat + max_lat) / 2, (min_lon + max_lon) / 2, 9

    return 54.5, -2.5, 5


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Delivery Risk Dashboard", layout="wide")

st.title("Delivery Risk Dashboard — Phase 1")
st.caption("Live operations view for continuous incoming jobs and driver monitoring.")


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Data Sources")
    orders_path = st.text_input("Orders CSV path", value="data/orders_with_locations.csv")
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

    st.header("Table Filters")
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

    st.header("Map Filters")
    map_status_filter = st.multiselect(
        "Show job statuses",
        options=STATUS_OPTIONS,
        default=[status for status in STATUS_OPTIONS if status != "delivered"],
    )
    show_unassigned_only = st.checkbox("Show only unassigned jobs", value=False)
    show_nearby_available_only = st.checkbox("Show only nearby available drivers", value=False)
    hide_delivered = st.checkbox("Hide delivered jobs", value=True)
    show_routes = st.checkbox("Show routes", value=True)
    max_jobs_on_map = st.slider("Max jobs on map", min_value=10, max_value=100, value=40, step=10)
    region_filter = st.selectbox("Region focus", options=["All UK"] + list(CITY_BOXES.keys()), index=0)

    st.header("Search Filters")
    job_id_search = st.text_input("Search Job ID", value="", placeholder="e.g. J001")
    driver_id_search = st.text_input("Search Driver ID", value="", placeholder="e.g. D01")


# ----------------------------
# Load and transform operations data
# ----------------------------
df = load_data(orders_path)
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
).copy()

ops_df = ops_df[ops_df["risk_level"].isin(risk_filter) & ops_df["alert_level"].isin(alert_filter)].copy()

ops_df["job_id"] = ops_df["job_id"].astype(str).str.strip().str.upper()
ops_df["driver_id"] = ops_df["driver_id"].astype(str).str.strip().str.upper()
ops_df["status"] = ops_df["status"].astype(str).str.strip().str.lower()

for col in ["pickup_lat", "pickup_lon", "drop_lat", "drop_lon", "driver_lat", "driver_lon"]:
    if col in ops_df.columns:
        ops_df[col] = pd.to_numeric(ops_df[col], errors="coerce")

ops_df = ops_df[contains_filter(ops_df["job_id"], job_id_search)]
ops_df = ops_df[contains_filter(ops_df["driver_id"], driver_id_search)]

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
    st.warning(
        f"Telemetry file `{telemetry_path}` not found. "
        f"Showing synthetic fallback coordinates in {fallback_region}."
    )

st.subheader("Live Driver Map")
st.caption("⭐ = closest available driver, 🚚 = driver, 📦 = pickup, 🏁 = drop.")

# ----------------------------
# Build job map dataframe
# ----------------------------
map_df = ops_df.copy()

if map_status_filter:
    map_df = map_df[map_df["status"].isin(map_status_filter)]
else:
    map_df = map_df.iloc[0:0]

if hide_delivered:
    map_df = map_df[map_df["status"] != "delivered"]

if show_unassigned_only:
    map_df = map_df[map_df["driver_id"].isin(["", "NAN"])]

required_location_cols = ["pickup_lat", "pickup_lon", "drop_lat", "drop_lon"]
missing_cols = [col for col in required_location_cols if col not in map_df.columns]
if missing_cols:
    st.error(f"Missing required columns in orders data: {missing_cols}")
    st.stop()

map_df = map_df.dropna(subset=required_location_cols)
map_df = apply_region_filter(map_df, region_filter)
map_df = map_df.head(max_jobs_on_map).copy()

# ----------------------------
# Build fleet dataframe independently so idle-driver search still works
# ----------------------------
if Path(telemetry_path).exists():
    fleet_df = normalize_fleet_telemetry(telemetry_path).copy()
else:
    fleet_df = pd.DataFrame(columns=["driver_id", "driver_lat", "driver_lon"])

fleet_df = fleet_df[contains_filter(fleet_df["driver_id"], driver_id_search)].copy()

if map_df.empty and fleet_df.empty:
    st.info("No rows match current job or driver filters, so nothing is shown. Splendid efficiency.")
    st.stop()

if not map_df.empty:
    selected_job = map_df.iloc[0]
    focus_lat = pd.to_numeric(selected_job.get("pickup_lat"), errors="coerce")
    focus_lon = pd.to_numeric(selected_job.get("pickup_lon"), errors="coerce")
elif not fleet_df.empty:
    selected_driver = fleet_df.iloc[0]
    focus_lat = pd.to_numeric(selected_driver.get("driver_lat"), errors="coerce")
    focus_lon = pd.to_numeric(selected_driver.get("driver_lon"), errors="coerce")
else:
    st.info("Nothing to display.")
    st.stop()

if pd.isna(focus_lat) or pd.isna(focus_lon):
    st.error("Could not determine map center from selected job or driver.")
    st.stop()

# Merge assignment info from jobs into telemetry drivers
if not fleet_df.empty:
    assigned_info = map_df[
        ["driver_id", "job_id", "alert_level", "risk_score", "status", "drop_lat", "drop_lon"]
    ].copy() if not map_df.empty else pd.DataFrame(
        columns=["driver_id", "job_id", "alert_level", "risk_score", "status", "drop_lat", "drop_lon"]
    )

    assigned_info["driver_id"] = assigned_info["driver_id"].astype(str).str.strip().str.lower()
    assigned_info = assigned_info[assigned_info["driver_id"].notna()]
    assigned_info = assigned_info[~assigned_info["driver_id"].isin(["", "nan"])]

    fleet_df = fleet_df.merge(assigned_info, on="driver_id", how="left")
    fleet_df["status"] = fleet_df["status"].astype(str).str.strip().str.lower()
    fleet_df["is_assigned"] = fleet_df["job_id"].notna() & (fleet_df["status"] != "delivered")

    # If no jobs are in map_df, still keep drivers visible by using focus point distance.
    fleet_df["distance_to_pickup_km"] = fleet_df.apply(
        lambda row: haversine_km(row["driver_lat"], row["driver_lon"], focus_lat, focus_lon),
        axis=1,
    ) if not fleet_df.empty else 0.0

    fleet_df["driver_status"] = fleet_df.apply(classify_driver, axis=1)
    fleet_df["is_closest_driver"] = False

    if not fleet_df.empty:
        available_mask = ~fleet_df["is_assigned"]
        closest_idx = (
            fleet_df.loc[available_mask, "distance_to_pickup_km"].idxmin()
            if available_mask.any()
            else fleet_df["distance_to_pickup_km"].idxmin()
        )
        fleet_df.loc[closest_idx, "is_closest_driver"] = True

    fleet_df["fill_color"] = fleet_df.apply(get_driver_fill, axis=1)

    if show_nearby_available_only:
        fleet_df = fleet_df[fleet_df["driver_status"] == "nearby_available"]

    fleet_df["tooltip_title"] = fleet_df["driver_id"].fillna("Unknown driver")
    fleet_df["tooltip_type"] = fleet_df["driver_status"].fillna("unknown")
    fleet_df["tooltip_job"] = fleet_df["job_id"].fillna("Unassigned")
    fleet_df["tooltip_driver"] = fleet_df["driver_id"].fillna("Unknown driver")
    fleet_df["tooltip_status"] = fleet_df["driver_status"]
    fleet_df["tooltip_alert"] = fleet_df["alert_level"].fillna("-")
    fleet_df["tooltip_distance"] = fleet_df.apply(compute_driver_distance, axis=1)

    fleet_df["icon_data"] = fleet_df["is_closest_driver"].apply(
        lambda is_closest: make_icon(STAR_ICON, 72) if is_closest else make_icon(TRUCK_ICON, 72)
    )
    fleet_df["icon_size"] = fleet_df["is_closest_driver"].apply(lambda is_closest: 5 if is_closest else 4)

# ----------------------------
# Build job layers only if jobs exist
# ----------------------------
if not map_df.empty:
    map_df["tooltip_title"] = map_df["job_id"]
    map_df["tooltip_job"] = map_df["job_id"]
    map_df["tooltip_driver"] = map_df["driver_id"].apply(lambda value: safe_text(value, "Unassigned"))
    map_df["tooltip_status"] = map_df["status"].fillna("unknown")
    map_df["tooltip_alert"] = map_df["alert_level"].fillna("unknown")
    map_df["tooltip_distance"] = "-"

    pickup_df = map_df[map_df["status"].isin(PRE_PICKUP_STATUSES)].copy()
    pickup_df["tooltip_type"] = "pickup"
    pickup_df["icon_data"] = [make_icon(BOX_ICON, 72)] * len(pickup_df)
    pickup_df["icon_size"] = 3

    drop_df = map_df[map_df["status"] != "delivered"].copy()
    drop_df["tooltip_type"] = "drop"
    drop_df["icon_data"] = [make_icon(FLAG_ICON, 72)] * len(drop_df)
    drop_df["icon_size"] = 3

    map_df["route"] = map_df.apply(build_route, axis=1)
    route_df = map_df[map_df["route"].notna()].copy()
    route_df["route_color"] = route_df["alert_level"].map(
        {
            "urgent": [255, 59, 48, 160],
            "high": [255, 149, 0, 150],
            "normal": [90, 90, 90, 120],
        }
    )
else:
    pickup_df = pd.DataFrame()
    drop_df = pd.DataFrame()
    route_df = pd.DataFrame()

# ----------------------------
# Layers
# ----------------------------
layers = []

if show_routes and not route_df.empty:
    layers.append(
        pdk.Layer(
            "LineLayer",
            data=route_df,
            get_source_position="route[0]",
            get_target_position="route[1]",
            get_color="route_color",
            get_width=3,
        )
    )

if not pickup_df.empty:
    layers.append(
        pdk.Layer(
            "IconLayer",
            data=pickup_df,
            get_icon="icon_data",
            get_position="[pickup_lon, pickup_lat]",
            get_size="icon_size",
            size_scale=10,
            pickable=True,
        )
    )

if not drop_df.empty:
    layers.append(
        pdk.Layer(
            "IconLayer",
            data=drop_df,
            get_icon="icon_data",
            get_position="[drop_lon, drop_lat]",
            get_size="icon_size",
            size_scale=10,
            pickable=True,
        )
    )

if not fleet_df.empty:
    layers.append(
        pdk.Layer(
            "IconLayer",
            data=fleet_df,
            get_icon="icon_data",
            get_position="[driver_lon, driver_lat]",
            get_size="icon_size",
            size_scale=10,
            pickable=True,
        )
    )

world_map_view = map_mode == "World (global view)"
center_lat, center_lon, zoom = get_map_center(region_filter, world_map_view, map_df, fleet_df)

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
    layers=layers,
    tooltip={
        "text": "ID: {tooltip_title}\nType: {tooltip_type}\nJob: {tooltip_job}\nAssigned Driver: {tooltip_driver}\nStatus: {tooltip_status}\nAlert: {tooltip_alert}\nDistance: {tooltip_distance}"
    },
)
st.pydeck_chart(deck, use_container_width=True)

if world_map_view:
    st.caption("World view enabled.")
elif region_filter != "All UK":
    st.caption(f"{region_filter} view enabled.")
elif not map_df.empty:
    st.caption("Job-focused UK view enabled.")
else:
    st.caption("Driver-focused UK view enabled.")

# ----------------------------
# Table and charts
# ----------------------------
st.subheader("Operational Queue")
table_columns = [
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
    "expected_delivery_time_min",
    "eta_drift",
    "driver_lat",
    "driver_lon",
    "last_telemetry_utc",
]
existing_columns = [col for col in table_columns if col in ops_df.columns]
st.dataframe(ops_df[existing_columns], use_container_width=True)

# ----------------------------
# AI OPS COPILOT (ADD HERE)
# ----------------------------
st.subheader("AI Ops Copilot")

if not ops_df.empty:
    selectable_jobs = ops_df.sort_values(by=["risk_score"], ascending=False)["job_id"].astype(str).tolist()
    selected_job_id = st.selectbox("Select a job for AI analysis", selectable_jobs)

    selected_row = ops_df[ops_df["job_id"].astype(str) == selected_job_id].iloc[0]

    if st.button("Generate AI Brief"):
        with st.spinner("Generating AI brief..."):
            ai_brief = generate_ai_brief(selected_row)

        st.markdown("### Risk Explanation")
        st.write(ai_brief.get("risk_explanation", "-"))

        st.markdown("### Recommended Ops Action")
        st.write(ai_brief.get("ops_recommendation", "-"))

        st.markdown("### Customer Update Draft")
        st.write(ai_brief.get("customer_message", "-"))
else:
    st.info("No jobs available for AI analysis.")

st.subheader("Risk Distribution")
if not ops_df.empty:
    st.bar_chart(ops_df["risk_level"].value_counts())

st.subheader("Alert Distribution")
if not ops_df.empty:
    st.bar_chart(ops_df["alert_level"].value_counts())
