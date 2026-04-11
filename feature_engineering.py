import math
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance between two latitude/longitude points in kilometers."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return None

    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return r * c


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ML-ready features from order/job data.
    Assumes columns are already cleaned and lowercased.
    """
    out = df.copy()

    # Delay
    if "delay" not in out.columns:
        out["delay"] = (out["actual_time"] - out["scheduled_time"]).clip(lower=0)

    # Distance
    out["distance_km"] = out.apply(
        lambda row: haversine_km(
            row.get("pickup_lat"),
            row.get("pickup_lon"),
            row.get("drop_lat"),
            row.get("drop_lon"),
        ),
        axis=1,
    )

    # Time-based simple feature
    out["is_late_start"] = (out["actual_time"] > out["scheduled_time"]).astype(int)

    # Encode traffic severity numerically as extra helper feature
    traffic_map = {"low": 1, "medium": 2, "heavy": 3}
    out["traffic_severity"] = out["traffic_level"].map(traffic_map).fillna(1)

    # Priority helper
    priority_map = {"low": 1, "medium": 2, "high": 3}
    out["priority_score"] = out["priority"].map(priority_map).fillna(1)

    return out


def create_target(df: pd.DataFrame, fail_delay_threshold: int = 15) -> pd.DataFrame:
    """
    Create a simple binary target:
    1 = job likely failed / breached SLA
    0 = job completed within acceptable threshold
    """
    out = df.copy()

    if "delay" not in out.columns:
        out["delay"] = (out["actual_time"] - out["scheduled_time"]).clip(lower=0)

    out["will_fail"] = (out["delay"] > fail_delay_threshold).astype(int)
    return out