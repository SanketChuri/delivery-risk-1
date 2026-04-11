import math
import pandas as pd

MIN_BATCH_SIZE = 3
MAX_BATCH_SIZE = 5
PICKUP_RADIUS_KM = 3.0
DROP_RADIUS_KM = 4.0
URGENT_DELAY_BLOCK = 20


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return round(r * c, 2)


def is_active_job(row):
    return str(row.get("status", "")).strip().lower() != "delivered"


def is_batchable_job(row):
    if not is_active_job(row):
        return False

    alert_level = str(row.get("alert_level", "")).strip().lower()
    delay = float(row.get("delay", 0) or 0)

    if alert_level == "urgent" and delay >= URGENT_DELAY_BLOCK:
        return False

    required = ["pickup_lat", "pickup_lon", "drop_lat", "drop_lon"]
    for col in required:
        if pd.isna(row.get(col)):
            return False

    return True


def pickup_dist(a, b):
    return haversine_km(a["pickup_lat"], a["pickup_lon"], b["pickup_lat"], b["pickup_lon"])


def drop_dist(a, b):
    return haversine_km(a["drop_lat"], a["drop_lon"], b["drop_lat"], b["drop_lon"])


def sort_route(batch_jobs):
    remaining = batch_jobs.copy()
    route = []

    current_lat = remaining["pickup_lat"].mean()
    current_lon = remaining["pickup_lon"].mean()

    # pickups first
    while not remaining.empty:
        remaining = remaining.copy()
        remaining["pickup_step_dist"] = remaining.apply(
            lambda row: haversine_km(current_lat, current_lon, row["pickup_lat"], row["pickup_lon"]),
            axis=1,
        )
        next_idx = remaining["pickup_step_dist"].idxmin()
        next_row = remaining.loc[next_idx]

        route.append(f"PICKUP:{next_row['job_id']}")
        current_lat = next_row["pickup_lat"]
        current_lon = next_row["pickup_lon"]

        remaining = remaining.drop(index=next_idx)

    remaining = batch_jobs.copy()

    # then drops
    while not remaining.empty:
        remaining = remaining.copy()
        remaining["drop_step_dist"] = remaining.apply(
            lambda row: haversine_km(current_lat, current_lon, row["drop_lat"], row["drop_lon"]),
            axis=1,
        )
        next_idx = remaining["drop_step_dist"].idxmin()
        next_row = remaining.loc[next_idx]

        route.append(f"DROP:{next_row['job_id']}")
        current_lat = next_row["drop_lat"]
        current_lon = next_row["drop_lon"]

        remaining = remaining.drop(index=next_idx)

    return route


def build_multi_pickup_batches(ops_df: pd.DataFrame) -> pd.DataFrame:
    if ops_df.empty:
        return pd.DataFrame()

    df = ops_df.copy()

    numeric_cols = ["pickup_lat", "pickup_lon", "drop_lat", "drop_lon", "delay", "risk_score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df.apply(is_batchable_job, axis=1)].copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(by=["delay", "risk_score"], ascending=[True, True]).copy()

    used = set()
    batches = []

    for seed_idx in df.index:
        if seed_idx in used:
            continue

        seed = df.loc[seed_idx]
        chosen = [seed_idx]

        for other_idx in df.index:
            if other_idx == seed_idx or other_idx in used or other_idx in chosen:
                continue

            other = df.loc[other_idx]

            # every new job must fit tightly with all already chosen jobs
            pickup_ok = all(pickup_dist(df.loc[c], other) <= PICKUP_RADIUS_KM for c in chosen)
            drop_ok = all(drop_dist(df.loc[c], other) <= DROP_RADIUS_KM for c in chosen)

            if pickup_ok and drop_ok:
                chosen.append(other_idx)

            if len(chosen) >= MAX_BATCH_SIZE:
                break

        if len(chosen) >= MIN_BATCH_SIZE:
            chosen_df = df.loc[chosen].copy()

            # extra sanity check: tight pickup cluster and tight drop cluster
            pickup_center_lat = chosen_df["pickup_lat"].mean()
            pickup_center_lon = chosen_df["pickup_lon"].mean()
            drop_center_lat = chosen_df["drop_lat"].mean()
            drop_center_lon = chosen_df["drop_lon"].mean()

            chosen_df["pickup_spread"] = chosen_df.apply(
                lambda row: haversine_km(
                    pickup_center_lat, pickup_center_lon,
                    row["pickup_lat"], row["pickup_lon"]
                ),
                axis=1,
            )
            chosen_df["drop_spread"] = chosen_df.apply(
                lambda row: haversine_km(
                    drop_center_lat, drop_center_lon,
                    row["drop_lat"], row["drop_lon"]
                ),
                axis=1,
            )

            if chosen_df["pickup_spread"].max() > PICKUP_RADIUS_KM:
                continue
            if chosen_df["drop_spread"].max() > DROP_RADIUS_KM:
                continue

            batches.append({
                "batch_id": f"BATCH_{'_'.join(chosen_df['job_id'].astype(str).tolist())}",
                "jobs": chosen_df["job_id"].astype(str).tolist(),
                "batch_size": len(chosen_df),
                "pickup_zone_center": f"{round(pickup_center_lat, 4)}, {round(pickup_center_lon, 4)}",
                "drop_zone_center": f"{round(drop_center_lat, 4)}, {round(drop_center_lon, 4)}",
                "route_order": sort_route(chosen_df),
                "avg_risk_score": round(chosen_df["risk_score"].mean(), 1),
                "max_delay": round(chosen_df["delay"].max(), 1),
                "batch_type": "pickup_cluster_to_drop_cluster",
                "batch_note": (
                    f"{len(chosen_df)} jobs share one pickup cluster and one drop cluster."
                ),
            })

            for idx in chosen:
                used.add(idx)

    if not batches:
        return pd.DataFrame()

    return pd.DataFrame(batches).sort_values(
        by=["batch_size", "avg_risk_score", "max_delay"],
        ascending=[False, False, False]
    ).reset_index(drop=True)