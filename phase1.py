from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AlertConfig:
    high_risk_threshold: int = 70
    medium_risk_threshold: int = 40
    urgent_delay_minutes: int = 30


def _normalize_telemetry_columns(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize accepted telemetry column names to driver_lat/driver_lon."""
    out = telemetry_df.copy()
    rename_map = {
        'lat': 'driver_lat',
        'latitude': 'driver_lat',
        'lon': 'driver_lon',
        'lng': 'driver_lon',
        'longitude': 'driver_lon',
        'timestamp': 'last_telemetry_utc',
    }
    out.columns = [c.strip().lower() for c in out.columns]
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    required = {'driver_id', 'driver_lat', 'driver_lon'}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Telemetry file missing required columns: {sorted(missing)}")

    if 'last_telemetry_utc' not in out.columns:
        out['last_telemetry_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')

    out = out[['driver_id', 'driver_lat', 'driver_lon', 'last_telemetry_utc']].copy()
    out['driver_id'] = out['driver_id'].astype(str).str.strip().str.lower()
    out['driver_lat'] = pd.to_numeric(out['driver_lat'], errors='coerce')
    out['driver_lon'] = pd.to_numeric(out['driver_lon'], errors='coerce')
    out = out.dropna(subset=['driver_lat', 'driver_lon'])
    return out


def _fallback_synthetic_telemetry(df: pd.DataFrame, region: str = 'uk') -> pd.DataFrame:
    out = df.copy()
    driver_nums = out['driver_id'].str.extract(r"(\d+)", expand=False).fillna('0').astype(int)
    if region.lower() == 'us':
        base_lat, base_lon = 40.0, -74.0
    else:
        # Default to London area so fallback is UK-friendly.
        base_lat, base_lon = 51.5074, -0.1278
    out['driver_lat'] = base_lat + (driver_nums % 20) * 0.01
    out['driver_lon'] = base_lon - (driver_nums % 20) * 0.01
    out['last_telemetry_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    return out


def attach_driver_telemetry(
    df: pd.DataFrame,
    telemetry_path: str | None = None,
    fallback_region: str = 'uk',
) -> pd.DataFrame:
    """Attach driver locations.

    If telemetry_path exists, use real coordinates from that file.
    Otherwise, fallback to deterministic synthetic coordinates.
    """
    out = df.copy()

    if telemetry_path:
        path = Path(telemetry_path)
        if path.exists():
            telemetry_df = pd.read_csv(path)
            telemetry_df = _normalize_telemetry_columns(telemetry_df)
            out['driver_id'] = out['driver_id'].astype(str).str.strip().str.lower()
            out = out.merge(telemetry_df, how='left', on='driver_id')

            needs_fallback = out['driver_lat'].isna() | out['driver_lon'].isna()
            if needs_fallback.any():
                fallback = _fallback_synthetic_telemetry(
                    out.loc[needs_fallback, ['driver_id']].copy(),
                    region=fallback_region,
                )
                out.loc[needs_fallback, 'driver_lat'] = fallback['driver_lat'].values
                out.loc[needs_fallback, 'driver_lon'] = fallback['driver_lon'].values
                out.loc[needs_fallback, 'last_telemetry_utc'] = fallback['last_telemetry_utc'].values
            return out

    return _fallback_synthetic_telemetry(out, region=fallback_region)


def attach_external_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Attach traffic/weather effect fields used for operational alerts."""
    out = df.copy()

    traffic_factor = out['traffic_level'].map({'low': 1.0, 'medium': 1.1, 'heavy': 1.25}).fillna(1.0)
    out['weather_severity'] = out['traffic_level'].map({'low': 'low', 'medium': 'moderate', 'heavy': 'high'}).fillna('low')

    out['expected_delivery_time_min'] = (out['scheduled_time'] * traffic_factor).round(1)
    out['eta_drift'] = (out['expected_delivery_time_min'] - out['scheduled_time']).round(1)
    return out


def attach_alerts(df: pd.DataFrame, config: AlertConfig | None = None) -> pd.DataFrame:
    """Classify rows for dashboard urgency and action queue."""
    cfg = config or AlertConfig()
    out = df.copy()

    urgent_mask = (out['risk_score'] >= cfg.high_risk_threshold) | (out['delay'] >= cfg.urgent_delay_minutes)
    high_mask = (~urgent_mask) & (out['risk_score'] >= cfg.medium_risk_threshold)

    out['alert_level'] = 'normal'
    out.loc[high_mask, 'alert_level'] = 'high'
    out.loc[urgent_mask, 'alert_level'] = 'urgent'

    out['ops_action'] = out['alert_level'].map(
        {
            'urgent': 'Escalate now: reroute/reassign and notify customer',
            'high': 'Monitor actively and pre-stage backup driver',
            'normal': 'No immediate intervention',
        }
    )
    out['alert_created_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    return out


def build_phase1_operational_view(
    df: pd.DataFrame,
    config: AlertConfig | None = None,
    telemetry_path: str | None = None,
    fallback_region: str = 'uk',
) -> pd.DataFrame:
    """Single entrypoint used by CLI and dashboard for phase-1 operations."""
    out = attach_driver_telemetry(df, telemetry_path=telemetry_path, fallback_region=fallback_region)
    out = attach_external_signals(out)
    out = attach_alerts(out, config=config)
    return out