from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd


@dataclass(frozen=True)
class AlertConfig:
    high_risk_threshold: int = 70
    medium_risk_threshold: int = 40
    urgent_delay_minutes: int = 30


def attach_driver_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Phase-1 placeholder for real-time location / heartbeat data.

    In production this should join against a live driver telemetry source.
    For now, we attach stable synthetic coordinates so the dashboard can
    demonstrate the operational workflow.
    """
    out = df.copy()

    # deterministic pseudo-coordinates by driver id for demo dashboards
    driver_nums = out['driver_id'].str.extract(r"(\d+)", expand=False).fillna('0').astype(int)
    out['driver_lat'] = 40.0 + (driver_nums % 20) * 0.01
    out['driver_lon'] = -74.0 - (driver_nums % 20) * 0.01
    out['last_telemetry_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    return out


def attach_external_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Attach traffic/weather effect fields used for operational alerts."""
    out = df.copy()

    traffic_factor = out['traffic_level'].map({'low': 1.0, 'medium': 1.1, 'heavy': 1.25}).fillna(1.0)
    out['weather_severity'] = out['traffic_level'].map({'low': 'low', 'medium': 'moderate', 'heavy': 'high'}).fillna('low')

    out['expected_delivery_time'] = (out['scheduled_time'] * traffic_factor).round(1)
    out['eta_drift'] = (out['expected_delivery_time'] - out['scheduled_time']).round(1)
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


def build_phase1_operational_view(df: pd.DataFrame, config: AlertConfig | None = None) -> pd.DataFrame:
    """Single entrypoint used by CLI and dashboard for phase-1 operations."""
    out = attach_driver_telemetry(df)
    out = attach_external_signals(out)
    out = attach_alerts(out, config=config)
    return out