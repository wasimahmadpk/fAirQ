"""Join measurements + weather and add a few simple columns."""

from __future__ import annotations

import pandas as pd

from fairq.db import connect

STATION_CODE = {"mc032": 0, "mc042": 1, "mc174": 2}
# Same-pollutant lags: recent hours (odd steps up to 12) plus yesterday.
LAGS = (1, 3, 5, 7, 9, 11, 12, 24)
FEATURE_COLS = [
    "station_code",
    "hour",
    "weekday",
    "temperature_c",
    "wind_speed_ms",
    "wind_direction_deg",
    "precipitation_mm",
    *[f"lag_{k}" for k in LAGS],
]


def feature_vector(
    station_code: int,
    hour: int,
    weekday: int,
    temperature_c: float,
    wind_speed_ms: float,
    wind_direction_deg: float,
    precipitation_mm: float,
    past: list[float],
) -> list[float]:
    return [
        station_code,
        hour,
        weekday,
        temperature_c,
        wind_speed_ms,
        wind_direction_deg,
        precipitation_mm,
        *[past[-k] for k in LAGS],
    ]


def load_frame() -> pd.DataFrame:
    db = connect()
    air = db.query_df(
        "SELECT station_id, observed_at, pollutant, value FROM measurements"
    )
    weather = db.query_df("SELECT * FROM weather")
    air["observed_at"] = pd.to_datetime(air["observed_at"]).dt.floor("h")
    weather["observed_at"] = pd.to_datetime(weather["observed_at"]).dt.floor("h")
    frame = air.merge(weather, on="observed_at", how="left")
    frame = frame.sort_values(["station_id", "pollutant", "observed_at"])
    group = frame.groupby(["station_id", "pollutant"], sort=False)["value"]
    for k in LAGS:
        frame[f"lag_{k}"] = group.shift(k)
    frame["hour"] = frame["observed_at"].dt.hour
    frame["weekday"] = frame["observed_at"].dt.weekday
    frame["station_code"] = frame["station_id"].map(STATION_CODE)
    return frame.dropna(subset=FEATURE_COLS + ["value"]).reset_index(drop=True)
