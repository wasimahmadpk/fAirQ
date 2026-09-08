"""Join measurements + weather and add a few simple columns."""

from __future__ import annotations

import pandas as pd

from fairq.db import connect

STATION_CODE = {"mc032": 0, "mc042": 1, "mc174": 2}
FEATURE_COLS = [
    "station_code",
    "hour",
    "weekday",
    "temperature_c",
    "wind_speed_ms",
    "wind_direction_deg",
    "precipitation_mm",
    "lag_1",
    "lag_24",
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
    frame["lag_1"] = group.shift(1)
    frame["lag_24"] = group.shift(24)
    frame["hour"] = frame["observed_at"].dt.hour
    frame["weekday"] = frame["observed_at"].dt.weekday
    frame["station_code"] = frame["station_id"].map(STATION_CODE)
    return frame.dropna(subset=FEATURE_COLS + ["value"]).reset_index(drop=True)
