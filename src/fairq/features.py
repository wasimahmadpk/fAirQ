"""Join measurements + weather and add calendar, wind, and lag columns."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from fairq.db import connect

STATION_CODE = {"mc032": 0, "mc042": 1, "mc174": 2}
# Same-pollutant lags: recent hours (odd steps up to 12) plus yesterday.
LAGS = (1, 3, 5, 7, 9, 11, 12, 24)
# Easter Sundays used to derive Berlin moveable holidays.
_EASTER = {
    2024: date(2024, 3, 31),
    2025: date(2025, 4, 20),
    2026: date(2026, 4, 5),
    2027: date(2027, 3, 28),
}
CALENDAR_WEATHER = [
    "hour",
    "weekday",
    "month",
    "is_holiday",
    "temperature_c",
    "wind_speed_ms",
    "wind_sin",
    "wind_cos",
    "precipitation_mm",
    "relative_humidity",
]
FEATURE_COLS = [
    "station_code",
    *CALENDAR_WEATHER,
    *[f"lag_{k}" for k in LAGS],
]
DIRECT_HORIZONS = (24, 96, 168)


def direct_feature_cols(horizon: int) -> list[str]:
    return [
        "station_code",
        *[f"{col}_h{horizon}" for col in CALENDAR_WEATHER],
        *[f"lag_{k}" for k in LAGS],
    ]


DIRECT_24_COLS = direct_feature_cols(24)


def wind_components(direction_deg: float) -> tuple[float, float]:
    rad = math.radians(direction_deg)
    return math.sin(rad), math.cos(rad)


def berlin_holidays(year: int) -> set[date]:
    easter = _EASTER.get(year)
    if easter is None:
        return {
            date(year, 1, 1),
            date(year, 3, 8),
            date(year, 5, 1),
            date(year, 10, 3),
            date(year, 12, 25),
            date(year, 12, 26),
        }
    return {
        date(year, 1, 1),
        date(year, 3, 8),
        easter - timedelta(days=2),
        easter + timedelta(days=1),
        date(year, 5, 1),
        easter + timedelta(days=39),
        easter + timedelta(days=50),
        date(year, 10, 3),
        date(year, 12, 25),
        date(year, 12, 26),
    }


def is_berlin_holiday(day: date) -> int:
    return int(day in berlin_holidays(day.year))


def feature_vector(
    station_code: int,
    hour: int,
    weekday: int,
    month: int,
    is_holiday: int,
    temperature_c: float,
    wind_speed_ms: float,
    wind_sin: float,
    wind_cos: float,
    precipitation_mm: float,
    relative_humidity: float,
    past: list[float],
) -> list[float]:
    return [
        station_code,
        hour,
        weekday,
        month,
        is_holiday,
        temperature_c,
        wind_speed_ms,
        wind_sin,
        wind_cos,
        precipitation_mm,
        relative_humidity,
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
    if "relative_humidity" not in weather.columns:
        weather["relative_humidity"] = 70.0
    frame = air.merge(weather, on="observed_at", how="left")
    frame = frame.sort_values(["station_id", "pollutant", "observed_at"])
    frame["relative_humidity"] = (
        frame["relative_humidity"].ffill().bfill().fillna(70.0)
    )
    rad = np.deg2rad(frame["wind_direction_deg"].fillna(0.0).to_numpy())
    frame["wind_sin"] = np.sin(rad)
    frame["wind_cos"] = np.cos(rad)
    frame["hour"] = frame["observed_at"].dt.hour
    frame["weekday"] = frame["observed_at"].dt.weekday
    frame["month"] = frame["observed_at"].dt.month
    years = {ts.year for ts in frame["observed_at"]}
    holiday_days = set().union(*(berlin_holidays(year) for year in years))
    frame["is_holiday"] = frame["observed_at"].dt.date.isin(holiday_days).astype(int)
    frame["station_code"] = frame["station_id"].map(STATION_CODE)
    group = frame.groupby(["station_id", "pollutant"], sort=False)
    values = group["value"]
    for k in LAGS:
        frame[f"lag_{k}"] = values.shift(k)
    for horizon in DIRECT_HORIZONS:
        for col in CALENDAR_WEATHER:
            frame[f"{col}_h{horizon}"] = group[col].shift(-horizon)
        frame[f"target_{horizon}"] = values.shift(-horizon)
    return frame.dropna(subset=FEATURE_COLS + ["value"]).reset_index(drop=True)
