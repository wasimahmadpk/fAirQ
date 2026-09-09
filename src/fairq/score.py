"""Write a 7-day hourly forecast using the saved LightGBM models."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import httpx
import lightgbm as lgb
import pandas as pd

from fairq.db import connect
from fairq.features import FEATURE_COLS, STATION_CODE, feature_vector
from fairq.ingest import BERLIN_LAT, BERLIN_LON, BRIGHTSKY, STATIONS, _utc

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
HORIZON_HOURS = 168
POLLUTANTS = ("NO2", "PM10", "PM2.5")


def latest_version(db) -> str:
    row = db.query("SELECT max(model_version) FROM model_versions").first_row
    if not row or not row[0]:
        raise RuntimeError("no trained model; run python -m fairq.train first")
    return row[0]


def load_history(db) -> pd.DataFrame:
    air = db.query_df(
        "SELECT station_id, observed_at, pollutant, value FROM measurements"
    )
    air["observed_at"] = pd.to_datetime(air["observed_at"]).dt.floor("h")
    return air.sort_values(["station_id", "pollutant", "observed_at"])


def fetch_future_weather(start: datetime, hours: int) -> pd.DataFrame:
    end = start + timedelta(hours=hours)
    response = httpx.get(
        BRIGHTSKY,
        params={
            "lat": BERLIN_LAT,
            "lon": BERLIN_LON,
            "date": start.date().isoformat(),
            "last_date": end.date().isoformat(),
        },
        timeout=60.0,
        headers={"User-Agent": "fairq-score"},
    )
    response.raise_for_status()
    rows = []
    for item in response.json().get("weather") or []:
        if item.get("temperature") is None:
            continue
        wind_kmh = item.get("wind_speed")
        rows.append(
            {
                "observed_at": _utc(item["timestamp"]).replace(minute=0, second=0, microsecond=0),
                "temperature_c": float(item["temperature"]),
                "wind_speed_ms": float(wind_kmh) / 3.6 if wind_kmh is not None else 0.0,
                "wind_direction_deg": float(item.get("wind_direction") or 0.0),
                "precipitation_mm": float(item.get("precipitation") or 0.0),
            }
        )
    weather = pd.DataFrame(rows).drop_duplicates("observed_at").set_index("observed_at")
    if weather.empty:
        raise RuntimeError("empty weather forecast")
    return weather


def run() -> None:
    db = connect()
    version = latest_version(db)
    history = load_history(db)
    last_obs = history["observed_at"].max().to_pydatetime()
    start = last_obs + timedelta(hours=1)
    weather = fetch_future_weather(start, HORIZON_HOURS)
    models = {
        pollutant: lgb.Booster(model_file=os.path.join(MODELS_DIR, f"{pollutant}.txt"))
        for pollutant in POLLUTANTS
    }

    rows = []
    for station_id in STATIONS:
        for pollutant in POLLUTANTS:
            past = history[
                (history["station_id"] == station_id) & (history["pollutant"] == pollutant)
            ]["value"].tolist()
            if len(past) < 24:
                raise RuntimeError(f"not enough history for {station_id} {pollutant}")
            last_weather = weather.iloc[0]
            for step in range(1, HORIZON_HOURS + 1):
                when = start + timedelta(hours=step - 1)
                if when in weather.index:
                    last_weather = weather.loc[when]
                features = pd.DataFrame(
                    [
                        feature_vector(
                            STATION_CODE[station_id],
                            when.hour,
                            when.weekday(),
                            float(last_weather["temperature_c"]),
                            float(last_weather["wind_speed_ms"]),
                            float(last_weather["wind_direction_deg"]),
                            float(last_weather["precipitation_mm"]),
                            past,
                        )
                    ],
                    columns=FEATURE_COLS,
                )
                pred = float(models[pollutant].predict(features)[0])
                past.append(pred)
                rows.append((station_id, when, step, pollutant, pred, version))

    db.command("TRUNCATE TABLE forecasts")
    db.insert(
        "forecasts",
        rows,
        column_names=[
            "station_id",
            "forecast_at",
            "horizon_hours",
            "pollutant",
            "value",
            "model_version",
        ],
    )
    print(f"forecasts {len(rows)}  model_version {version}")


if __name__ == "__main__":
    run()
