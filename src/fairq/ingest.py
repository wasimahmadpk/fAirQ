"""Download Berlin air measurements and DWD weather into ClickHouse."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from fairq.db import connect

STATIONS = ("mc174", "mc042", "mc032")  # traffic, city background, suburb
POLLUTANTS = {
    "no2": "NO2",
    "pm10": "PM10",
    "pm2": "PM2.5",
}
BERLIN_LAT = 52.52
BERLIN_LON = 13.405
BLUME = "https://luftdaten.berlin.de/api"
BRIGHTSKY = "https://api.brightsky.dev/weather"


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc).replace(tzinfo=None)


def fetch_measurements(client: httpx.Client) -> list[tuple]:
    rows: list[tuple] = []
    for station_id in STATIONS:
        for core, pollutant in POLLUTANTS.items():
            url = f"{BLUME}/stations/{station_id}/data"
            response = client.get(
                url,
                params={"core": core, "period": "1h", "timespan": "lastmonth"},
            )
            response.raise_for_status()
            payload = response.json()
            if not payload:
                raise RuntimeError(f"empty measurements for {station_id} {pollutant}")
            for item in payload:
                if item.get("value") is None:
                    continue
                rows.append(
                    (
                        station_id,
                        _utc(item["datetime"]),
                        pollutant,
                        float(item["value"]),
                    )
                )
    return rows


def fetch_weather(client: httpx.Client, start: datetime, end: datetime) -> list[tuple]:
    response = client.get(
        BRIGHTSKY,
        params={
            "lat": BERLIN_LAT,
            "lon": BERLIN_LON,
            "date": start.date().isoformat(),
            "last_date": end.date().isoformat(),
        },
    )
    response.raise_for_status()
    payload = response.json().get("weather") or []
    if not payload:
        raise RuntimeError("empty weather response")
    rows: list[tuple] = []
    for item in payload:
        if item.get("temperature") is None:
            continue
        wind_kmh = item.get("wind_speed")
        rows.append(
            (
                _utc(item["timestamp"]),
                float(item["temperature"]),
                float(wind_kmh) / 3.6 if wind_kmh is not None else 0.0,
                float(item.get("wind_direction") or 0.0),
                float(item.get("precipitation") or 0.0),
            )
        )
    return rows


def run() -> None:
    http = httpx.Client(timeout=60.0, headers={"User-Agent": "fairq-ingest"})
    db = connect()
    measurements = fetch_measurements(http)
    starts = [row[1] for row in measurements]
    weather = fetch_weather(http, min(starts), max(starts))

    db.command("TRUNCATE TABLE fairq.measurements")
    db.command("TRUNCATE TABLE fairq.weather")
    db.insert(
        "measurements",
        measurements,
        column_names=["station_id", "observed_at", "pollutant", "value"],
    )
    db.insert(
        "weather",
        weather,
        column_names=[
            "observed_at",
            "temperature_c",
            "wind_speed_ms",
            "wind_direction_deg",
            "precipitation_mm",
        ],
    )
    print(f"measurements {len(measurements)}")
    print(f"weather {len(weather)}")
    counts = db.query(
        "SELECT station_id, pollutant, count() AS n "
        "FROM measurements GROUP BY station_id, pollutant ORDER BY station_id, pollutant"
    )
    for station_id, pollutant, n in counts.result_rows:
        print(f"  {station_id} {pollutant}: {n}")


if __name__ == "__main__":
    run()
