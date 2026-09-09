"""Download Berlin air measurements and DWD weather into ClickHouse."""

from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime, timezone
from time import sleep

import httpx

from fairq.db import connect

STATIONS = ("mc174", "mc042", "mc032")  # traffic, city background, suburb
POLLUTANTS = {
    "no2": "NO2",
    "pm10": "PM10",
    "pm2": "PM2.5",
}
N_MONTHS = 12
BERLIN_LAT = 52.52
BERLIN_LON = 13.405
BLUME = "https://luftdaten.berlin.de/api"
BRIGHTSKY = "https://api.brightsky.dev/weather"


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc).replace(tzinfo=None)


def month_windows(n_months: int) -> list[tuple[date, date]]:
    """Oldest month first. Hourly BLUME allows at most one month per call."""
    end = date.today()
    year, month = end.year, end.month
    windows: list[tuple[date, date]] = []
    for i in range(n_months):
        last_day = monthrange(year, month)[1]
        stop = end if i == 0 else date(year, month, last_day)
        windows.append((date(year, month, 1), stop))
        if month == 1:
            year, month = year - 1, 12
        else:
            month -= 1
    windows.reverse()
    return windows


def fetch_measurements(client: httpx.Client) -> list[tuple]:
    rows: dict[tuple, tuple] = {}
    for start, stop in month_windows(N_MONTHS):
        print(f"air {start} .. {stop}")
        for station_id in STATIONS:
            for core, pollutant in POLLUTANTS.items():
                response = client.get(
                    f"{BLUME}/stations/{station_id}/data",
                    params={
                        "core": core,
                        "period": "1h",
                        "timespan": "custom",
                        "start": start.strftime("%d.%m.%Y 00:00"),
                        "end": stop.strftime("%d.%m.%Y 23:00"),
                    },
                )
                response.raise_for_status()
                payload = response.json() or []
                for item in payload:
                    if item.get("value") is None:
                        continue
                    observed = _utc(item["datetime"])
                    rows[(station_id, observed, pollutant)] = (
                        station_id,
                        observed,
                        pollutant,
                        float(item["value"]),
                    )
                sleep(0.15)
    if not rows:
        raise RuntimeError("empty measurements")
    return list(rows.values())


def fetch_weather(client: httpx.Client, start: date, stop: date) -> list[tuple]:
    rows: dict[datetime, tuple] = {}
    for month_start, month_stop in month_windows(N_MONTHS):
        if month_stop < start or month_start > stop:
            continue
        print(f"weather {month_start} .. {month_stop}")
        response = client.get(
            BRIGHTSKY,
            params={
                "lat": BERLIN_LAT,
                "lon": BERLIN_LON,
                "date": month_start.isoformat(),
                "last_date": month_stop.isoformat(),
            },
        )
        response.raise_for_status()
        payload = response.json().get("weather") or []
        for item in payload:
            if item.get("temperature") is None:
                continue
            wind_kmh = item.get("wind_speed")
            observed = _utc(item["timestamp"])
            rows[observed] = (
                observed,
                float(item["temperature"]),
                float(wind_kmh) / 3.6 if wind_kmh is not None else 0.0,
                float(item.get("wind_direction") or 0.0),
                float(item.get("precipitation") or 0.0),
            )
        sleep(0.15)
    if not rows:
        raise RuntimeError("empty weather response")
    return list(rows.values())


def run() -> None:
    http = httpx.Client(timeout=60.0, headers={"User-Agent": "fairq-ingest"})
    db = connect()
    measurements = fetch_measurements(http)
    starts = [row[1] for row in measurements]
    weather = fetch_weather(http, min(starts).date(), max(starts).date())

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
    print(f"from {min(starts)} to {max(starts)}")
    counts = db.query(
        "SELECT station_id, pollutant, count() AS n "
        "FROM measurements GROUP BY station_id, pollutant ORDER BY station_id, pollutant"
    )
    for station_id, pollutant, n in counts.result_rows:
        print(f"  {station_id} {pollutant}: {n}")


if __name__ == "__main__":
    run()
