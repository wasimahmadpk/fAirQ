"""Snapshot freshness and last skill. Exit 1 if the serving path looks stale."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from fairq.db import connect
from fairq.ingest import POLLUTANTS, STATIONS
from fairq.ops import Check, report
from fairq.score import HORIZON_HOURS

MAX_MEASUREMENT_AGE_HOURS = float(os.environ.get("FAIRQ_MAX_MEASUREMENT_AGE_HOURS", "48"))
MAX_FORECAST_AGE_HOURS = float(os.environ.get("FAIRQ_MAX_FORECAST_AGE_HOURS", "36"))
EXPECTED_FORECASTS = len(STATIONS) * len(POLLUTANTS) * HORIZON_HOURS
POLL_NAMES = tuple(POLLUTANTS.values())


def age_hours(value) -> float | None:
    if value is None:
        return None
    when = pd_timestamp(value)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return (now - when).total_seconds() / 3600.0


def pd_timestamp(value):
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().replace(tzinfo=None)
    if getattr(value, "tzinfo", None):
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def run() -> int:
    db = connect()
    checks: list[Check] = []

    latest_air = db.query("SELECT max(observed_at) FROM measurements").first_row[0]
    air_age = age_hours(latest_air)
    checks.append(
        Check(
            "measurements_fresh",
            air_age is not None and air_age <= MAX_MEASUREMENT_AGE_HOURS,
            f"latest={latest_air} age_h={None if air_age is None else round(air_age, 1)}",
        )
    )

    forecast_n = int(db.query("SELECT count() FROM forecasts").first_row[0])
    latest_fc = db.query("SELECT min(forecast_at) FROM forecasts").first_row[0]
    fc_age = age_hours(latest_fc)
    checks.append(
        Check(
            "forecast_count",
            forecast_n >= EXPECTED_FORECASTS,
            f"n={forecast_n} expected={EXPECTED_FORECASTS}",
        )
    )
    checks.append(
        Check(
            "forecasts_fresh",
            fc_age is not None and fc_age <= MAX_FORECAST_AGE_HOURS,
            f"latest={latest_fc} age_h={None if fc_age is None else round(fc_age, 1)}",
        )
    )

    rows = db.query(
        "SELECT pollutant, metrics FROM model_versions ORDER BY trained_at DESC LIMIT 3"
    ).result_rows
    seen = {}
    for pollutant, metrics in rows:
        if pollutant not in seen:
            seen[pollutant] = json.loads(metrics)
    for pollutant in POLL_NAMES:
        metrics = seen.get(pollutant, {})
        skill = metrics.get("skill_24h")
        checks.append(
            Check(
                f"last_skill_24h_{pollutant}",
                skill is not None,
                f"skill_24h={skill}",
            )
        )

    ok = report("monitor", checks)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
