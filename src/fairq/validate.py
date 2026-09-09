"""Fail the job if data or model skill look wrong."""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

from fairq.db import connect
from fairq.features import direct_feature_cols, load_frame
from fairq.ingest import POLLUTANTS, STATIONS
from fairq.ops import Check, report, skill_score

MIN_ROWS = int(os.environ.get("FAIRQ_MIN_ROWS", "2000"))
MIN_SKILL_24H = float(os.environ.get("FAIRQ_MIN_SKILL_24H", "0.0"))
POLL_NAMES = tuple(POLLUTANTS.values())


def data_checks(db) -> list[Check]:
    checks: list[Check] = []
    counts = db.query(
        "SELECT station_id, pollutant, count() AS n, max(observed_at) AS latest "
        "FROM measurements GROUP BY station_id, pollutant"
    )
    seen = {(row[0], row[1]): (int(row[2]), row[3]) for row in counts.result_rows}
    for station_id in STATIONS:
        for pollutant in POLL_NAMES:
            n, latest = seen.get((station_id, pollutant), (0, None))
            checks.append(
                Check(
                    f"rows_{station_id}_{pollutant}",
                    n >= MIN_ROWS,
                    f"n={n} latest={latest}",
                )
            )
    weather_n = db.query("SELECT count() FROM weather").first_row[0]
    checks.append(Check("weather_rows", int(weather_n) > 0, f"n={weather_n}"))
    return checks


def model_checks(db) -> list[Check]:
    rows = db.query(
        "SELECT pollutant, metrics FROM model_versions ORDER BY trained_at DESC LIMIT 3"
    ).result_rows
    by_pollutant = {}
    for pollutant, metrics in rows:
        if pollutant not in by_pollutant:
            by_pollutant[pollutant] = json.loads(metrics)
    checks: list[Check] = []
    for pollutant in POLL_NAMES:
        metrics = by_pollutant.get(pollutant)
        if not metrics:
            checks.append(Check(f"model_{pollutant}", False, "missing"))
            continue
        skill = float(metrics.get("skill_24h", -1))
        checks.append(
            Check(
                f"skill_24h_{pollutant}",
                skill >= MIN_SKILL_24H,
                f"skill_24h={skill:+.3f}",
            )
        )
    return checks


def second_holdout_checks(frame: pd.DataFrame) -> list[Check]:
    """Train on data before the previous 14 days; score that window. Not the last fortnight."""
    from fairq.train import TEST_DAYS, fit_booster
    max_t = frame["observed_at"].max()
    hold_b_end = max_t - pd.Timedelta(days=int(TEST_DAYS))
    hold_b_start = hold_b_end - pd.Timedelta(days=int(TEST_DAYS))
    checks: list[Check] = []
    for pollutant, part in frame.groupby("pollutant"):
        cols = direct_feature_cols(24)
        ready = part.dropna(subset=cols + ["target_24"])
        train = ready[ready["observed_at"] < hold_b_start - pd.Timedelta("24h")]
        test = ready[
            (ready["observed_at"] >= hold_b_start) & (ready["observed_at"] < hold_b_end)
        ]
        if len(train) < 500 or len(test) < 100:
            checks.append(
                Check(
                    f"holdout2_skill_24h_{pollutant}",
                    False,
                    f"too few rows train={len(train)} test={len(test)}",
                )
            )
            continue
        model = fit_booster(train, cols, "target_24")
        preds = model.predict(test[cols])
        g_mae = float((test["target_24"] - preds).abs().mean())
        p_mae = float((test["target_24"] - test["value"]).abs().mean())
        skill = skill_score(g_mae, p_mae)
        checks.append(
            Check(
                f"holdout2_skill_24h_{pollutant}",
                skill >= MIN_SKILL_24H,
                f"skill={skill:+.3f} gbm={g_mae:.2f} persist={p_mae:.2f} n_test={len(test)}",
            )
        )
    return checks


def run() -> int:
    db = connect()
    checks = data_checks(db)
    checks.extend(model_checks(db))
    if all(check.ok for check in checks if check.name.startswith("rows_")):
        checks.extend(second_holdout_checks(load_frame()))
    ok = report("validate", checks)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
