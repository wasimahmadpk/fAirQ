"""Train one LightGBM model per pollutant and store it on disk."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import lightgbm as lgb
import pandas as pd

from fairq.db import connect
from fairq.features import FEATURE_COLS, feature_vector, load_frame

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
TEST_DAYS = 14
HORIZONS = (1, 24, 96, 168)


def walk_mae(model, hist: pd.DataFrame, future: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """One origin at the start of future. Persist is frozen at the last real value."""
    past = hist["value"].tolist()
    persist_val = past[-1]
    gbm_err: dict[int, float] = {}
    persist_err: dict[int, float] = {}
    for step, row in enumerate(future.itertuples(index=False), start=1):
        features = pd.DataFrame(
            [
                feature_vector(
                    row.station_code,
                    row.hour,
                    row.weekday,
                    row.temperature_c,
                    row.wind_speed_ms,
                    row.wind_direction_deg,
                    row.precipitation_mm,
                    past,
                )
            ],
            columns=FEATURE_COLS,
        )
        pred = float(model.predict(features)[0])
        past.append(pred)
        if step in HORIZONS:
            gbm_err[step] = abs(row.value - pred)
            persist_err[step] = abs(row.value - persist_val)
    return {h: (gbm_err[h], persist_err[h]) for h in HORIZONS if h in gbm_err}


def run() -> None:
    frame = load_frame()
    cutoff = frame["observed_at"].max() - pd.Timedelta(TEST_DAYS, unit="D")
    os.makedirs(MODELS_DIR, exist_ok=True)
    db = connect()
    db.command("TRUNCATE TABLE model_versions")
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")

    for pollutant, part in frame.groupby("pollutant"):
        train = part[part["observed_at"] < cutoff]
        test = part[part["observed_at"] >= cutoff]
        model = lgb.train(
            {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1},
            lgb.Dataset(train[FEATURE_COLS], label=train["value"]),
            num_boost_round=200,
        )
        preds = model.predict(test[FEATURE_COLS])
        mae = float((test["value"] - preds).abs().mean())
        persist_mae = float((test["value"] - test["lag_1"]).abs().mean())
        path = os.path.join(MODELS_DIR, f"{pollutant}.txt")
        model.save_model(path)

        print(
            f"{pollutant}  1h-with-real-lag  LightGBM {mae:.2f}  persist {persist_mae:.2f}  "
            f"n_train={len(train)} n_test={len(test)}"
        )
        horizon_gbm = {h: [] for h in HORIZONS}
        horizon_persist = {h: [] for h in HORIZONS}
        for _, station_part in part.groupby("station_id"):
            hist = station_part[station_part["observed_at"] < cutoff]
            future = station_part[station_part["observed_at"] >= cutoff]
            if len(hist) < 24 or len(future) < max(HORIZONS):
                continue
            scores = walk_mae(model, hist, future)
            for h, (g, p) in scores.items():
                horizon_gbm[h].append(g)
                horizon_persist[h].append(p)
        for h in HORIZONS:
            if not horizon_gbm[h]:
                continue
            g = sum(horizon_gbm[h]) / len(horizon_gbm[h])
            p = sum(horizon_persist[h]) / len(horizon_persist[h])
            print(f"{pollutant}  {h:>3}h-ahead        LightGBM {g:.2f}  persist {p:.2f}")

        db.insert(
            "model_versions",
            [[
                version,
                pollutant,
                datetime.now(timezone.utc).replace(tzinfo=None),
                json.dumps({"mae_1h": round(mae, 3), "persist_1h": round(persist_mae, 3)}),
            ]],
            column_names=["model_version", "pollutant", "trained_at", "metrics"],
        )
    print(f"model_version {version}")


if __name__ == "__main__":
    run()
