"""Train one LightGBM model per pollutant and store it on disk."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import lightgbm as lgb
import pandas as pd

from fairq.db import connect
from fairq.features import (
    DIRECT_HORIZONS,
    FEATURE_COLS,
    direct_feature_cols,
    feature_vector,
    load_frame,
)
from fairq.ops import skill_score

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
TEST_DAYS = 14
HORIZONS = (1, 24, 96, 168)
LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbosity": -1,
}


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
                    row.month,
                    row.is_holiday,
                    row.temperature_c,
                    row.wind_speed_ms,
                    row.wind_sin,
                    row.wind_cos,
                    row.precipitation_mm,
                    row.relative_humidity,
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


def fit_booster(frame: pd.DataFrame, columns: list[str], label: str) -> lgb.Booster:
    return lgb.train(
        LGB_PARAMS,
        lgb.Dataset(frame[columns], label=frame[label]),
        num_boost_round=200,
    )


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
        model = fit_booster(train, FEATURE_COLS, "value")
        preds = model.predict(test[FEATURE_COLS])
        mae = float((test["value"] - preds).abs().mean())
        persist_mae = float((test["value"] - test["lag_1"]).abs().mean())
        path = os.path.join(MODELS_DIR, f"{pollutant}.txt")
        model.save_model(path)

        skill_1h = skill_score(mae, persist_mae)
        stored = {
            "mae_1h": round(mae, 3),
            "persist_1h": round(persist_mae, 3),
            "skill_1h": round(skill_1h, 3),
        }
        print(
            f"{pollutant}  1h-with-real-lag  LightGBM {mae:.2f}  persist {persist_mae:.2f}  "
            f"skill {skill_1h:+.2f}  n_train={len(train)} n_test={len(test)}"
        )

        for horizon in DIRECT_HORIZONS:
            cols = direct_feature_cols(horizon)
            target = f"target_{horizon}"
            ready = part.dropna(subset=cols + [target])
            train_h = ready[ready["observed_at"] < cutoff - pd.Timedelta(hours=int(horizon))]
            test_h = ready[ready["observed_at"] >= cutoff]
            if not len(train_h) or not len(test_h):
                continue
            model_h = fit_booster(train_h, cols, target)
            preds_h = model_h.predict(test_h[cols])
            g_mae = float((test_h[target] - preds_h).abs().mean())
            p_mae = float((test_h[target] - test_h["value"]).abs().mean())
            skill = skill_score(g_mae, p_mae)
            model_h.save_model(os.path.join(MODELS_DIR, f"{pollutant}_h{horizon}.txt"))
            stored[f"gbm_{horizon}h"] = round(g_mae, 3)
            stored[f"persist_{horizon}h"] = round(p_mae, 3)
            stored[f"skill_{horizon}h"] = round(skill, 3)
            print(
                f"{pollutant}  {horizon:>3}h-direct        LightGBM {g_mae:.2f}  persist {p_mae:.2f}  "
                f"skill {skill:+.2f}  n_test={len(test_h)}"
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
            skill = skill_score(g, p)
            stored[f"gbm_{h}h_walk"] = round(g, 3)
            stored[f"persist_{h}h_walk"] = round(p, 3)
            stored[f"skill_{h}h_walk"] = round(skill, 3)
            print(
                f"{pollutant}  {h:>3}h-recursive    LightGBM {g:.2f}  persist {p:.2f}  "
                f"skill {skill:+.2f}"
            )

        db.insert(
            "model_versions",
            [[
                version,
                pollutant,
                datetime.now(timezone.utc).replace(tzinfo=None),
                json.dumps(stored),
            ]],
            column_names=["model_version", "pollutant", "trained_at", "metrics"],
        )
    print(f"model_version {version}")


if __name__ == "__main__":
    run()
