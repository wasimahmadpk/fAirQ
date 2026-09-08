"""Train one LightGBM model per pollutant and store it on disk."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import lightgbm as lgb
import pandas as pd

from fairq.db import connect
from fairq.features import FEATURE_COLS, load_frame

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


def run() -> None:
    frame = load_frame()
    cutoff = frame["observed_at"].max() - pd.Timedelta(7, unit="D")
    os.makedirs(MODELS_DIR, exist_ok=True)
    db = connect()
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
        path = os.path.join(MODELS_DIR, f"{pollutant}.txt")
        model.save_model(path)
        db.insert(
            "model_versions",
            [[
                version,
                pollutant,
                datetime.now(timezone.utc).replace(tzinfo=None),
                json.dumps({"mae": round(mae, 3)}),
            ]],
            column_names=["model_version", "pollutant", "trained_at", "metrics"],
        )
        print(f"{pollutant} test MAE {mae:.2f}  n_train={len(train)} n_test={len(test)}")
    print(f"model_version {version}")


if __name__ == "__main__":
    run()
