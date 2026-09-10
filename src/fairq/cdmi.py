"""Lightweight CDMI: Tiny-DeepAR + Gaussian knockoffs, then a traffic do-query.

Same invariance idea as https://github.com/wasimahmadpk/cdmi, without GluonTS
or DeepKnockoffs. The forecaster is a small LSTM with Gaussian NLL (DeepAR-like).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fairq.db import connect
from fairq.features import is_berlin_holiday

NODES = ("no2_street", "no2_bg", "temp", "wind", "humidity", "traffic")
STREET = "mc174"
BACKGROUND = "mc032"
TEST_DAYS = 14
SEQ_LEN = 24
HIDDEN = 16
EPOCHS = 8
BATCH = 256
KS_P_MAX = float(os.environ.get("FAIRQ_CDMI_P_MAX", "0.10"))
KS_STAT_MIN = float(os.environ.get("FAIRQ_CDMI_KS_MIN", "0.03"))
REL_MAE_MIN = float(os.environ.get("FAIRQ_CDMI_REL_MAE", "0.015"))
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
STREET_MODEL = os.path.join(MODELS_DIR, "cdmi_no2_street.pt")


def accept_edge(ks_stat: float, p_value: float, rel_mae: float) -> bool:
    """Keep only links that also hurt the forecast. KS or p-value is not enough alone."""
    if rel_mae < REL_MAE_MIN:
        return False
    return p_value < KS_P_MAX or ks_stat >= KS_STAT_MIN


def traffic_proxy(hour: int, weekday: int, holiday: int) -> float:
    """0–1 stand-in for cars. Not counts — calendar rush vs quiet days."""
    if holiday:
        return 0.08
    if weekday == 6:
        return 0.10
    if weekday == 5:
        return 0.25
    if hour in (7, 8, 9, 16, 17, 18):
        return 1.0
    if 10 <= hour <= 15:
        return 0.55
    if 6 <= hour <= 21:
        return 0.35
    return 0.12


def ks_2samp(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic and a cheap asymptotic p-value. No scipy."""
    a = np.sort(np.asarray(left, dtype=float))
    b = np.sort(np.asarray(right, dtype=float))
    n, m = len(a), len(b)
    if n < 8 or m < 8:
        return 0.0, 1.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / n
    cdf_b = np.searchsorted(b, grid, side="right") / m
    stat = float(np.max(np.abs(cdf_a - cdf_b)))
    z = stat * np.sqrt(n * m / (n + m))
    p_value = float(min(1.0, 2.0 * np.exp(-2.0 * z * z)))
    return stat, p_value


def gaussian_knockoffs(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Equicorrelated second-order (Gaussian) knockoffs. In-distribution, null."""
    n_rows, n_cols = values.shape
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standard = (values - center) / scale
    if n_cols == 1:
        noise = rng.standard_normal((n_rows, 1))
        return (standard + noise) * scale + center
    sigma = np.corrcoef(standard, rowvar=False)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = (sigma + sigma.T) / 2.0 + 1e-3 * np.eye(n_cols)
    eig_min = max(float(np.linalg.eigvalsh(sigma).min()), 1e-6)
    strength = min(1.0, 1.7 * eig_min)
    shrink = strength * np.eye(n_cols)
    precision = np.linalg.pinv(sigma)
    gram = 2.0 * shrink - shrink @ precision @ shrink
    gram = (gram + gram.T) / 2.0
    eigenvalues, vectors = np.linalg.eigh(gram)
    eigenvalues = np.clip(eigenvalues, 1e-8, None)
    sqrt_gram = vectors @ np.diag(np.sqrt(eigenvalues)) @ vectors.T
    mapped = standard @ (np.eye(n_cols) - precision @ shrink).T
    knock = mapped + rng.standard_normal((n_rows, n_cols)) @ sqrt_gram
    return knock * scale + center


def load_system() -> pd.DataFrame:
    db = connect()
    air = db.query_df(
        "SELECT station_id, observed_at, value FROM measurements WHERE pollutant = 'NO2'"
    )
    weather = db.query_df("SELECT * FROM weather")
    if air.empty or weather.empty:
        raise RuntimeError("need measurements and weather; run ingest first")
    air["observed_at"] = pd.to_datetime(air["observed_at"]).dt.floor("h")
    weather["observed_at"] = pd.to_datetime(weather["observed_at"]).dt.floor("h")
    street = air.loc[air["station_id"] == STREET, ["observed_at", "value"]].rename(
        columns={"value": "no2_street"}
    )
    background = air.loc[air["station_id"] == BACKGROUND, ["observed_at", "value"]].rename(
        columns={"value": "no2_bg"}
    )
    frame = street.merge(background, on="observed_at", how="inner")
    frame = frame.merge(weather, on="observed_at", how="inner")
    if "relative_humidity" not in frame.columns:
        frame["relative_humidity"] = 70.0
    frame["temp"] = frame["temperature_c"]
    frame["wind"] = frame["wind_speed_ms"]
    frame["humidity"] = frame["relative_humidity"].ffill().bfill().fillna(70.0)
    frame["hour"] = frame["observed_at"].dt.hour
    frame["weekday"] = frame["observed_at"].dt.weekday
    frame["is_holiday"] = [is_berlin_holiday(ts.date()) for ts in frame["observed_at"]]
    frame["traffic"] = [
        traffic_proxy(int(hour), int(weekday), int(holiday))
        for hour, weekday, holiday in zip(frame["hour"], frame["weekday"], frame["is_holiday"])
    ]
    frame = frame.sort_values("observed_at").drop_duplicates("observed_at")
    return frame[["observed_at", *NODES]].dropna().reset_index(drop=True)


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def _make_model(n_in: int, hidden: int = HIDDEN):
    torch, nn = _torch()

    class TinyDeepAR(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, num_layers=1, batch_first=True)
            self.mu = nn.Linear(hidden, 1)
            self.log_sigma = nn.Linear(hidden, 1)

        def forward(self, seq):
            encoded, _ = self.lstm(seq)
            last = encoded[:, -1]
            return self.mu(last).squeeze(-1), self.log_sigma(last).squeeze(-1).clamp(-4.0, 2.0)

    return TinyDeepAR()


def _windows(values: np.ndarray, seq: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    rows = len(values)
    if rows <= seq:
        return np.empty((0, seq, values.shape[1])), np.empty((0, values.shape[1]))
    series = np.stack([values[i : i + seq] for i in range(rows - seq)])
    targets = values[seq:]
    return series, targets


def _standardize(train: np.ndarray, whole: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (whole - center) / scale, center, scale


def _fit_lstm(windows: np.ndarray, target: np.ndarray, epochs: int = EPOCHS):
    torch, _ = _torch()
    model = _make_model(windows.shape[-1])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_all = torch.tensor(windows, dtype=torch.float32)
    y_all = torch.tensor(target, dtype=torch.float32)
    n_rows = len(x_all)
    for _ in range(epochs):
        order = torch.randperm(n_rows)
        for start in range(0, n_rows, BATCH):
            idx = order[start : start + BATCH]
            mu, log_sigma = model(x_all[idx])
            var = torch.exp(log_sigma * 2) + 1e-4
            loss = 0.5 * (torch.log(var) + (y_all[idx] - mu) ** 2 / var).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _predict(model, windows: np.ndarray) -> np.ndarray:
    torch, _ = _torch()
    if len(windows) == 0:
        return np.array([])
    model.eval()
    with torch.no_grad():
        mu, _ = model(torch.tensor(windows, dtype=torch.float32))
    return mu.numpy()


def discover(system: pd.DataFrame) -> dict:
    rng = np.random.default_rng(7)
    cutoff = system["observed_at"].max() - pd.Timedelta(days=int(TEST_DAYS))
    raw = system[list(NODES)].to_numpy(dtype=float)
    times = system["observed_at"]
    train_mask = times < cutoff
    scaled, center, scale = _standardize(raw[train_mask.to_numpy()], raw)
    series, targets = _windows(scaled)
    time_of_y = times.iloc[SEQ_LEN:].reset_index(drop=True)
    train_y = time_of_y < cutoff
    test_y = time_of_y >= cutoff
    if int(train_y.sum()) < 400 or int(test_y.sum()) < 80:
        raise RuntimeError("not enough hourly rows for CDMI")
    knock_scaled = (gaussian_knockoffs(raw, rng) - center) / scale
    edges: list[dict] = []

    for effect_i, effect in enumerate(NODES):
        model = _fit_lstm(series[train_y.to_numpy()], targets[train_y.to_numpy(), effect_i])
        x_test = series[test_y.to_numpy()]
        y_test = targets[test_y.to_numpy(), effect_i]
        pred_obs = _predict(model, x_test)
        residual_obs = y_test - pred_obs
        mae_obs = float(np.mean(np.abs(residual_obs)))
        if effect == "no2_street":
            os.makedirs(MODELS_DIR, exist_ok=True)
            _torch()[0].save(
                {
                    "state": model.state_dict(),
                    "center": center,
                    "scale": scale,
                    "nodes": list(NODES),
                    "seq_len": SEQ_LEN,
                    "hidden": HIDDEN,
                },
                STREET_MODEL,
            )
        for cause_i, cause in enumerate(NODES):
            if cause == effect:
                continue
            mixed = scaled.copy()
            mixed[:, cause_i] = knock_scaled[:, cause_i]
            series_k, _ = _windows(mixed)
            x_do = series_k[test_y.to_numpy()]
            residual_k = y_test - _predict(model, x_do)
            stat, p_value = ks_2samp(residual_obs, residual_k)
            mae_k = float(np.mean(np.abs(residual_k)))
            rel_mae = (mae_k - mae_obs) / mae_obs if mae_obs else 0.0
            accepted = int(accept_edge(stat, p_value, rel_mae))
            edges.append(
                {
                    "cause": cause,
                    "effect": effect,
                    "ks_stat": round(stat, 4),
                    "p_value": round(p_value, 4),
                    "rel_mae": round(rel_mae, 4),
                    "accepted": accepted,
                }
            )
            mark = "edge" if accepted else "none"
            print(
                f"cdmi  {mark:4}  {cause:>12} -> {effect:<12}  "
                f"KS={stat:.3f}  p={p_value:.3f}  dMAE={rel_mae:+.3f}"
            )

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    return {"model_version": version, "nodes": list(NODES), "edges": edges}


def persist(result: dict) -> None:
    db = connect()
    db.command("TRUNCATE TABLE fairq.causal_edges")
    db.command("TRUNCATE TABLE fairq.causal_graphs")
    trained = datetime.now(timezone.utc).replace(tzinfo=None)
    if result["edges"]:
        db.insert(
            "causal_edges",
            [
                [
                    result["model_version"],
                    trained,
                    edge["cause"],
                    edge["effect"],
                    edge["ks_stat"],
                    edge["p_value"],
                    edge["rel_mae"],
                    edge["accepted"],
                ]
                for edge in result["edges"]
            ],
            column_names=[
                "model_version",
                "trained_at",
                "cause",
                "effect",
                "ks_stat",
                "p_value",
                "rel_mae",
                "accepted",
            ],
        )
    note = (
        "Light CDMI: Tiny-DeepAR (LSTM + Gaussian NLL) + Gaussian knockoffs. "
        f"Edge if test MAE rises >={REL_MAE_MIN:.1%} and (KS p<{KS_P_MAX} or KS>={KS_STAT_MIN}). "
        "traffic is a calendar proxy, not vehicle counts. "
        "Assumes sufficiency and stationarity; hidden regional sources possible."
    )
    db.insert(
        "causal_graphs",
        [[result["model_version"], trained, json.dumps(result["nodes"]), json.dumps(result["edges"]), note]],
        column_names=["model_version", "trained_at", "nodes", "edges", "notes"],
    )
    print(f"cdmi  model_version {result['model_version']}  edges={sum(e['accepted'] for e in result['edges'])}")


def latest_graph(db) -> dict | None:
    row = db.query(
        "SELECT model_version, nodes, edges, notes FROM causal_graphs ORDER BY trained_at DESC LIMIT 1"
    ).first_row
    if not row:
        return None
    version, nodes, edges, notes = row
    return {
        "model_version": version,
        "nodes": json.loads(nodes),
        "edges": json.loads(edges),
        "notes": notes,
    }


def simulate(traffic_scale: float, hours: int = 24) -> dict:
    if traffic_scale <= 0 or traffic_scale > 2:
        raise ValueError("traffic_scale must be in (0, 2]")
    db = connect()
    graph = latest_graph(db)
    if graph is None:
        raise RuntimeError("no causal graph; run python -m fairq.cdmi first")
    if not os.path.exists(STREET_MODEL):
        raise RuntimeError("missing cdmi_no2_street.pt; run python -m fairq.cdmi first")
    torch, _ = _torch()
    bundle = torch.load(STREET_MODEL, map_location="cpu")
    model = _make_model(len(NODES), hidden=int(bundle["hidden"]))
    model.load_state_dict(bundle["state"])
    system = load_system()
    if len(system) < SEQ_LEN + hours:
        raise RuntimeError("not enough hourly rows for simulate")
    raw = system[list(NODES)].to_numpy(dtype=float)
    scaled = (raw - bundle["center"]) / bundle["scale"]
    intervened = system.copy()
    intervened["traffic"] = intervened["traffic"] * traffic_scale
    mixed = (intervened[list(NODES)].to_numpy(dtype=float) - bundle["center"]) / bundle["scale"]
    series_obs, _ = _windows(scaled)
    series_do, _ = _windows(mixed)
    times = system["observed_at"].iloc[SEQ_LEN:]
    pred_obs = _predict(model, series_obs[-hours:])
    pred_do = _predict(model, series_do[-hours:])
    street_i = NODES.index("no2_street")
    obs_nat = pred_obs * bundle["scale"][street_i] + bundle["center"][street_i]
    do_nat = pred_do * bundle["scale"][street_i] + bundle["center"][street_i]
    stamp = times.iloc[-hours:]
    parents = [
        edge
        for edge in graph["edges"]
        if edge["effect"] == "no2_street" and edge["accepted"]
    ]
    traffic_is_parent = any(edge["cause"] == "traffic" for edge in parents)
    rows = []
    for when, obs, do_val in zip(stamp, obs_nat, do_nat):
        rows.append(
            {
                "observed_at": str(when),
                "no2_observational": round(float(obs), 3),
                "no2_do_traffic": round(float(do_val), 3),
                "delta": round(float(do_val - obs), 3),
            }
        )
    mean_delta = float(np.mean([row["delta"] for row in rows])) if rows else 0.0
    return {
        "question": f"do(traffic := {traffic_scale} * traffic) on street NO2 (mc174)",
        "traffic_scale": traffic_scale,
        "traffic_is_parent": traffic_is_parent,
        "parents_of_no2_street": parents,
        "mean_delta": round(mean_delta, 3),
        "model_version": graph["model_version"],
        "notes": graph["notes"],
        "rows": rows,
    }


def run() -> None:
    persist(discover(load_system()))


if __name__ == "__main__":
    run()
