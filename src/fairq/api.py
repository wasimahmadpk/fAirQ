from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from fairq.db import connect

app = FastAPI(title="fAirQ", version="0.3.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "fairq-api"}


@app.get("/ready")
def ready():
    try:
        db = connect()
        air = db.query("SELECT count(), max(observed_at) FROM measurements").first_row
        fc = db.query("SELECT count(), max(forecast_at) FROM forecasts").first_row
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": str(exc)},
        )
    air_n, air_latest = air
    fc_n, fc_latest = fc
    if not air_n or not fc_n:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "measurements": int(air_n or 0),
                "forecasts": int(fc_n or 0),
            },
        )
    return {
        "status": "ready",
        "measurements": int(air_n),
        "forecasts": int(fc_n),
        "latest_measurement": str(air_latest),
        "latest_forecast": str(fc_latest),
    }


@app.get("/status")
def status() -> dict:
    db = connect()
    models = db.query(
        "SELECT pollutant, model_version, metrics FROM model_versions "
        "ORDER BY trained_at DESC LIMIT 3"
    ).result_rows
    runs = db.query(
        "SELECT job, max(ran_at) AS last_run, argMax(ok, ran_at) AS last_ok "
        "FROM pipeline_runs GROUP BY job ORDER BY job"
    ).result_rows
    return {
        "models": [
            {
                "pollutant": pollutant,
                "model_version": version,
                "metrics": metrics,
            }
            for pollutant, version, metrics in models
        ],
        "pipeline": [
            {"job": job, "last_run": str(last_run), "ok": bool(last_ok)}
            for job, last_run, last_ok in runs
        ],
    }


@app.get("/forecast")
def forecast(
    station_id: str = Query(default="mc174"),
    pollutant: str = Query(default="NO2"),
) -> dict:
    db = connect()
    result = db.query(
        "SELECT station_id, forecast_at, horizon_hours, pollutant, value, model_version "
        "FROM forecasts "
        "WHERE station_id = {sid:String} AND pollutant = {pol:String} "
        "ORDER BY horizon_hours",
        parameters={"sid": station_id, "pol": pollutant},
    )
    rows = [
        {
            "station_id": station_id,
            "forecast_at": str(forecast_at),
            "horizon_hours": horizon_hours,
            "pollutant": pollutant,
            "value": value,
            "model_version": model_version,
        }
        for station_id, forecast_at, horizon_hours, pollutant, value, model_version in result.result_rows
    ]
    return {"count": len(rows), "rows": rows}
