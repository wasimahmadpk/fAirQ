from fastapi import FastAPI, Query

from fairq.db import connect

app = FastAPI(title="fAirQ", version="0.2.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "fairq-api"}


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
