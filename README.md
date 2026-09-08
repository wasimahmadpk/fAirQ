# fAirQ

Hourly **NO₂, PM10, and PM2.5** forecasts for Berlin, four days ahead.

Stack: Python, LightGBM, ClickHouse, Docker, Kubernetes jobs, FastAPI.

## What works now

```bash
docker compose up --build -d
docker compose run --rm api python -m fairq.ingest
docker compose run --rm api python -m fairq.train
docker compose run --rm api python -m fairq.score
curl "http://localhost:8000/forecast?station_id=mc174&pollutant=NO2"
```

Ingest pulls the last month of hourly **NO₂, PM10, PM2.5** from three Berlin stations plus DWD weather. Train fits one LightGBM model per pollutant (last 7 days held out). Score writes a 4-day hourly forecast.

## Next

5. Validation, monitoring, CI, Kubernetes CronJob YAML  
6. Later: causal `/simulate` (traffic effect)

## Layout

```
src/fairq/api.py       FastAPI
src/fairq/ingest.py    BLUME + weather download
src/fairq/features.py  lags + weather join
src/fairq/train.py     LightGBM
src/fairq/score.py     4-day forecast
src/fairq/db.py        ClickHouse client
```
