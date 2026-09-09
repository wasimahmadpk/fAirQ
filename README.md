# fAirQ

Hourly **NO₂, PM10, and PM2.5** forecasts for Berlin, seven days ahead.

Stack: Python, LightGBM, ClickHouse, Docker, Kubernetes jobs, FastAPI.

## Local pipeline

```bash
docker compose up --build -d
docker compose run --rm api python -m fairq.ingest
docker compose run --rm api python -m fairq.train
docker compose run --rm api python -m fairq.validate
docker compose run --rm api python -m fairq.score
docker compose run --rm api python -m fairq.monitor
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/status
curl "http://localhost:8000/forecast?station_id=mc174&pollutant=NO2"
```

Ingest pulls **12 months** of hourly **NO₂, PM10, PM2.5** from three Berlin stations plus DWD weather. Train fits a 1h model and direct 24h / 96h / 168h models (last 14 days held out) and records skill vs persist. Validate checks row counts, last-window skill, and a **second 14-day holdout**. Score writes a 7-day hourly forecast. Monitor fails if measurements or forecasts are stale.

`GET /health` is process liveness. `GET /ready` needs ClickHouse plus measurements and forecasts. `GET /status` returns the latest model metrics and `pipeline_runs`.

## Kubernetes

```bash
kubectl apply -f k8s/
```

CronJobs: ingest hourly, train+validate daily at 03:00 UTC, score hourly, monitor every 15 minutes. Set `CLICKHOUSE_HOST` in `k8s/config.yaml` to your ClickHouse service. Build and load `fairq:latest` into the cluster before the jobs run.

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
PYTHONPATH=src pytest -q
```

## Layout

```
src/fairq/api.py       FastAPI
src/fairq/ingest.py    BLUME + weather download
src/fairq/features.py  lags + weather join
src/fairq/train.py     LightGBM
src/fairq/validate.py  data contracts + second holdout
src/fairq/score.py     7-day forecast
src/fairq/monitor.py   freshness checks
src/fairq/db.py        ClickHouse client
k8s/                   API Deployment + CronJobs
```
