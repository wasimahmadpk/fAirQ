# fAirQ

Mini clone of INWT’s **FAirQ** Berlin air-quality forecast — for an MLOps interview, not a product.

This repo is standalone. It has nothing to do with TraceDefect / vcare.

FAirQ (INWT’s name for the model; they do not publish a long form) forecasts hourly **NO₂, PM10, PM2.5** for Berlin, four days ahead. Their stack: Python, XGBoost, ClickHouse, Docker, Kubernetes jobs, FastAPI.

## What works now (step 1)

Local stack only: ClickHouse + a FastAPI `/health` endpoint.

```bash
docker compose up --build
curl http://localhost:8000/health
```

Tables created on first boot: `measurements`, `weather`, `forecasts`, `model_versions`.

Not on Vercel. The model and database stay in Docker. A public host (Railway / Render / Fly) comes after the pipeline runs locally.

## Next

2. Ingest BLUME stations + Bright Sky (DWD) weather  
3. Features + XGBoost train/score  
4. `/forecast` API  
5. Validation, monitoring, CI, Kubernetes CronJob YAML  
6. Later: causal `/simulate` (traffic effect). Not in this step.

## Layout

```
src/fairq/api.py    FastAPI
sql/init.sql        ClickHouse schema
docker-compose.yml  ClickHouse + API
```
