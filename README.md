# fAirQ

Hourly **NO₂, PM10, and PM2.5** forecasts for Berlin, four days ahead.

Stack: Python, XGBoost, ClickHouse, Docker, Kubernetes jobs, FastAPI.

## What works now

ClickHouse + FastAPI `/health`, plus an ingest job that fills the tables.

```bash
docker compose up --build -d
curl http://localhost:8000/health
docker compose run --rm api python -m fairq.ingest
```

Ingest pulls the last month of hourly **NO₂, PM10, PM2.5** from three Berlin stations (Frankfurter Allee, Neukölln, Grunewald) and matching DWD weather via Bright Sky.

## Next

3. Features + XGBoost train/score  
4. `/forecast` API  
5. Validation, monitoring, CI, Kubernetes CronJob YAML  
6. Later: causal `/simulate` (traffic effect)

## Layout

```
src/fairq/api.py      FastAPI
src/fairq/ingest.py   BLUME + weather download
src/fairq/db.py       ClickHouse client
sql/init.sql          schema
docker-compose.yml    ClickHouse + API
```
