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
docker compose run --rm api python -m fairq.cdmi
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/status
curl "http://localhost:8000/forecast?station_id=mc174&pollutant=NO2"
curl http://localhost:8000/graph
open http://localhost:8000/graph/view
docker compose run --rm api python -m fairq.plot_graph
curl "http://localhost:8000/simulate?traffic_scale=0.8"
```

Ingest pulls **12 months** of hourly **NO₂, PM10, PM2.5** from three Berlin stations plus DWD weather. Train fits a 1h model and direct 24h / 96h / 168h models (last 14 days held out) and records skill vs persist. Validate checks row counts, last-window skill, and a **second 14-day holdout**. Score writes a 7-day hourly forecast. Monitor fails if measurements or forecasts are stale.

`GET /health` is process liveness. `GET /ready` needs ClickHouse plus measurements and forecasts. `GET /status` returns the latest model metrics and `pipeline_runs`.

`python -m fairq.cdmi` is a **light CDMI** (same idea as the DeepAR+knockoff paper, not that codebase): six hourly series, a small LSTM DeepAR-style forecaster (Gaussian NLL), Gaussian knockoffs, KS on residuals. Accepted edges go to `causal_edges`. `GET /graph` returns the weighted graph. `GET /simulate?traffic_scale=0.8` is `do(traffic := 0.8 × traffic)` on street NO₂. `traffic` is a calendar proxy, not vehicle counts.

The **expected** graph (physics / domain, not what CDMI currently draws):

```mermaid
flowchart TD
  traffic[Traffic]
  temp[Temp]
  humidity[Humidity]
  wind[Wind]
  street[Street NO2]
  forest[Forest NO2]

  traffic --> street
  wind --> street
  wind --> forest
  temp --> street
  temp --> forest
  humidity --> street
  humidity --> forest
  street --> forest
  humidity --- temp
```

Weather (wind, mixing / inversions, humidity) should move both NO₂ sites. Traffic should hit the street canyon first. Street NO₂ can then spill to the forest / background site. Temp and humidity are coupled meteorology, not an intervention. Arrows that should **not** appear: NO₂ → weather, NO₂ → traffic, traffic → temp. The learned graph is denser and often reversed because of shared daily and seasonal clocks; that is the gap to close next.

## Kubernetes (minikube)

A one-node cluster on this machine. Compose can stay up; use port **18000** for the cluster API so it does not clash with `localhost:8000`.

```bash
minikube start --driver=docker --memory=4096 --cpus=2
docker build -t fairq:latest .
minikube image load fairq:latest
kubectl apply -f k8s/
kubectl -n fairq get pods,cronjobs
kubectl -n fairq port-forward deploy/fairq-api 18000:8000
curl http://127.0.0.1:18000/health
```

`/ready` stays 503 until ingest+score have filled the **in-cluster** ClickHouse (empty on first boot; not the Compose database). CronJobs: ingest hourly, train+validate daily at 03:00 UTC, score hourly, monitor every 15 minutes.

```bash
minikube stop    # keep the cluster, free Docker RAM
minikube delete  # wipe it
```

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
src/fairq/cdmi.py      light causal graph + simulate
src/fairq/db.py        ClickHouse client
k8s/                   API Deployment + CronJobs
```
