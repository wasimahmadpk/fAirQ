CREATE DATABASE IF NOT EXISTS fairq;

CREATE TABLE IF NOT EXISTS fairq.measurements
(
    station_id String,
    observed_at DateTime,
    pollutant String,
    value Float64
)
ENGINE = MergeTree
ORDER BY (station_id, pollutant, observed_at);

CREATE TABLE IF NOT EXISTS fairq.weather
(
    observed_at DateTime,
    temperature_c Float64,
    wind_speed_ms Float64,
    wind_direction_deg Float64,
    precipitation_mm Float64,
    relative_humidity Float64
)
ENGINE = MergeTree
ORDER BY observed_at;

CREATE TABLE IF NOT EXISTS fairq.forecasts
(
    station_id String,
    forecast_at DateTime,
    horizon_hours UInt16,
    pollutant String,
    value Float64,
    model_version String
)
ENGINE = MergeTree
ORDER BY (station_id, pollutant, forecast_at, horizon_hours);

CREATE TABLE IF NOT EXISTS fairq.model_versions
(
    model_version String,
    pollutant String,
    trained_at DateTime,
    metrics String
)
ENGINE = MergeTree
ORDER BY (pollutant, trained_at);

CREATE TABLE IF NOT EXISTS fairq.pipeline_runs
(
    ran_at DateTime,
    job String,
    ok UInt8,
    detail String
)
ENGINE = MergeTree
ORDER BY (job, ran_at);

CREATE TABLE IF NOT EXISTS fairq.causal_edges
(
    model_version String,
    trained_at DateTime,
    cause String,
    effect String,
    ks_stat Float64,
    p_value Float64,
    rel_mae Float64,
    accepted UInt8
)
ENGINE = MergeTree
ORDER BY (effect, cause, trained_at);

CREATE TABLE IF NOT EXISTS fairq.causal_graphs
(
    model_version String,
    trained_at DateTime,
    nodes String,
    edges String,
    notes String
)
ENGINE = MergeTree
ORDER BY trained_at;
