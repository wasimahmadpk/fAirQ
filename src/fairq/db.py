import os

import clickhouse_connect

CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "fairq")


def connect():
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username="default",
        password=CLICKHOUSE_PASSWORD,
        database="fairq",
    )
    client.command(
        "ALTER TABLE fairq.weather "
        "ADD COLUMN IF NOT EXISTS relative_humidity Float64 DEFAULT 0"
    )
    client.command(
        "CREATE TABLE IF NOT EXISTS fairq.pipeline_runs "
        "(ran_at DateTime, job String, ok UInt8, detail String) "
        "ENGINE = MergeTree ORDER BY (job, ran_at)"
    )
    client.command(
        "CREATE TABLE IF NOT EXISTS fairq.causal_edges "
        "(model_version String, trained_at DateTime, cause String, effect String, "
        "ks_stat Float64, p_value Float64, rel_mae Float64, accepted UInt8) "
        "ENGINE = MergeTree ORDER BY (effect, cause, trained_at)"
    )
    client.command(
        "CREATE TABLE IF NOT EXISTS fairq.causal_graphs "
        "(model_version String, trained_at DateTime, nodes String, edges String, notes String) "
        "ENGINE = MergeTree ORDER BY trained_at"
    )
    client.command(
        "ALTER TABLE fairq.causal_edges "
        "ADD COLUMN IF NOT EXISTS rel_mae Float64 DEFAULT 0"
    )
    return client
