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
    return client
