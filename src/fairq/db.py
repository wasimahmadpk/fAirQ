import os

import clickhouse_connect

CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD", "fairq")


def connect():
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username="default",
        password=CLICKHOUSE_PASSWORD,
        database="fairq",
    )
