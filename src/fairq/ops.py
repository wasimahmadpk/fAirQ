"""Shared check objects and pipeline_runs writes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from fairq.db import connect


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def skill_score(model_mae: float, persist_mae: float) -> float:
    if persist_mae == 0:
        return 0.0
    return (persist_mae - model_mae) / persist_mae


def write_run(job: str, checks: list[Check]) -> None:
    db = connect()
    db.insert(
        "pipeline_runs",
        [[
            datetime.now(timezone.utc).replace(tzinfo=None),
            job,
            int(all(check.ok for check in checks)),
            json.dumps({check.name: {"ok": check.ok, "detail": check.detail} for check in checks}),
        ]],
        column_names=["ran_at", "job", "ok", "detail"],
    )


def report(job: str, checks: list[Check]) -> bool:
    for check in checks:
        mark = "ok" if check.ok else "FAIL"
        print(f"{job}  {mark:4}  {check.name}  {check.detail}")
    write_run(job, checks)
    return all(check.ok for check in checks)
