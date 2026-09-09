from fairq.ops import Check, report
from fairq.validate import MIN_SKILL_24H, model_checks


class FakeResult:
    def __init__(self, rows):
        self.result_rows = rows


class FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, _sql):
        return FakeResult(self._rows)


def test_model_checks_pass_on_positive_skill():
    db = FakeDB(
        [
            ("NO2", '{"skill_24h": 0.3}'),
            ("PM10", '{"skill_24h": 0.1}'),
            ("PM2.5", '{"skill_24h": 0.2}'),
        ]
    )
    checks = model_checks(db)
    assert all(check.ok for check in checks)
    assert MIN_SKILL_24H == 0.0


def test_model_checks_fail_when_skill_missing():
    db = FakeDB([("NO2", '{"mae_1h": 2.0}')])
    names = {check.name: check.ok for check in model_checks(db)}
    assert names["skill_24h_NO2"] is False
    assert names["model_PM10"] is False


def test_report_returns_false_on_failure(monkeypatch):
    monkeypatch.setattr("fairq.ops.write_run", lambda job, checks: None)
    ok = report("validate", [Check("rows", True, "n=1"), Check("skill", False, "-0.1")])
    assert ok is False
