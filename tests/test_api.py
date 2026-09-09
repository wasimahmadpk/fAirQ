from fairq.api import health


def test_health():
    assert health()["status"] == "ok"
