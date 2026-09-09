from fairq.ingest import month_windows
from fairq.ops import skill_score


def test_skill_score_win_and_tie():
    assert skill_score(4.0, 8.0) == 0.5
    assert skill_score(5.0, 5.0) == 0.0
    assert skill_score(6.0, 5.0) == -0.2
    assert skill_score(1.0, 0.0) == 0.0


def test_month_windows_oldest_first():
    windows = month_windows(12)
    assert len(windows) == 12
    assert windows[0][0] < windows[-1][0]
    assert windows[-1][1] >= windows[-1][0]
