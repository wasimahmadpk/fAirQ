from datetime import date

from fairq.features import (
    FEATURE_COLS,
    berlin_holidays,
    feature_vector,
    is_berlin_holiday,
    wind_components,
)


def test_feature_vector_matches_columns():
    past = list(range(30))
    row = feature_vector(2, 8, 0, 1, 0, 3.0, 2.0, 0.0, 1.0, 0.0, 70.0, past)
    assert len(row) == len(FEATURE_COLS)


def test_wind_north_and_east():
    sin0, cos0 = wind_components(0.0)
    assert abs(sin0) < 1e-9
    assert abs(cos0 - 1.0) < 1e-9
    sin90, cos90 = wind_components(90.0)
    assert abs(sin90 - 1.0) < 1e-9
    assert abs(cos90) < 1e-9


def test_berlin_holiday_2026():
    days = berlin_holidays(2026)
    assert date(2026, 1, 1) in days
    assert date(2026, 3, 8) in days
    assert date(2026, 4, 3) in days
    assert date(2026, 4, 6) in days
    assert is_berlin_holiday(date(2026, 4, 4)) == 0
    assert is_berlin_holiday(date(2026, 10, 3)) == 1
