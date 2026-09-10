import numpy as np

from fairq.cdmi import (
    accept_edge,
    allowed_pair,
    gaussian_knockoffs,
    ks_2samp,
    mean_noise_knockoffs,
    residual_report,
    studentize,
    traffic_proxy,
    uniform_knockoffs,
)


def test_accept_edge_needs_both_interventions():
    assert accept_edge("wind", "no2_street", 0.004, 0.04) is False
    assert accept_edge("wind", "no2_street", 0.04, 0.004) is False
    assert accept_edge("wind", "no2_street", 0.04, 0.02) is True


def test_accept_edge_path_recovers_wind_like_links():
    assert accept_edge("wind", "no2_street", 0.004, 0.15, rel_mae_gaussian=0.03) is True
    assert accept_edge("wind", "no2_street", 0.004, 0.15, rel_mae_gaussian=0.004) is False


def test_accept_edge_blocks_reverse_physics():
    assert allowed_pair("no2_street", "temp") is False
    assert allowed_pair("traffic", "temp") is False
    assert allowed_pair("humidity", "traffic") is False
    assert accept_edge("no2_street", "temp", 0.5, 0.5) is False
    assert accept_edge("traffic", "humidity", 0.5, 0.5) is False
    assert accept_edge("wind", "no2_bg", 0.04, 0.02) is True
    assert allowed_pair("humidity", "wind") is False
    assert allowed_pair("temp", "humidity") is True


def test_traffic_quiet_vs_rush():
    assert traffic_proxy(8, 0, 0) > traffic_proxy(8, 6, 0)
    assert traffic_proxy(8, 0, 1) < traffic_proxy(8, 0, 0)


def test_ks_identical_samples_not_rejected():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=200)
    stat, p_value = ks_2samp(sample, sample)
    assert stat == 0.0
    assert p_value > 0.5


def test_ks_shifted_samples_rejected():
    rng = np.random.default_rng(1)
    left = rng.normal(size=200)
    right = rng.normal(loc=2.0, size=200)
    _stat, p_value = ks_2samp(left, right)
    assert p_value < 0.05


def test_ks_is_full_cdf_not_mean_only():
    rng = np.random.default_rng(3)
    left = rng.normal(scale=1.0, size=400)
    right = rng.normal(scale=3.0, size=400)
    stat, p_value = ks_2samp(left, right)
    assert stat > 0.15
    assert p_value < 0.05
    report = residual_report(left, right)
    assert abs(report["delta_mean"]) < 0.35
    assert report["rel_std"] > 1.0


def test_studentized_ks_ignores_pure_mean_shift():
    rng = np.random.default_rng(4)
    left = rng.normal(size=300)
    right = left + 3.0
    raw, _ = ks_2samp(left, right)
    shaped, _ = ks_2samp(studentize(left), studentize(right))
    assert raw > 0.5
    assert shaped < 0.08


def test_gaussian_knockoffs_shape_and_mean():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(400, 4))
    knock = gaussian_knockoffs(values, rng)
    assert knock.shape == values.shape
    assert np.allclose(knock.mean(axis=0), values.mean(axis=0), atol=0.35)


def test_mean_noise_hugs_the_column_mean():
    rng = np.random.default_rng(5)
    values = rng.normal(loc=10.0, scale=4.0, size=(500, 3))
    knock = mean_noise_knockoffs(values, rng)
    assert knock.shape == values.shape
    assert np.allclose(knock.mean(axis=0), values.mean(axis=0), atol=0.15)
    assert np.all(knock.std(axis=0) < 0.2 * values.std(axis=0))


def test_uniform_stays_inside_observed_range():
    rng = np.random.default_rng(6)
    values = rng.uniform(2.0, 8.0, size=(300, 2))
    knock = uniform_knockoffs(values, rng)
    assert knock.min() >= values.min() - 1e-9
    assert knock.max() <= values.max() + 1e-9
