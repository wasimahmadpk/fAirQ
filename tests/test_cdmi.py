import numpy as np

from fairq.cdmi import accept_edge, gaussian_knockoffs, ks_2samp, traffic_proxy


def test_accept_edge_drops_weak_links():
    assert accept_edge(0.016, 1.0, 0.004) is False
    assert accept_edge(0.05, 1.0, 0.005) is False
    assert accept_edge(0.005, 0.05, 0.0) is False
    assert accept_edge(0.032, 1.0, 0.019) is True
    assert accept_edge(0.02, 0.05, 0.02) is True


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


def test_gaussian_knockoffs_shape_and_mean():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(400, 4))
    knock = gaussian_knockoffs(values, rng)
    assert knock.shape == values.shape
    assert np.allclose(knock.mean(axis=0), values.mean(axis=0), atol=0.35)
