import numpy as np
import pytest
from fraud_detection.train import generate_data, metrics_at_threshold, run, select_threshold


def test_data_is_reproducible_and_imbalanced():
    x1, y1 = generate_data(2000, 42)
    x2, y2 = generate_data(2000, 42)
    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert y1.mean() < 0.05


def test_threshold_respects_precision_when_feasible():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    threshold = select_threshold(y, scores, min_precision=1.0)
    result = metrics_at_threshold(y, scores, threshold)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_no_feasible_threshold_abstains():
    assert select_threshold(np.array([1, 0]), np.array([0.1, 0.9]), 1.0) == 1.0


def test_run_is_reproducible():
    first = run(seed=123, n_samples=2000, min_precision=0.30)
    second = run(seed=123, n_samples=2000, min_precision=0.30)
    assert first == second


def test_run_has_disjoint_split_sizes_and_finite_metrics():
    result = run(n_samples=2000)
    assert sum(result["split"].values()) == 2000
    assert 0 <= result["test"]["average_precision"] <= 1


def test_bad_inputs():
    with pytest.raises(ValueError):
        run(min_precision=0)
    with pytest.raises(ValueError):
        run(n_samples=10)
