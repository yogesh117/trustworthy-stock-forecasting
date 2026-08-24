# Team 2
# Olin Dsouza (odsouza1@stevens.edu), Yogesh Patel (ypatel33@stevens.edu), Kevin Gwinn (kgwinn@stevens.edu)
# Date: 8/24/26
# Description: Engineering Python Final Project, Trustworthy Stock Forecaster
# File name: test_edge_cases.py

"""
test_edge_cases.py adds edge-case and input-validation tests on top of the main
test_toolkit.py suite. These focus on the failure paths: bad constructor arguments,
unknown lookups, and invalid metric inputs.

Run from the project root with:  pytest -v
"""

import numpy as np
import pytest

from forecasters import EWMAForecaster, RandomWalkForecaster
from portfolio import Portfolio


def test_ewma_rejects_invalid_decay():
    """decay outside (0, 1) must be rejected at construction time"""
    with pytest.raises(ValueError):
        EWMAForecaster(decay=1.5)
    with pytest.raises(ValueError):
        EWMAForecaster(decay=0.0)


def test_predict_interval_rejects_bad_level():
    """confidence levels outside (0, 1) must raise a ValueError"""
    model = RandomWalkForecaster().fit(np.random.default_rng(1).normal(0, 0.01, 100))
    with pytest.raises(ValueError):
        model.predict_interval(1.5)
    with pytest.raises(ValueError):
        model.predict_interval(0.0)


def test_portfolio_unknown_symbol_raises_keyerror():
    """asking a portfolio for a symbol it does not hold must raise KeyError"""
    with pytest.raises(KeyError):
        Portfolio(name="Empty").get_asset("AAPL")


def test_crps_rejects_nonpositive_sigma():
    """a zero or negative predictive sigma is not a valid distribution"""
    from calibration import crps_normal
    with pytest.raises(ValueError):
        crps_normal(0.0, 0.0, 0.01)
    with pytest.raises(ValueError):
        crps_normal(0.0, -0.5, 0.01)


def test_coverage_rejects_misaligned_inputs():
    """intervals and observations of different lengths must be rejected"""
    from calibration import empirical_coverage
    with pytest.raises(ValueError):
        empirical_coverage([(-1, 1), (-1, 1)], [0.0])
    with pytest.raises(ValueError):
        empirical_coverage([], [])


def test_generator_rejects_too_short_series():
    """a series shorter than one train+test window must raise a ValueError"""
    from backtest import rolling_window_splits
    with pytest.raises(ValueError):
        list(rolling_window_splits(100, train_size=252, test_size=21))


if __name__ == "__main__":    #allow the suite to run directly without the pytest CLI
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
