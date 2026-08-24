# Team 2
# Olin Dsouza (odsouza1@stevens.edu), Yogesh Patil (ypatel33@stevens.edu), Kevin Gwinn (kgwinn@stevens.edu)
# Date: 8/24/26
# Description: Engineering Python Final Project, Trustworthy Stock Forecaster
# File name: test_toolkit.py

"""
test_toolkit.py is the Pytest suite for the Trustworthy Stock Forecaster toolkit.

It validates the program logic on synthetic series with KNOWN outcomes: the coverage
arithmetic, regime labeling, CRPS ordering, the rolling-window generator, the
portfolio operator overloads, and both required exception scenarios
(FileNotFoundError for a missing data file and ValueError for insufficient data).

Run from the project root with:  pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from backtest import rolling_window_splits, summarize_by_regime
from calibration import crps_normal, empirical_coverage
from data_loader import (
    compute_log_returns,
    label_regime,
    load_daily_data,
    rolling_volatility,
)
from forecasters import EWMAForecaster, RandomWalkForecaster
from portfolio import Portfolio, StockAssets


def _synthetic_price_frame(n_days=300, seed=551):
    """
    Build a synthetic daily price DataFrame shaped like load_daily_data output.

    Parameters
    n_days : int
        Number of trading days to simulate.
    seed : int
        Random seed so tests are reproducible.

    Returns
    pd.DataFrame
        Positive Close prices indexed by business days.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    log_returns = rng.normal(0.0, 0.01, size=n_days)
    prices = 100.0 * np.exp(np.cumsum(log_returns))    #build a price path from the returns
    return pd.DataFrame({"Close": prices}, index=dates)


def test_empirical_coverage_known_outcome():
    """coverage must equal the exact fraction of hits on a hand-built example"""
    intervals = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    observed = [0.0, 0.5, 2.0, -3.0]    #exactly 2 of the 4 observations fall inside
    assert empirical_coverage(intervals, observed) == pytest.approx(0.5)


def test_regime_labeling_flags_volatile_period():
    """days inside an injected high-volatility block must be labeled volatile"""
    rng = np.random.default_rng(7)
    calm_part = rng.normal(0.0, 0.005, size=200)
    wild_part = rng.normal(0.0, 0.05, size=60)    #ten times the volatility
    returns = pd.Series(
        np.concatenate([calm_part, wild_part, calm_part]),
        index=pd.bdate_range("2020-01-01", periods=460),
    )

    #use the 70th percentile of rolling volatility as the threshold
    vol = rolling_volatility(returns, window=20).dropna()
    labels = label_regime(returns, window=20, threshold=vol.quantile(0.70))

    wild_center = labels.iloc[230:250]    #the middle of the wild block
    assert (wild_center == "volatile").all()
    #the tail of the final calm block must be almost entirely calm (single
    #borderline days near the threshold are a normal statistical artifact)
    calm_tail = labels.iloc[-50:]
    assert (calm_tail == "calm").mean() >= 0.90


def test_crps_rewards_honest_uncertainty():
    """an honest forecaster must beat an over-confident one on CRPS"""
    rng = np.random.default_rng(551)
    truth = rng.normal(0.0, 0.01, size=2000)

    honest = crps_normal(0.0, 0.01, truth).mean()
    overconfident = crps_normal(0.0, 0.001, truth).mean()

    assert honest < overconfident


def test_random_walk_coverage_close_to_nominal():
    """on truly Gaussian data the random walk's 90% coverage must be about 90%"""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0, 0.01, size=1500)

    model = RandomWalkForecaster().fit(returns[:500])
    interval = model.predict_interval(0.90)
    coverage = empirical_coverage([interval] * len(returns[500:]), returns[500:])

    assert 0.85 <= coverage <= 0.95


def test_ewma_reacts_to_recent_volatility():
    """EWMA sigma must rise sharply when recent returns turn violent"""
    rng = np.random.default_rng(3)
    calm_history = rng.normal(0.0, 0.005, size=200)
    violent_tail = np.concatenate([calm_history[:150], rng.normal(0.0, 0.05, size=50)])

    calm_sigma = EWMAForecaster().fit(calm_history).sigma
    violent_sigma = EWMAForecaster().fit(violent_tail).sigma

    assert violent_sigma > 2.0 * calm_sigma


def test_rolling_window_generator_boundaries():
    """the generator must produce contiguous, correctly sized windows"""
    splits = list(rolling_window_splits(300, train_size=100, test_size=20, step=20))

    assert splits[0] == (0, 100, 120)
    for train_start, train_end, test_end in splits:
        assert train_end - train_start == 100
        assert test_end - train_end == 20
        assert test_end <= 300
    assert len(splits) == 10    #300 observations, window 120, step 20 -> 10 splits


def test_missing_file_raises_file_not_found():
    """exception scenario 1: loading a nonexistent CSV must raise"""
    with pytest.raises(FileNotFoundError):
        load_daily_data("data/DOES_NOT_EXIST.csv")


def test_insufficient_data_raises_value_error():
    """exception scenario 2: too little data for the rolling window must raise"""
    tiny = pd.Series([0.01, -0.02, 0.005])
    with pytest.raises(ValueError):
        label_regime(tiny, window=20)


def test_forecaster_equality_by_configuration():
    """__eq__ must compare configurations, not object identities"""
    assert RandomWalkForecaster() == RandomWalkForecaster()
    assert EWMAForecaster(0.94) == EWMAForecaster(0.94)
    assert EWMAForecaster(0.94) != EWMAForecaster(0.90)
    assert RandomWalkForecaster() != EWMAForecaster()


def test_unfitted_forecaster_is_rejected():
    """the require_fitted decorator must block predictions from unfitted models"""
    with pytest.raises(RuntimeError):
        RandomWalkForecaster().predict_interval(0.90)


def test_portfolio_operators_and_type_check():
    """__len__, __add__, __eq__, __str__ and the TypeError guard must all work"""
    frame = _synthetic_price_frame()
    alpha = Portfolio(name="Alpha").add_asset(StockAssets("AAA", frame))
    beta = Portfolio(name="Beta").add_asset(StockAssets("BBB", frame))

    combined = alpha + beta    #__add__ overload
    assert len(combined) == 2
    assert sorted(combined.symbols()) == ["AAA", "BBB"]
    assert "AAA" in str(combined)    #__str__ overload
    assert combined == combined      #__eq__ overload
    assert alpha != beta

    with pytest.raises(TypeError):    #only StockAssets objects may be added
        alpha.add_asset("not an asset")


def test_summarize_by_regime_partitions_records():
    """per-regime reports must partition the records exactly"""
    rng = np.random.default_rng(9)
    records = [
        {
            "date": i,
            "mu": 0.0,
            "sigma": 0.01,
            "actual": float(rng.normal(0.0, 0.01)),
            "regime": "calm" if i % 2 == 0 else "volatile",
        }
        for i in range(200)
    ]

    reports = summarize_by_regime(records)

    assert set(reports) == {"calm", "volatile", "all"}
    assert reports["calm"]["n"] + reports["volatile"]["n"] == reports["all"]["n"]
    assert reports["all"]["mean_crps"] > 0.0


def test_log_returns_recover_price_ratio():
    """the sum of log-returns must equal the log of the total price ratio"""
    frame = pd.DataFrame({"Close": [100.0, 110.0, 99.0, 120.5]})
    returns = compute_log_returns(frame)
    assert returns.sum() == pytest.approx(np.log(120.5 / 100.0))


if __name__ == "__main__":    #allow the suite to run directly without the pytest CLI
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
