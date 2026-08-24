# Team 2
# Olin Dsouza (odsouza1@stevens.edu), Yogesh Patil (yp36032@stevens.edu), Kevin Gwinn (kgwinn@stevens.edu)
# Date: 8/24/26
# Description: Engineering Python Final Project, Trustworthy Stock Forecaster
# File name: backtest.py

"""
backtest.py module drives the whole experiment.

rolling_window_splits is a GENERATOR function that yields successive train/test index
windows over the return history (train one year, test one month, slide monthly) -
exactly the way a model would be re-estimated and used in real time.

run_backtest re-fits every forecaster on each training window and records its
predictive distribution against every realized return in the test window, together
with that day's market regime.

summarize_by_regime splits the records per regime (with set operations validating the
labels) and computes a calibration report for each part, and reports_to_frame
flattens the nested reports into one tidy pandas DataFrame ready to write to CSV.
"""

import pandas as pd

from calibration import compute_calibration


def rolling_window_splits(n_observations, train_size=252, test_size=21, step=21):
    """
    Generator yielding successive rolling train/test index windows.

    Each split trains on train_size consecutive days and tests on the test_size days
    that immediately follow, then the whole window slides forward by step days.

    Parameters
    n_observations : int
        Total number of return observations available.
    train_size : int
        Training window length (252 = one trading year).
    test_size : int
        Evaluation window length (21 = one trading month).
    step : int
        Days the window advances between splits.

    Yields
    tuple
        (train_start, train_end, test_end) index boundaries, meaning
        train = [train_start, train_end) and test = [train_end, test_end).
    """
    if min(train_size, test_size, step) <= 0:    #zero or negative sizes make no sense
        raise ValueError("train_size, test_size, and step must be positive")
    if n_observations < train_size + test_size:    #need room for at least one split
        raise ValueError(
            f"Need at least {train_size + test_size} observations for one split, got {n_observations}"
        )

    train_start = 0
    while train_start + train_size + test_size <= n_observations:    #slide until the test window runs out of data
        train_end = train_start + train_size
        yield (train_start, train_end, train_end + test_size)    #yield makes this a generator
        train_start += step


def run_backtest(returns, regimes, forecasters, train_size=252, test_size=21, step=21):
    """
    Run the rolling-window backtest for every forecaster.

    For each split every forecaster is fit on the training returns and its predictive
    distribution is compared against each realized return in the test window (a
    one-step-ahead scheme re-estimated monthly).

    Parameters
    returns : pd.Series
        Daily log-returns indexed by date (from data_loader.compute_log_returns).
    regimes : pd.Series
        'calm'/'volatile' labels indexed by date (from data_loader.label_regime).
    forecasters : list
        Forecaster instances to evaluate.
    train_size : int
        Training window length in days.
    test_size : int
        Test window length in days.
    step : int
        Window advance between splits.

    Returns
    dict
        {forecaster_name: [record, ...]} where each record is a dict with keys
        date, mu, sigma, actual, regime.
    """
    aligned = returns.loc[regimes.index]    #only evaluate days that have a regime label
    records_by_model = {str(model.name): [] for model in forecasters}

    #outer loop walks the rolling windows produced by the generator
    for train_start, train_end, test_end in rolling_window_splits(
            len(aligned), train_size, test_size, step):
        train_returns = aligned.iloc[train_start:train_end]
        test_returns = aligned.iloc[train_end:test_end]

        #inner loop re-fits every forecaster on this training window
        for model in forecasters:
            model.fit(train_returns.to_numpy())
            for date, actual in test_returns.items():    #score the model on each test day
                records_by_model[str(model.name)].append(
                    {
                        "date": date,
                        "mu": model.mu,
                        "sigma": model.sigma,
                        "actual": float(actual),
                        "regime": regimes.loc[date],
                    }
                )

    return records_by_model


def summarize_by_regime(records):
    """
    Split forecast records by market regime and calibrate each part separately.

    Parameters
    records : list of dict
        Forecast records produced by run_backtest.

    Returns
    dict
        {regime_name: calibration_report} plus a combined 'all' entry.
    """
    if not records:
        raise ValueError("No records to summarize")

    #set operations validate the regime labels found in the records
    observed_regimes = {rec["regime"] for rec in records}    #set comprehension
    expected_regimes = {"calm", "volatile"}
    known = observed_regimes & expected_regimes      #intersection = labels we understand
    unknown = observed_regimes - expected_regimes    #difference = anything unexpected

    if unknown:    #fail loudly on corrupted labels instead of silently mis-grouping
        raise ValueError(f"Unexpected regime labels: {sorted(unknown)}")

    #dict comprehension with a nested list comprehension filtering per regime
    reports = {
        regime: compute_calibration(
            [rec for rec in records if rec["regime"] == regime]
        )
        for regime in sorted(known)
    }
    reports["all"] = compute_calibration(records)    #plus the combined report
    return reports


def reports_to_frame(reports_by_model):
    """
    Flatten nested calibration reports into one tidy DataFrame.

    Parameters
    reports_by_model : dict
        {model_name: {regime_name: report}} nested reports.

    Returns
    pd.DataFrame
        One row per (model, regime) with CRPS and coverage columns, ready for CSV.
    """
    rows = []
    for model_name, regime_reports in reports_by_model.items():
        for regime_name, report in regime_reports.items():
            row = {
                "model": model_name,
                "regime": regime_name,
                "n_days": report["n"],
                "mean_crps": report["mean_crps"],
                "median_crps": report["median_crps"],
            }
            #one column per confidence level, for example coverage_90
            row.update(
                {
                    f"coverage_{int(level * 100)}": value
                    for level, value in report["coverage"].items()
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":    #allow module to run on its own to test
    splits = list(rolling_window_splits(756, train_size=252, test_size=21, step=21))
    print(f"756 observations -> {len(splits)} rolling splits")
    print("First split:", splits[0], " Last split:", splits[-1])
