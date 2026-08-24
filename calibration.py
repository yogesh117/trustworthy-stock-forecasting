# Team 2
# Olin Dsouza (odsouza1@stevens.edu), Yogesh Patil (yp36032@stevens.edu), Kevin Gwinn (kgwinn@stevens.edu)
# Date: 8/24/26
# Description: Engineering Python Final Project, Trustworthy Stock Forecaster
# File name: calibration.py

"""
calibration.py module measures HOW TRUSTWORTHY a probabilistic forecast is.

crps_normal computes the Continuous Ranked Probability Score of a Gaussian forecast in
closed form. CRPS generalizes absolute error to full predictive distributions - lower
is better - and it rewards both accuracy and honest uncertainty.

empirical_coverage computes the fraction of realized returns that actually fall inside
the stated prediction intervals. A well calibrated 90% interval should contain the
truth on about 90% of days; much less than that means the model is over-confident.

compute_calibration aggregates both metrics into one report, and
plot_reliability_diagram draws nominal vs empirical coverage per market regime so any
over/under-confidence is visible at a glance (points below the diagonal = intervals
too narrow = over-confidence).
"""

import math
import statistics

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)    #confidence levels evaluated everywhere (immutable tuple)


def _standard_normal_cdf(z):
    """
    Standard normal CDF using the built in math.erf function.

    Parameters
    z : numpy.ndarray
        Standardized values.

    Returns
    numpy.ndarray
        Phi(z) for each element.
    """
    #map + lambda apply the scalar erf formula across the whole array
    return np.array(list(map(lambda v: 0.5 * (1.0 + math.erf(v / math.sqrt(2.0))), z)))


def crps_normal(mu, sigma, observed):
    """
    Closed form CRPS of Normal(mu, sigma) forecasts against observations.

    Formula: CRPS = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (x - mu) / sigma. Derivation in Gneiting & Raftery (2007).

    Parameters
    mu : float or array
        Predictive means.
    sigma : float or array
        Predictive standard deviations, all strictly positive.
    observed : float or array
        Realized values.

    Returns
    numpy.ndarray
        CRPS value per observation (lower is better).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    observed = np.asarray(observed, dtype=float)

    if np.any(sigma <= 0.0):    #a zero or negative spread is not a valid distribution
        raise ValueError("All predictive sigmas must be positive")

    z = (observed - mu) / sigma
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    cdf = _standard_normal_cdf(z)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def empirical_coverage(intervals, observed):
    """
    Fraction of observations that fall inside their stated intervals.

    Parameters
    intervals : iterable of tuple
        (lower, upper) interval per day.
    observed : iterable of float
        Realized returns aligned with the intervals.

    Returns
    float
        Empirical coverage in [0, 1].
    """
    interval_list = list(intervals)
    observed_list = list(observed)

    if len(interval_list) != len(observed_list):    #misaligned inputs would silently corrupt the metric
        raise ValueError(
            f"Got {len(interval_list)} intervals but {len(observed_list)} observations"
        )
    if not interval_list:
        raise ValueError("Cannot compute coverage of an empty sample")

    #zip pairs each interval with its outcome, the list comprehension counts the hits
    hits = [
        1 if lower <= actual <= upper else 0
        for (lower, upper), actual in zip(interval_list, observed_list)
    ]
    return sum(hits) / len(hits)


def compute_calibration(records, levels=DEFAULT_LEVELS):
    """
    Aggregate per-day forecast records into one calibration report.

    Parameters
    records : list of dict
        One dict per forecast day with keys mu, sigma, actual
        (produced by backtest.run_backtest).
    levels : tuple
        Confidence levels at which coverage is evaluated.

    Returns
    dict
        {"n": ..., "mean_crps": ..., "median_crps": ..., "coverage": {level: value}}
    """
    if not records:
        raise ValueError("Cannot calibrate an empty set of records")

    mus = np.array([rec["mu"] for rec in records])          #list comprehensions unpack the records
    sigmas = np.array([rec["sigma"] for rec in records])
    actuals = np.array([rec["actual"] for rec in records])

    crps_values = crps_normal(mus, sigmas, actuals)

    #dict comprehension: empirical coverage at every confidence level
    coverage = {
        level: empirical_coverage(
            [_normal_interval(m, s, level) for m, s in zip(mus, sigmas)],
            actuals,
        )
        for level in levels
    }

    #the built in statistics module supplies the outlier-robust median
    return {
        "n": len(records),
        "mean_crps": float(np.mean(crps_values)),
        "median_crps": statistics.median(crps_values.tolist()),
        "coverage": coverage,
    }


def _normal_interval(mu, sigma, level):
    """
    Central Normal interval at one confidence level.

    Parameters
    mu : float
        Predictive mean.
    sigma : float
        Predictive standard deviation.
    level : float
        Confidence level in (0, 1).

    Returns
    tuple
        (lower, upper) bounds.
    """
    from forecasters import Forecaster    #imported here to avoid a circular import at load time

    z_score = math.sqrt(2.0) * Forecaster._erfinv(level)
    return (mu - z_score * sigma, mu + z_score * sigma)


def plot_reliability_diagram(reports_by_regime, title, output_path=None):
    """
    Plot nominal vs empirical coverage for each market regime.

    A perfectly calibrated model lies on the diagonal. Points below the diagonal
    mean over-confidence (intervals too narrow), points above mean under-confidence.

    Parameters
    reports_by_regime : dict
        {regime_name: calibration_report} as produced by compute_calibration.
    title : str
        Plot title, usually the model name.
    output_path : str or None
        If given the figure is also saved to this path as a PNG.

    Returns
    matplotlib.figure.Figure
        The created figure.
    """
    figure, axis = plt.subplots(figsize=(6.5, 5.5))

    #the diagonal is the perfect-calibration reference
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")

    for regime_name, report in sorted(reports_by_regime.items()):
        levels = sorted(report["coverage"])
        empirical = [report["coverage"][lvl] for lvl in levels]
        axis.plot(levels, empirical, marker="o", label=f"{regime_name} (n={report['n']})")

    axis.set_xlabel("Nominal coverage (stated confidence)")
    axis.set_ylabel("Empirical coverage (observed)")
    axis.set_title(title)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.legend(loc="lower right")
    axis.grid(alpha=0.3)
    figure.tight_layout()

    if output_path is not None:    #write the PNG to disk when a path is provided (data output)
        figure.savefig(output_path, dpi=150)

    return figure


if __name__ == "__main__":    #allow module to run on its own to test
    #an honest forecaster must score a lower CRPS than an over-confident one
    rng = np.random.default_rng(seed=551)
    truth = rng.normal(0.0, 0.01, size=500)

    honest = crps_normal(0.0, 0.01, truth).mean()
    overconfident = crps_normal(0.0, 0.002, truth).mean()
    print(f"CRPS honest={honest:.6f}  overconfident={overconfident:.6f}")
    print("Honest forecast wins:", honest < overconfident)
