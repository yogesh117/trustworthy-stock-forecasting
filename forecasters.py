# Team 2
# Olin Dsouza (odsouza1@stevens.edu), Yogesh Patil (yp36032@stevens.edu), Kevin Gwinn (kgwinn@stevens.edu)
# Date: 8/24/26
# Description: Engineering Python Final Project, Trustworthy Stock Forecaster
# File name: forecasters.py

"""
forecasters.py module defines the probabilistic forecasting models of the toolkit.

A Forecaster base class holds everything the models share, and two concrete models
INHERIT from it (this is the inheritance class relationship of the project):

RandomWalkForecaster - the classic "no predictability" baseline. The next day's return
is modeled as Normal(0, sigma) where sigma is the plain standard deviation of the
training window, so every training day counts equally.

EWMAForecaster - RiskMetrics style exponentially weighted moving average volatility.
Recent shocks dominate the variance estimate, so the model adapts quickly when the
market switches between calm and volatile regimes.

Both models output a full predictive distribution (mu, sigma) from which a central
prediction interval at any confidence level can be produced. A require_fitted
decorator (with a closure over the wrapped method) guards the prediction methods so
they can never run on a model that was not trained first.
"""

import functools
import math

import numpy as np


def require_fitted(method):
    """
    Decorator that raises a clear error if a forecaster is used before fitting.

    Parameters
    method : callable
        Bound forecaster method that needs a fitted model to make sense.

    Returns
    callable
        Wrapped method that validates the model state before running.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # the closure captures 'method' from the enclosing scope
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before calling {method.__name__}()"
            )
        return method(self, *args, **kwargs)

    return wrapper


class Forecaster:
    """
    Base class for one-step-ahead probabilistic return forecasters.

    A forecaster is fit on a window of historical log-returns and then predicts the
    distribution of the NEXT day's return as a Normal(mu, sigma) pair.

    Attributes
    name (str)
        Human readable model name used in printouts and result tables.
    mu (float or None)
        Predictive mean, set by fit().
    sigma (float or None)
        Predictive standard deviation, set by fit().
    """

    MIN_TRAIN_SIZE = 10    #minimum observations any model needs to be fit sensibly

    def __init__(self, name):
        """store the display name and start in the unfitted state"""
        self.name = name
        self.mu = None
        self.sigma = None

    @property
    def is_fitted(self):
        """True once fit() has produced a predictive distribution"""
        return self.mu is not None and self.sigma is not None

    def _validate_training_data(self, returns):
        """
        Convert training data to a numpy array and check there is enough of it.

        Parameters
        returns : array-like
            Sequence of daily log-returns.

        Returns
        numpy.ndarray
            The returns as a float array.
        """
        values = np.asarray(returns, dtype=float)
        if values.size < self.MIN_TRAIN_SIZE:    #reject windows too small to estimate volatility
            raise ValueError(
                f"{self.name} needs at least {self.MIN_TRAIN_SIZE} returns to fit, got {values.size}"
            )
        return values

    def fit(self, returns):
        """subclasses must implement their own fitting logic"""
        raise NotImplementedError("Subclasses must implement fit()")

    @require_fitted
    def predict_interval(self, level=0.90):
        """
        Central prediction interval for the next day's return.

        Parameters
        level : float
            Confidence level in (0, 1), for example 0.90 for a 90% interval.

        Returns
        tuple
            Immutable (lower, upper) bounds of the interval.
        """
        if not 0.0 < level < 1.0:    #confidence must be a proper probability
            raise ValueError(f"level must be in (0, 1), got {level}")

        # two-sided Normal quantile: z = sqrt(2) * erfinv(level)
        z_score = math.sqrt(2.0) * self._erfinv(level)
        half_width = z_score * self.sigma
        return (self.mu - half_width, self.mu + half_width)

    @staticmethod
    def _erfinv(y, tolerance=1e-12):
        """
        Inverse error function computed with Newton's method on math.erf.

        Parameters
        y : float
            Target value in (-1, 1).
        tolerance : float
            Convergence threshold for the iteration.

        Returns
        float
            x such that erf(x) == y.
        """
        x = 0.0
        for _ in range(100):    #Newton iterations, converges in a handful of steps
            error = math.erf(x) - y
            if abs(error) < tolerance:
                break
            # erf'(x) = 2/sqrt(pi) * exp(-x^2)
            x -= error / (2.0 / math.sqrt(math.pi) * math.exp(-x * x))
        return x

    def __str__(self):
        """overload print to show the model name and its fitted distribution"""
        if self.is_fitted:
            return f"{self.name}(mu={self.mu:.6f}, sigma={self.sigma:.6f})"
        return f"{self.name}(unfitted)"

    def __eq__(self, other):
        """overload == so two forecasters are equal when their configurations match"""
        if not isinstance(other, Forecaster):
            return NotImplemented
        return type(self) is type(other) and self.config() == other.config()

    def config(self):
        """return the model configuration as an immutable tuple used by __eq__"""
        return (type(self).__name__, self.name)


class RandomWalkForecaster(Forecaster):
    """
    Random-walk baseline: next return ~ Normal(0, sample standard deviation).

    Every day in the training window counts equally, so the model reacts slowly -
    yesterday matters exactly as much as a day eleven months ago.
    """

    def __init__(self):
        """initialize with the fixed display name"""
        super().__init__(name="RandomWalk")

    def fit(self, returns):
        """
        Fit by estimating the sample standard deviation of the training window.

        Parameters
        returns : array-like
            Sequence of daily log-returns.

        Returns
        RandomWalkForecaster
            self, so calls can be chained like model.fit(x).predict_interval()
        """
        values = self._validate_training_data(returns)
        self.mu = 0.0
        self.sigma = float(np.std(values, ddof=1))    #ddof=1 gives the unbiased sample std
        return self


class EWMAForecaster(Forecaster):
    """
    RiskMetrics EWMA volatility model.

    The variance evolves as var_t = decay * var_<t-1> + (1 - decay) * r_<t-1>^2 so
    recent shocks dominate and the model adapts when the market regime changes.

    Attributes
    decay (float)
        Exponential decay factor lambda in (0, 1). 0.94 is the standard
        RiskMetrics value for daily data.
    """

    def __init__(self, decay=0.94):
        """validate and store the decay factor, then initialize the base class"""
        if not 0.0 < decay < 1.0:    #decay outside (0,1) would blow up the recursion
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        super().__init__(name=f"EWMA(lambda={decay})")
        self.decay = decay

    def fit(self, returns):
        """
        Fit by running the EWMA variance recursion across the training window.

        Parameters
        returns : array-like
            Sequence of daily log-returns.

        Returns
        EWMAForecaster
            self, so calls can be chained.
        """
        values = self._validate_training_data(returns)

        variance = float(np.var(values, ddof=1))    #seed the recursion with the sample variance
        for observed_return in values:              #then update it one day at a time
            variance = self.decay * variance + (1.0 - self.decay) * observed_return**2

        self.mu = 0.0
        self.sigma = math.sqrt(variance)
        return self

    def config(self):
        """configuration tuple also includes the decay so EWMA(0.94) != EWMA(0.90)"""
        return super().config() + (self.decay,)


if __name__ == "__main__":    #allow module to run on its own to test
    rng = np.random.default_rng(seed=551)
    synthetic = rng.normal(0.0, 0.01, size=250)    #one synthetic year of calm returns

    for model in (RandomWalkForecaster(), EWMAForecaster()):
        model.fit(synthetic)
        low, high = model.predict_interval(0.90)
        print(f"{model}  90% interval: ({low:.5f}, {high:.5f})")

    print("RandomWalk == RandomWalk:", RandomWalkForecaster() == RandomWalkForecaster())
    print("RandomWalk == EWMA:      ", RandomWalkForecaster() == EWMAForecaster())
