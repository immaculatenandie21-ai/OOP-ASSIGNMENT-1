"""UBOS district population forecaster (Mini-Project 1).

Classes
-------
DistrictPopulation   validated population series (thousands) for one district
Forecaster (ABC)     fit(years, values) / predict(horizon) interface
  LinearTrend, ExponentialGrowth, FibonacciRatio

Functions
---------
mae, rmse, mape, evaluate_models, bootstrap_prediction_intervals,
classrooms_needed

Units: population in thousands of people; years are calendar years.
"""
from __future__ import annotations

import copy
import math
import statistics
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------
# Task 1-3: DistrictPopulation
# --------------------------------------------------------------------------
class DistrictPopulation:
    """Consecutive yearly population estimates (thousands) for one district."""

    def __init__(self, name: str, years: Sequence[int], population: Sequence[float]) -> None:
        yrs = np.asarray(years)
        pop = np.asarray(population, dtype=float)
        if yrs.ndim != 1 or pop.ndim != 1:
            raise ValueError("years and population must be 1-D sequences")
        if len(yrs) != len(pop):
            raise ValueError(f"years ({len(yrs)}) and population ({len(pop)}) must have equal length")
        if len(pop) < 2:
            raise ValueError("need at least two observations")
        if not np.all(np.isfinite(pop)) or np.any(pop < 0):
            raise ValueError("population values must be finite and non-negative")
        if np.any(np.diff(yrs) != 1):
            raise ValueError("years must be consecutive and increasing")
        self.name = name
        self.years = yrs.astype(int)
        self.population = pop

    # -- basics -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.population)

    def __repr__(self) -> str:
        return (f"DistrictPopulation({self.name!r}, {self.years[0]}-{self.years[-1]}, "
                f"last={self.population[-1]:,.0f}k)")

    def split(self, last_train_year: int) -> tuple["DistrictPopulation", "DistrictPopulation"]:
        """Split into (train, test) at ``last_train_year`` (inclusive)."""
        mask = self.years <= last_train_year
        if mask.sum() < 2 or (~mask).sum() < 1:
            raise ValueError("split leaves too few training or test points")
        return (DistrictPopulation(self.name, self.years[mask], self.population[mask]),
                DistrictPopulation(self.name, self.years[~mask], self.population[~mask]))

    # -- Task 2: descriptive statistics ------------------------------------
    def describe_statistics(self) -> dict[str, float]:
        """mean, median, sample variance (n-1) and stdev via ``statistics``."""
        v = self.population.tolist()
        return {"mean": statistics.mean(v), "median": statistics.median(v),
                "variance": statistics.variance(v), "std": statistics.stdev(v)}

    def describe_numpy(self, ddof: int = 0) -> dict[str, float]:
        """Same measures via NumPy. ``ddof`` is the divisor offset: the
        variance divides by n - ddof (np.var defaults to ddof=0)."""
        p = self.population
        return {"mean": float(np.mean(p)), "median": float(np.median(p)),
                "variance": float(np.var(p, ddof=ddof)), "std": float(np.std(p, ddof=ddof))}

    # -- Task 3: growth ----------------------------------------------------
    def yoy_growth(self) -> np.ndarray:
        """Year-on-year growth rates (fractions), length n - 1."""
        prev = self.population[:-1]
        if np.any(prev == 0):
            raise ValueError("growth rate undefined after a zero population")
        return self.population[1:] / prev - 1.0

    def cagr(self) -> float:
        """Compound annual growth rate = (last/first)^(1/(n-1)) - 1."""
        if self.population[0] == 0:
            raise ValueError("CAGR undefined when the first value is zero")
        return float((self.population[-1] / self.population[0]) ** (1.0 / (len(self) - 1)) - 1.0)


# --------------------------------------------------------------------------
# Task 4: forecasters
# --------------------------------------------------------------------------
def fibonacci(n: int) -> list[int]:
    """First n Fibonacci numbers: 1, 1, 2, 3, 5, ..."""
    if n < 0:
        raise ValueError("n must be non-negative")
    out, a, b = [], 1, 1
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


class Forecaster(ABC):
    """Abstract forecaster. ``fit`` learns from a consecutive yearly series,
    ``predict(horizon)`` forecasts the next ``horizon`` years."""

    name: str = "forecaster"

    def fit(self, years: Sequence[int], values: Sequence[float]) -> "Forecaster":
        yrs, vals = np.asarray(years, dtype=int), np.asarray(values, dtype=float)
        if yrs.shape != vals.shape or yrs.ndim != 1 or len(yrs) < 2:
            raise ValueError("years and values must be equal-length 1-D with >= 2 points")
        if np.any(np.diff(yrs) != 1):
            raise ValueError("years must be consecutive")
        self.years_, self.values_ = yrs, vals
        self._fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not hasattr(self, "years_"):
            raise RuntimeError("call fit() before predict()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        return self._predict(horizon)

    def forecast_years(self, horizon: int) -> np.ndarray:
        return self.years_[-1] + np.arange(1, horizon + 1)

    def fitted(self) -> np.ndarray:
        """In-sample fitted values (NaN where the model has no fit)."""
        return np.full(len(self.years_), np.nan)

    def _fit(self) -> None:
        """Estimation step (optional)."""

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray: ...

    def __repr__(self) -> str:
        return self.name


class LinearTrend(Forecaster):
    """Least-squares straight line, fitted with np.polyfit."""

    name = "Linear trend"

    def _fit(self) -> None:
        self.slope_, self.intercept_ = np.polyfit(self.years_ - self.years_[0], self.values_, 1)

    def fitted(self) -> np.ndarray:
        return self.slope_ * (self.years_ - self.years_[0]) + self.intercept_

    def _predict(self, horizon: int) -> np.ndarray:
        t = self.forecast_years(horizon) - self.years_[0]
        return self.slope_ * t + self.intercept_


class ExponentialGrowth(Forecaster):
    """Constant compound growth at the training-period CAGR.

    Forecast = last observed value x (1 + CAGR)^k.
    """

    name = "Exponential (CAGR)"

    def _fit(self) -> None:
        if self.values_[0] <= 0 or self.values_[-1] <= 0:
            raise ValueError("exponential model needs positive first and last values")
        self.g_ = (self.values_[-1] / self.values_[0]) ** (1.0 / (len(self.values_) - 1)) - 1.0

    def fitted(self) -> np.ndarray:
        return self.values_[0] * (1.0 + self.g_) ** (self.years_ - self.years_[0])

    def _predict(self, horizon: int) -> np.ndarray:
        return self.values_[-1] * (1.0 + self.g_) ** np.arange(1, horizon + 1)


class FibonacciRatio(Forecaster):
    """Scale the last value by successive Fibonacci ratios F(k+1)/F(k).

    With ``start=3`` the factors are 3/2, 5/3, 8/5, 13/8, ... (-> 1.618),
    applied cumulatively. The factors do not depend on the data at all.
    """

    name = "Fibonacci ratio"

    def __init__(self, start: int = 3) -> None:
        if start < 1:
            raise ValueError("start must be >= 1")
        self.start = start

    def ratios(self, horizon: int) -> np.ndarray:
        f = fibonacci(self.start + horizon)
        return np.array([f[k] / f[k - 1] for k in range(self.start, self.start + horizon)])

    def _predict(self, horizon: int) -> np.ndarray:
        return self.values_[-1] * np.cumprod(self.ratios(horizon))


# --------------------------------------------------------------------------
# Task 5: error metrics
# --------------------------------------------------------------------------
def _check(actual: Sequence[float], pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a, p = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    if a.shape != p.shape or a.size == 0:
        raise ValueError("actual and predicted must be non-empty and the same length")
    return a, p


def mae(actual, pred) -> float:
    a, p = _check(actual, pred)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, pred) -> float:
    a, p = _check(actual, pred)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual, pred) -> float:
    """Mean absolute percentage error, in percent."""
    a, p = _check(actual, pred)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((a - p) / a)) * 100.0)


def evaluate_models(district: DistrictPopulation, models: Sequence[Forecaster],
                    last_train_year: int = 2021) -> list[dict]:
    """Train each model on years <= last_train_year, test on the rest."""
    train, test = district.split(last_train_year)
    rows = []
    for m in models:
        m.fit(train.years, train.population)
        pred = m.predict(len(test))
        rows.append({"model": m.name, "MAE": mae(test.population, pred),
                     "RMSE": rmse(test.population, pred), "MAPE (%)": mape(test.population, pred)})
    return rows


# --------------------------------------------------------------------------
# Extension: bootstrap prediction intervals
# --------------------------------------------------------------------------
def bootstrap_prediction_intervals(model: Forecaster, years: Sequence[int],
                                   values: Sequence[float], horizon: int,
                                   n_boot: int = 2000, level: float = 0.95,
                                   seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual-bootstrap prediction intervals.

    1. Fit ``model`` and take in-sample residuals e = y - fitted.
    2. Repeat ``n_boot`` times: resample residuals with replacement, build
       y* = fitted + e*, refit a copy of the model, forecast, and add one
       resampled residual per horizon step (future noise).
    3. Return (lower, upper, median) percentile bands of the forecasts.
    """
    if n_boot < 1000:
        raise ValueError("use at least 1,000 bootstrap resamples")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    m = copy.deepcopy(model).fit(years, values)
    fit = m.fitted()
    if np.any(np.isnan(fit)):
        raise ValueError(f"{m.name} has no in-sample fit, so residuals are undefined")
    resid = np.asarray(values, dtype=float) - fit
    rng = np.random.default_rng(seed)
    sims = np.empty((n_boot, horizon))
    for b in range(n_boot):
        y_star = fit + rng.choice(resid, size=len(resid), replace=True)
        mb = copy.deepcopy(model).fit(years, y_star)
        sims[b] = mb.predict(horizon) + rng.choice(resid, size=horizon, replace=True)
    alpha = (1 - level) / 2
    return (np.quantile(sims, alpha, axis=0), np.quantile(sims, 1 - alpha, axis=0),
            np.quantile(sims, 0.5, axis=0))


# --------------------------------------------------------------------------
# Task 8: planning
# --------------------------------------------------------------------------
def classrooms_needed(population_thousands: float, school_age_share: float = 0.18,
                      pupils_per_classroom: int = 53) -> int:
    """Classrooms = ceil(population x share / pupils per classroom)."""
    if population_thousands < 0:
        raise ValueError("population cannot be negative")
    if not 0 <= school_age_share <= 1:
        raise ValueError("school_age_share must be between 0 and 1")
    if pupils_per_classroom < 1:
        raise ValueError("pupils_per_classroom must be >= 1")
    return math.ceil(population_thousands * 1000 * school_age_share / pupils_per_classroom)
