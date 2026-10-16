"""Taxi route revenue, pricing & fleet planner (Mini-Project 5).

Classes
-------
Route               daily passenger counts + fare, revenue and statistics
MarketEquilibrium   linear demand/supply system solved with scipy.linalg
Forecaster (ABC)    fit(y) / predict(horizon) interface
  MovingAverage, ExponentialSmoothing, LinearTrend, SeasonalNaive

Functions
---------
rolling_origin_backtest, tune_alpha, vehicles_needed, simulate_weekly_demand

Units: passengers per day, fares in UGX, revenue in UGX.
"""
from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import solve


# --------------------------------------------------------------------------
# Task 1: Route
# --------------------------------------------------------------------------
class Route:
    """A taxi route: daily passenger counts (NumPy array) and a fixed fare."""

    def __init__(self, name: str, passengers: Sequence[float], fare: float) -> None:
        arr = np.asarray(passengers, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("passengers must be a non-empty 1-D sequence")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("passenger counts must be finite and non-negative")
        if fare <= 0:
            raise ValueError("fare must be positive")
        self.name, self.passengers, self.fare = name, arr, float(fare)

    def daily_revenue(self) -> np.ndarray:
        """Revenue per day in UGX = passengers x fare."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def describe(self) -> dict[str, float]:
        """Mean, sample variance and standard deviation of passengers
        (``statistics`` module)."""
        values = self.passengers.tolist()
        if len(values) < 2:
            raise ValueError("need at least two days for variance")
        return {
            "mean": statistics.mean(values),
            "variance": statistics.variance(values),
            "std": statistics.stdev(values),
        }

    def __len__(self) -> int:
        return len(self.passengers)

    def __repr__(self) -> str:
        return f"Route({self.name!r}, {len(self)} days, fare UGX {self.fare:,.0f})"


# --------------------------------------------------------------------------
# Task 2: supply-demand equilibrium
# --------------------------------------------------------------------------
class MarketEquilibrium:
    """Demand Qd = a_d - b_d*P and supply Qs = a_s + b_s*P.

    Written as a 2x2 linear system in (Q, P):
        Q + b_d*P = a_d
        Q - b_s*P = a_s
    """

    def __init__(self, a_d: float, b_d: float, a_s: float, b_s: float) -> None:
        if b_d <= 0 or b_s <= 0:
            raise ValueError("b_d and b_s must be positive (demand falls, supply rises with price)")
        self.a_d, self.b_d, self.a_s, self.b_s = a_d, b_d, a_s, b_s

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[1.0, self.b_d], [1.0, -self.b_s]])

    @property
    def rhs(self) -> np.ndarray:
        return np.array([self.a_d, self.a_s])

    def solve(self) -> tuple[float, float]:
        """Return (equilibrium quantity Q*, equilibrium price P*)."""
        q, p = solve(self.matrix, self.rhs)
        if p <= 0 or q <= 0:
            raise ValueError("equilibrium is not economically meaningful (P* or Q* <= 0)")
        return float(q), float(p)

    def demand(self, price: float) -> float:
        return self.a_d - self.b_d * price

    def supply(self, price: float) -> float:
        return self.a_s + self.b_s * price

    def excess_demand(self, price: float) -> float:
        """Qd - Qs at ``price``: positive = shortage, negative = surplus."""
        return self.demand(price) - self.supply(price)


# --------------------------------------------------------------------------
# Task 3: forecasters
# --------------------------------------------------------------------------
class Forecaster(ABC):
    """Abstract forecaster: ``fit`` on a history, then ``predict`` ahead."""

    name: str = "forecaster"

    def fit(self, y: Sequence[float]) -> "Forecaster":
        arr = np.asarray(y, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("history must be a non-empty 1-D sequence")
        self._validate_length(arr)
        self._y = arr
        self._fit(arr)
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        if not hasattr(self, "_y"):
            raise RuntimeError("call fit() before predict()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        return np.maximum(self._predict(horizon), 0.0)   # counts cannot be negative

    def _validate_length(self, y: np.ndarray) -> None:
        """Subclasses raise ValueError if the history is too short."""

    def _fit(self, y: np.ndarray) -> None:
        """Optional estimation step."""

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray: ...

    def __repr__(self) -> str:
        return self.name


class MovingAverage(Forecaster):
    """Flat forecast = mean of the last ``window`` observations."""

    def __init__(self, window: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.name = f"{window}-day moving average"

    def _validate_length(self, y: np.ndarray) -> None:
        if len(y) < self.window:
            raise ValueError(f"need at least {self.window} observations")

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._y[-self.window:].mean())


class ExponentialSmoothing(Forecaster):
    """Simple exponential smoothing: l_t = a*y_t + (1-a)*l_{t-1}, l_0 = y_0.
    The forecast is flat at the final level."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.name = f"SES (alpha={alpha:.1f})"

    def _fit(self, y: np.ndarray) -> None:
        level = y[0]
        for v in y[1:]:
            level = self.alpha * v + (1 - self.alpha) * level
        self.level_ = float(level)

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level_)


class LinearTrend(Forecaster):
    """Least-squares straight line through the history (np.polyfit)."""

    name = "linear trend"

    def _validate_length(self, y: np.ndarray) -> None:
        if len(y) < 2:
            raise ValueError("need at least 2 observations")

    def _fit(self, y: np.ndarray) -> None:
        self.slope_, self.intercept_ = np.polyfit(np.arange(len(y)), y, 1)

    def _predict(self, horizon: int) -> np.ndarray:
        t = np.arange(len(self._y), len(self._y) + horizon)
        return self.slope_ * t + self.intercept_


class SeasonalNaive(Forecaster):
    """Forecast = the value observed one season (``period`` days) earlier."""

    def __init__(self, period: int = 7) -> None:
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.name = f"seasonal naive (period {period})"

    def _validate_length(self, y: np.ndarray) -> None:
        if len(y) < self.period:
            raise ValueError(f"need at least one full period ({self.period} observations)")

    def _predict(self, horizon: int) -> np.ndarray:
        n = len(self._y)
        return np.array([self._y[n - self.period + (h % self.period)] for h in range(horizon)])


# --------------------------------------------------------------------------
# Task 4: backtesting and tuning
# --------------------------------------------------------------------------
def rolling_origin_backtest(forecaster: Forecaster, y: Sequence[float],
                            first_target_day: int) -> dict:
    """Walk-forward one-step-ahead backtest.

    ``first_target_day`` is 1-based: for days 4-10 pass 4. For each target
    day t, the model is fitted on days 1..t-1 only and predicts day t.
    Returns predictions, actuals and MAE (in the units of ``y``).
    """
    y = np.asarray(y, dtype=float)
    if not 2 <= first_target_day <= len(y):
        raise ValueError("first_target_day must be between 2 and len(y)")
    preds, actual = [], []
    for t in range(first_target_day - 1, len(y)):
        preds.append(forecaster.fit(y[:t]).predict(1)[0])
        actual.append(y[t])
    preds, actual = np.array(preds), np.array(actual)
    return {"pred": preds, "actual": actual, "mae": float(np.mean(np.abs(preds - actual)))}


def tune_alpha(y: Sequence[float], first_target_day: int,
               grid: Optional[Sequence[float]] = None) -> tuple[float, dict[float, float]]:
    """Grid-search alpha for ExponentialSmoothing by backtest MAE.

    Returns (best alpha, {alpha: MAE}). Ties go to the smaller alpha.
    """
    grid = list(np.round(np.arange(0.1, 1.01, 0.1), 2)) if grid is None else list(grid)
    if not grid:
        raise ValueError("grid is empty")
    maes = {float(a): rolling_origin_backtest(ExponentialSmoothing(a), y, first_target_day)["mae"]
            for a in grid}
    best = min(maes, key=lambda a: (maes[a], a))
    return best, maes


# --------------------------------------------------------------------------
# Task 6: fleet planning
# --------------------------------------------------------------------------
def vehicles_needed(passengers: float, trips_per_day: int = 8,
                    seats: int = 14, buffer: float = 0.15) -> int:
    """Vehicles required = ceil(passengers x (1 + buffer) / capacity per vehicle)
    where capacity = trips_per_day x seats (passengers per vehicle per day).
    At least one vehicle is always returned for positive demand."""
    if passengers < 0 or buffer < 0:
        raise ValueError("passengers and buffer must be non-negative")
    if trips_per_day < 1 or seats < 1:
        raise ValueError("trips_per_day and seats must be >= 1")
    if passengers == 0:
        return 0
    return max(1, math.ceil(passengers * (1 + buffer) / (trips_per_day * seats)))


# --------------------------------------------------------------------------
# Extension: weekly-pattern simulation
# --------------------------------------------------------------------------
# Multipliers Mon..Sun: busier Friday, quieter Sunday
WEEKLY_PATTERN = np.array([1.00, 0.97, 1.00, 1.05, 1.30, 1.10, 0.65])


def simulate_weekly_demand(base: float, days: int = 60, noise_cv: float = 0.05,
                           seed: int = 42, pattern: np.ndarray = WEEKLY_PATTERN) -> np.ndarray:
    """Passengers/day = base x weekly multiplier x (1 + Normal(0, noise_cv)),
    rounded to whole passengers. Day 1 is a Monday."""
    if base <= 0 or days < 1 or noise_cv < 0:
        raise ValueError("base > 0, days >= 1 and noise_cv >= 0 required")
    rng = np.random.default_rng(seed)
    mult = pattern[np.arange(days) % 7]
    y = base * mult * (1 + rng.normal(0, noise_cv, size=days))
    return np.maximum(np.round(y), 0.0)
