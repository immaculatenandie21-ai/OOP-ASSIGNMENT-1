"""Lake Victoria fish stock & export risk model (Mini-Project 3).

Classes
-------
FishStock     discrete logistic growth with proportional harvesting
PriceModel    seeded, bounded random-walk price (UGX/kg)
RiskAssessor  CV-based risk classification and Monte Carlo Value-at-Risk

Units used throughout: stock and harvest in tonnes, price in UGX/kg,
revenue in UGX, time step = 1 week (so r and h are per-week rates).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

KG_PER_TONNE = 1_000


# --------------------------------------------------------------------------
# Task 1: Fibonacci baseline
# --------------------------------------------------------------------------
def fibonacci(n: int) -> list[int]:
    """Return the first ``n`` Fibonacci numbers (1, 1, 2, 3, 5, ...)."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    seq: list[int] = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


# --------------------------------------------------------------------------
# Task 2: FishStock
# --------------------------------------------------------------------------
def closed_season_mask(weeks: int, closed_weeks: int = 8, start_week: int = 0) -> np.ndarray:
    """Boolean mask (True = fishing CLOSED) repeating every 52 weeks.

    The closure covers ``closed_weeks`` consecutive weeks of each year,
    beginning at ``start_week`` (0-based) of the year.
    """
    if not 0 <= closed_weeks <= 52:
        raise ValueError("closed_weeks must be between 0 and 52")
    week_of_year = np.arange(weeks) % 52
    return (week_of_year >= start_week) & (week_of_year < start_week + closed_weeks)


@dataclass
class FishStock:
    """Discrete logistic growth with proportional harvesting.

    N(t+1) = N(t) + r*N(t)*(1 - N(t)/K) - h*N(t)

    Attributes
    ----------
    r : intrinsic growth rate per week (> 0)
    K : carrying capacity in tonnes (> 0)
    n0 : initial stock in tonnes (>= 0)
    """

    r: float = 0.4
    K: float = 10_000.0
    n0: float = 4_000.0

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("r must be positive")
        if self.K <= 0:
            raise ValueError("K must be positive")
        if self.n0 < 0:
            raise ValueError("n0 must be non-negative")

    # -- theory -----------------------------------------------------------
    @property
    def msy(self) -> float:
        """Theoretical maximum sustainable yield rK/4 (tonnes per week)."""
        return self.r * self.K / 4.0

    @property
    def h_msy(self) -> float:
        """Harvest rate that attains MSY (= r/2)."""
        return self.r / 2.0

    def equilibrium(self, h: float) -> float:
        """Non-trivial equilibrium stock K(1 - h/r); 0 if h >= r (collapse)."""
        self._check_h(h)
        return max(0.0, self.K * (1.0 - h / self.r))

    def sustainable_yield(self, h: float) -> float:
        """Equilibrium harvest h*N* in tonnes per week."""
        return h * self.equilibrium(h)

    # -- simulation -------------------------------------------------------
    def simulate(
        self,
        weeks: int,
        h: float,
        closed: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate ``weeks`` steps.

        Parameters
        ----------
        weeks : number of weekly steps (>= 1)
        h : weekly harvest rate (fraction of stock removed)
        closed : optional boolean mask of length ``weeks``; True = no fishing

        Returns
        -------
        stock : array of length weeks + 1 (stock[0] = n0)
        harvest : array of length weeks, tonnes caught in each week
        """
        if weeks < 1:
            raise ValueError("weeks must be >= 1")
        self._check_h(h)
        if closed is None:
            closed = np.zeros(weeks, dtype=bool)
        closed = np.asarray(closed, dtype=bool)
        if closed.shape != (weeks,):
            raise ValueError("closed mask must have length `weeks`")

        stock = np.empty(weeks + 1)
        harvest = np.empty(weeks)
        stock[0] = self.n0
        for t in range(weeks):
            n = stock[t]
            h_t = 0.0 if closed[t] else h
            harvest[t] = h_t * n
            stock[t + 1] = max(0.0, n + self.r * n * (1.0 - n / self.K) - harvest[t])
        return stock, harvest

    @staticmethod
    def _check_h(h: float) -> None:
        if not 0.0 <= h <= 1.0:
            raise ValueError("harvest rate h must be in [0, 1]")

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K:,.0f} t, n0={self.n0:,.0f} t)"


# --------------------------------------------------------------------------
# Task 3: PriceModel
# --------------------------------------------------------------------------
class PriceModel:
    """Bounded Gaussian random walk for the weekly price in UGX/kg.

    p[t+1] = clip(p[t] + N(0, sigma), low, high); p[0] = start.
    Clipping acts as a reflecting-style barrier, keeping prices in a
    plausible market band.
    """

    def __init__(
        self,
        start: float = 12_000.0,
        sigma: float = 350.0,
        low: float = 9_000.0,
        high: float = 16_000.0,
        seed: int = 42,
    ) -> None:
        if not low <= start <= high:
            raise ValueError("start must lie within [low, high]")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.start, self.sigma, self.low, self.high = start, sigma, low, high
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def simulate(self, weeks: int = 52, n_paths: int = 1) -> np.ndarray:
        """Return prices with shape (n_paths, weeks); column 0 is ``start``."""
        if weeks < 1 or n_paths < 1:
            raise ValueError("weeks and n_paths must be >= 1")
        prices = np.empty((n_paths, weeks))
        prices[:, 0] = self.start
        steps = self.rng.normal(0.0, self.sigma, size=(n_paths, weeks - 1))
        for t in range(1, weeks):
            prices[:, t] = np.clip(prices[:, t - 1] + steps[:, t - 1], self.low, self.high)
        return prices


def weekly_revenue(harvest_tonnes: np.ndarray, price_ugx_per_kg: np.ndarray) -> np.ndarray:
    """Revenue (UGX) = harvest (kg) x price (UGX/kg). Broadcasts over paths."""
    return np.asarray(harvest_tonnes) * KG_PER_TONNE * np.asarray(price_ugx_per_kg)


# --------------------------------------------------------------------------
# Tasks 4-5: descriptive stats, CV, risk, VaR
# --------------------------------------------------------------------------
def describe(values: Sequence[float]) -> dict[str, float]:
    """Mean, median, sample variance, st. dev. and CV using ``statistics``."""
    values = list(values)
    if len(values) < 2:
        raise ValueError("need at least two values")
    mean = statistics.mean(values)
    if mean == 0:
        raise ValueError("CV is undefined when the mean is zero")
    sd = statistics.stdev(values)
    return {
        "mean": mean,
        "median": statistics.median(values),
        "variance": statistics.variance(values),
        "std": sd,
        "cv": sd / mean,
    }


class RiskAssessor:
    """Classify revenue risk by coefficient of variation and compute VaR.

    Rule (unit-free, so comparable across scenarios):
        CV <  low_cut           -> "Low"
        low_cut <= CV < high_cut -> "Medium"
        CV >= high_cut          -> "High"
    """

    def __init__(self, low_cut: float = 0.15, high_cut: float = 0.30) -> None:
        if not 0 < low_cut < high_cut:
            raise ValueError("require 0 < low_cut < high_cut")
        self.low_cut, self.high_cut = low_cut, high_cut

    def classify(self, cv: float) -> str:
        if cv < 0:
            raise ValueError("CV cannot be negative")
        if cv < self.low_cut:
            return "Low"
        if cv < self.high_cut:
            return "Medium"
        return "High"

    @staticmethod
    def cv(values: Sequence[float]) -> float:
        return describe(values)["cv"]

    @staticmethod
    def value_at_risk(annual_revenue: np.ndarray, alpha: float = 0.05) -> float:
        """alpha-quantile of annual revenue: with probability 1-alpha the
        cooperative earns at least this much."""
        arr = np.asarray(annual_revenue, dtype=float)
        if arr.size == 0:
            raise ValueError("annual_revenue is empty")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        return float(np.quantile(arr, alpha))

    def assess(self, annual_revenue: np.ndarray, alpha: float = 0.05) -> dict[str, float | str]:
        """CV-based class plus VaR for a vector of simulated annual revenues."""
        stats = describe(annual_revenue)
        var_q = self.value_at_risk(annual_revenue, alpha)
        return {
            "mean": stats["mean"],
            "cv": stats["cv"],
            "risk": self.classify(stats["cv"]),
            "var_quantile": var_q,
            "shortfall_vs_mean": stats["mean"] - var_q,
        }


# --------------------------------------------------------------------------
# Task 6: scenario runner (composition of the three classes)
# --------------------------------------------------------------------------
def run_scenario(
    stock_model: FishStock,
    h: float,
    price_model: PriceModel,
    assessor: RiskAssessor,
    weeks: int = 52,
    n_paths: int = 5_000,
    closed: Optional[np.ndarray] = None,
) -> dict:
    """Simulate stock once and price ``n_paths`` times for one harvest rate.

    The stock path is deterministic (only price is random), so every price
    path is multiplied by the same weekly harvest vector.

    Returns a dict with the stock path, harvest, simulated total revenue
    per path (UGX) and the RiskAssessor summary.
    """
    stock, harvest = stock_model.simulate(weeks, h, closed)
    prices = price_model.simulate(weeks, n_paths)
    total_revenue = weekly_revenue(harvest, prices).sum(axis=1)
    summary = assessor.assess(total_revenue)
    return {
        "h": h,
        "stock": stock,
        "harvest": harvest,
        "total_revenue": total_revenue,
        **summary,
    }
