"""Solar micro-grid dispatch planner (Mini-Project 2).

Daily demand constraints (x = solar, y = battery):
    3x + 2y = D1   (daytime load, kWh)
    4x +  y = D2   (critical-equipment load, kWh)

Classes
-------
MicroGrid          coefficient matrix, well-posedness checks, solvers
HybridMicroGrid    adds a diesel generator z and a third constraint (3x3)

Functions
---------
read_demand, generate_demand_csv, load_demand_csv, energy_cost,
monte_carlo_sensitivity
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.linalg import solve
from scipy.optimize import nnls

DEFAULT_A = np.array([[3.0, 2.0], [4.0, 1.0]])


# --------------------------------------------------------------------------
# Tasks 1, 3, 4: MicroGrid
# --------------------------------------------------------------------------
class MicroGrid:
    """Square linear system A @ s = d, where s are the energy sources."""

    source_names = ("solar", "battery")

    def __init__(self, A: Optional[Sequence[Sequence[float]]] = None) -> None:
        mat = np.array(DEFAULT_A if A is None else A, dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("coefficient matrix must be square")
        if not np.all(np.isfinite(mat)):
            raise ValueError("coefficient matrix must be finite")
        self.A = mat
        self.n = mat.shape[0]

    # -- Task 1: is the system well-posed? ---------------------------------
    @property
    def determinant(self) -> float:
        """det(A). Non-zero <=> the system has exactly one solution."""
        return float(np.linalg.det(self.A))

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.A))

    @property
    def condition_number(self) -> float:
        """2-norm condition number: the worst-case amplification of relative
        errors in the demands into relative errors in the solution."""
        return float(np.linalg.cond(self.A))

    @property
    def well_posed(self) -> bool:
        return self.rank == self.n

    def check_well_posed(self) -> dict:
        return {"determinant": self.determinant, "condition_number": self.condition_number,
                "rank": self.rank, "well_posed": self.well_posed}

    # -- demands -----------------------------------------------------------
    def _validate_demands(self, d) -> np.ndarray:
        arr = np.asarray(d, dtype=float)
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("demands must be finite and non-negative")
        return arr

    def solve_day(self, *demands: float) -> np.ndarray:
        """Exact solution for one day; may contain negative entries
        (physically infeasible)."""
        if len(demands) != self.n:
            raise ValueError(f"expected {self.n} demand values, got {len(demands)}")
        if not self.well_posed:
            raise ValueError("system is singular (linearly dependent equations): no unique solution")
        return solve(self.A, self._validate_demands(demands))

    # -- Task 3: loop vs vectorised ---------------------------------------
    def solve_loop(self, D: np.ndarray) -> np.ndarray:
        """Solve each row of D (n_days x n) with one scipy call per day."""
        D = self._validate_demands(D)
        return np.array([solve(self.A, row) for row in D])

    def solve_vectorised(self, D: np.ndarray) -> np.ndarray:
        """Solve all days in ONE call: right-hand side is n x n_days."""
        D = self._validate_demands(D)
        return solve(self.A, D.T).T

    # -- Task 4: infeasible days -------------------------------------------
    @staticmethod
    def infeasible_mask(X: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """True for days where any source is negative."""
        return np.any(np.asarray(X) < -tol, axis=1)

    def solve_nonneg(self, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Non-negative least squares per day (scipy.optimize.nnls).

        Returns (X, residuals): X >= 0 minimises ||A x - d||; residuals[i] is
        the leftover vector A x - d (zero for feasible days).
        """
        D = self._validate_demands(D)
        X, R = [], []
        for row in D:
            x, _ = nnls(self.A, row)
            X.append(x)
            R.append(self.A @ x - row)
        return np.array(X), np.array(R)

    def solve_clipped(self, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact solution with negative entries clipped to zero; returns
        (X, residuals) for comparison with ``solve_nonneg``."""
        D = self._validate_demands(D)
        X = np.clip(self.solve_vectorised(D), 0.0, None)
        return X, X @ self.A.T - D

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={self.n}, det={self.determinant:.3g}, cond={self.condition_number:.3g})"


class HybridMicroGrid(MicroGrid):
    """Adds a diesel generator z and a third equation (3x3 system).

    Default equations (x solar, y battery, z diesel):
        3x + 2y + z = D1       daytime load
        4x +  y + z = D2       critical-equipment load
         x +  y + z = D3       total energy dispatched (energy balance)
    """

    source_names = ("solar", "battery", "diesel")
    DEFAULT_A3 = np.array([[3.0, 2.0, 1.0], [4.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def __init__(self, A: Optional[Sequence[Sequence[float]]] = None) -> None:
        super().__init__(self.DEFAULT_A3 if A is None else A)
        if self.n != 3:
            raise ValueError("HybridMicroGrid needs a 3x3 matrix")

    def solve_least_squares(self, d: Sequence[float]) -> np.ndarray:
        """Minimum-norm least-squares solution via np.linalg.lstsq. Works for
        singular systems (picks one solution out of infinitely many, or the
        best compromise when the equations are inconsistent)."""
        return np.linalg.lstsq(self.A, self._validate_demands(d), rcond=None)[0]


# --------------------------------------------------------------------------
# Task 2: input modes
# --------------------------------------------------------------------------
def read_demand(prompt: str, input_fn: Callable[[str], str] = input,
                print_fn: Callable[[str], None] = print,
                max_attempts: Optional[int] = None) -> float:
    """Ask until a valid demand is entered.

    Rejects empty, non-numeric, non-finite and negative values and re-prompts.
    ``input_fn``/``print_fn`` are injectable so the function can be tested.
    """
    attempts = 0
    while True:
        attempts += 1
        raw = input_fn(prompt).strip()
        try:
            if not raw:
                raise ValueError("value is empty")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("value must be a finite number")
            if value < 0:
                raise ValueError("demand cannot be negative")
            return value
        except ValueError as exc:
            msg = str(exc) if str(exc).startswith(("value", "demand")) else f"{raw!r} is not a number"
            print_fn(f"Invalid input: {msg}. Please try again.")
            if max_attempts is not None and attempts >= max_attempts:
                raise ValueError(f"no valid input after {attempts} attempts") from exc


# Illustrative weekly means, Monday..Sunday (kWh). Weekday daytime load is
# high relative to the critical load; weekends are quieter.
D1_WEEK = np.array([57.0, 57.0, 56.0, 57.0, 56.0, 38.0, 35.0])
D2_WEEK = np.array([30.0, 30.0, 29.5, 30.0, 29.5, 30.0, 29.5])


def generate_demand_csv(path: str | Path, days: int = 30, noise_cv: float = 0.05,
                        seed: int = 2026) -> np.ndarray:
    """Write ``days`` of synthetic demand to CSV and return the array (days x 2).

    Columns: day, weekday, d1_kwh, d2_kwh. Weekly pattern + multiplicative
    Gaussian noise; day 1 is a Monday. Fully determined by ``seed``.
    """
    if days < 1 or noise_cv < 0:
        raise ValueError("days >= 1 and noise_cv >= 0 required")
    rng = np.random.default_rng(seed)
    dow = np.arange(days) % 7
    d1 = np.round(D1_WEEK[dow] * (1 + rng.normal(0, noise_cv, days)), 2)
    d2 = np.round(D2_WEEK[dow] * (1 + rng.normal(0, noise_cv, days)), 2)
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day", "weekday", "d1_kwh", "d2_kwh"])
        for i in range(days):
            w.writerow([i + 1, names[dow[i]], d1[i], d2[i]])
    return np.column_stack([d1, d2])


def load_demand_csv(path: str | Path) -> np.ndarray:
    """Read a demand CSV into an array (days x 2) of [d1, d2], validating it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"d1_kwh", "d2_kwh"} <= set(reader.fieldnames):
            raise ValueError("CSV must have columns d1_kwh and d2_kwh")
        for line_no, row in enumerate(reader, start=2):
            try:
                d1, d2 = float(row["d1_kwh"]), float(row["d2_kwh"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_no}: non-numeric demand") from exc
            if not (math.isfinite(d1) and math.isfinite(d2)) or d1 < 0 or d2 < 0:
                raise ValueError(f"line {line_no}: demands must be finite and non-negative")
            rows.append([d1, d2])
    if not rows:
        raise ValueError("CSV contains no data rows")
    return np.array(rows)


# --------------------------------------------------------------------------
# Task 6: cost
# --------------------------------------------------------------------------
def energy_cost(X: np.ndarray, prices: Sequence[float] = (150.0, 450.0)) -> np.ndarray:
    """Daily cost (UGX) = sum over sources of price (UGX/kWh) x energy (kWh)."""
    X, p = np.atleast_2d(np.asarray(X, dtype=float)), np.asarray(prices, dtype=float)
    if X.shape[1] != len(p):
        raise ValueError("one price per source required")
    if np.any(p < 0):
        raise ValueError("prices must be non-negative")
    return X @ p


# --------------------------------------------------------------------------
# Extension: Monte Carlo sensitivity
# --------------------------------------------------------------------------
def monte_carlo_sensitivity(grid: MicroGrid, d: Sequence[float], pct: float = 0.05,
                            n_draws: int = 1000, seed: int = 42) -> dict:
    """Perturb every demand independently by Uniform(-pct, +pct) and re-solve.

    Returns the solutions, the relative change of each source, and the
    amplification ratio (relative change of the solution / relative change of
    the demands, both in the 2-norm) for each draw. For any draw the ratio
    cannot exceed the condition number of A.
    """
    d = np.asarray(d, dtype=float)
    if pct <= 0 or n_draws < 1:
        raise ValueError("pct > 0 and n_draws >= 1 required")
    rng = np.random.default_rng(seed)
    base = grid.solve_day(*d)
    D = d * (1 + rng.uniform(-pct, pct, size=(n_draws, len(d))))
    X = grid.solve_vectorised(D)
    rel_in = np.linalg.norm(D - d, axis=1) / np.linalg.norm(d)
    rel_out = np.linalg.norm(X - base, axis=1) / np.linalg.norm(base)
    return {"base": base, "X": X, "rel_change_each": (X - base) / base,
            "amplification": rel_out / rel_in, "cond": grid.condition_number}
