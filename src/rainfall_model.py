"""Rainfall pattern & crop suitability analyser (Mini-Project 4).

Classes
-------
Region    monthly rainfall for one region (mm) with summary statistics and
          rainy-season (peak) detection
CropRule  suitable monthly rainfall range for a crop; classifies months

Functions
---------
cosine_similarity, pearson_correlation, euclidean_distance, pairwise_matrix

Units: rainfall in mm per month, months Jan..Dec (index 0..11).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from scipy.signal import find_peaks

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------
# Task 1: Region
# --------------------------------------------------------------------------
class Region:
    """Monthly rainfall (12 values, mm) for one region."""

    def __init__(self, name: str, rainfall_mm: Sequence[float]) -> None:
        arr = np.asarray(rainfall_mm, dtype=float)
        if arr.shape != (12,):
            raise ValueError("rainfall must contain exactly 12 monthly values")
        if not np.all(np.isfinite(arr)):
            raise ValueError("rainfall must be finite numbers")
        if np.any(arr < 0):
            raise ValueError("rainfall cannot be negative")
        self.name = name
        self.rainfall = arr

    # -- summary statistics ----------------------------------------------
    @property
    def annual_total(self) -> float:
        return float(self.rainfall.sum())

    @property
    def mean(self) -> float:
        return float(self.rainfall.mean())

    @property
    def wettest_month(self) -> str:
        return MONTHS[int(np.argmax(self.rainfall))]

    @property
    def driest_month(self) -> str:
        return MONTHS[int(np.argmin(self.rainfall))]

    @property
    def cv(self) -> float:
        """Coefficient of variation of monthly rainfall (sample std / mean).

        A high CV means rainfall is concentrated in a few months (strong
        seasonality), not that it is unreliable from year to year.
        """
        if self.mean == 0:
            raise ValueError("CV is undefined for a region with zero rainfall")
        return float(self.rainfall.std(ddof=1) / self.mean)

    # -- rainy seasons ------------------------------------------------------
    def peaks(self, rel_prominence: float = 0.25) -> list[int]:
        """Indices (0-11) of distinct rainy-season peaks.

        The year is treated as circular (December is followed by January) by
        tiling the series three times and keeping peaks in the middle copy.
        A peak counts only if its prominence is at least
        ``rel_prominence`` x the region's wettest monthly rainfall.
        """
        if not 0 <= rel_prominence < 1:
            raise ValueError("rel_prominence must be in [0, 1)")
        top = float(self.rainfall.max())
        if top == 0:
            return []
        tiled = np.tile(self.rainfall, 3)
        idx, _ = find_peaks(tiled, prominence=max(rel_prominence * top, 1e-9))
        return [int(i - 12) for i in idx if 12 <= i < 24]

    def modality(self, rel_prominence: float = 0.25) -> str:
        """'unimodal' (one rainy season), 'bimodal' (two) or 'multimodal'."""
        n = len(self.peaks(rel_prominence))
        return {0: "no rainy season", 1: "unimodal", 2: "bimodal"}.get(n, "multimodal")

    def __len__(self) -> int:
        return len(self.rainfall)

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total:,.0f} mm)"


# --------------------------------------------------------------------------
# Task 2: CropRule
# --------------------------------------------------------------------------
class CropRule:
    """Suitable monthly rainfall range [low, high] (mm) for a crop.

    Below ``low`` -> "Drought risk"; above ``high`` -> "Waterlogging risk";
    otherwise "Good for <crop>".
    """

    def __init__(self, crop: str, low: float, high: float,
                 season_months: Optional[int] = None, source: str = "") -> None:
        if low < 0 or high <= low:
            raise ValueError("require 0 <= low < high")
        if season_months is not None and not 1 <= season_months <= 12:
            raise ValueError("season_months must be between 1 and 12 (or None)")
        self.crop, self.low, self.high = crop, float(low), float(high)
        self.season_months = season_months      # None = perennial crop
        self.source = source

    def classify(self, rain_mm: float) -> str:
        if rain_mm < 0:
            raise ValueError("rainfall cannot be negative")
        if rain_mm < self.low:
            return "Drought risk"
        if rain_mm > self.high:
            return "Waterlogging risk"
        return f"Good for {self.crop}"

    def code(self, rain_mm: float) -> int:
        """-1 drought, 0 good, +1 waterlogging (handy for heatmaps)."""
        label = self.classify(rain_mm)
        return -1 if label == "Drought risk" else (1 if label == "Waterlogging risk" else 0)

    def classify_region(self, region: Region) -> list[str]:
        return [self.classify(x) for x in region.rainfall]

    def codes(self, region: Region) -> np.ndarray:
        return np.array([self.code(x) for x in region.rainfall])

    def planting_window(self, region: Region) -> Optional[tuple[str, float]]:
        """Best start month for an annual crop.

        Scans every start month (wrapping past December) and scores the
        fraction of the ``season_months`` that are classed good. Ties are
        broken by the earliest start. Returns (start month, score) or None
        for perennial crops.
        """
        if self.season_months is None:
            return None
        good = (self.codes(region) == 0).astype(float)
        scores = [np.mean([good[(s + k) % 12] for k in range(self.season_months)])
                  for s in range(12)]
        best = int(np.argmax(scores))
        return MONTHS[best], float(scores[best])

    def __repr__(self) -> str:
        return f"CropRule({self.crop!r}, {self.low:g}-{self.high:g} mm/month)"


# --------------------------------------------------------------------------
# Tasks 3-4: similarity / distance measures
# --------------------------------------------------------------------------
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """cos(theta) = a.b / (|a| |b|): the cosine of the ANGLE between vectors.

    (Not ``math.cos`` of anything: that takes a scalar angle in radians.)
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D vectors of equal length")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(a, b) / (na * nb))


def pearson_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson r = cosine similarity of the mean-centred vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D vectors of equal length")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("correlation is undefined for a constant series")
    return cosine_similarity(a - a.mean(), b - b.mean())


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D vectors of equal length")
    return float(np.linalg.norm(a - b))


def pairwise_matrix(regions: Sequence[Region],
                    func: Callable[[Sequence[float], Sequence[float]], float]) -> np.ndarray:
    """Square matrix of ``func`` applied to every pair of regions."""
    n = len(regions)
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = func(regions[i].rainfall, regions[j].rainfall)
    return out
