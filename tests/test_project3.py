"""Unit tests for Mini-Project 3 (run from the repo root: python -m pytest)."""
import numpy as np
import pytest

from src.fish_model import (
    FishStock,
    PriceModel,
    RiskAssessor,
    closed_season_mask,
    describe,
    fibonacci,
    weekly_revenue,
)


# ---------------- Fibonacci ----------------
def test_fibonacci_first_terms():
    assert fibonacci(8) == [1, 1, 2, 3, 5, 8, 13, 21]


def test_fibonacci_empty_and_invalid():
    assert fibonacci(0) == []
    with pytest.raises(ValueError):
        fibonacci(-1)


# ---------------- FishStock ----------------
def test_no_harvest_converges_to_carrying_capacity():
    stock, harvest = FishStock().simulate(weeks=200, h=0.0)
    assert stock[-1] == pytest.approx(10_000, rel=1e-3)
    assert harvest.sum() == 0.0


def test_matches_closed_form_equilibrium():
    fs = FishStock()
    stock, _ = fs.simulate(weeks=300, h=0.10)
    assert stock[-1] == pytest.approx(fs.equilibrium(0.10), rel=1e-3)  # 7,500 t


def test_msy_formula_and_h_msy_attains_it():
    fs = FishStock()
    assert fs.msy == 1_000.0
    assert fs.sustainable_yield(fs.h_msy) == pytest.approx(fs.msy)


def test_harvest_at_or_above_r_collapses():
    stock, _ = FishStock().simulate(weeks=200, h=0.5)
    assert stock[-1] < 1e-6


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        FishStock(r=-0.1)
    with pytest.raises(ValueError):
        FishStock(K=0)
    with pytest.raises(ValueError):
        FishStock().simulate(weeks=10, h=1.5)
    with pytest.raises(ValueError):
        FishStock().simulate(weeks=0, h=0.1)


def test_closed_season_has_zero_harvest_in_closed_weeks():
    mask = closed_season_mask(52, closed_weeks=8)
    _, harvest = FishStock().simulate(52, 0.2, closed=mask)
    assert mask.sum() == 8
    assert np.all(harvest[mask] == 0.0)
    assert np.all(harvest[~mask] > 0.0)


# ---------------- PriceModel ----------------
def test_price_bounds_start_and_shape():
    prices = PriceModel(sigma=2_000, seed=1).simulate(weeks=52, n_paths=200)
    assert prices.shape == (200, 52)
    assert np.all(prices[:, 0] == 12_000)
    assert prices.min() >= 9_000 and prices.max() <= 16_000


def test_price_is_reproducible_with_seed():
    a = PriceModel(seed=7).simulate(52, 5)
    b = PriceModel(seed=7).simulate(52, 5)
    assert np.array_equal(a, b)


def test_price_invalid_inputs():
    with pytest.raises(ValueError):
        PriceModel(start=20_000)
    with pytest.raises(ValueError):
        PriceModel().simulate(weeks=0)


# ---------------- Revenue, statistics, risk ----------------
def test_weekly_revenue_units():
    # 2 tonnes at 10,000 UGX/kg = 20,000,000 UGX
    assert weekly_revenue(np.array([2.0]), np.array([10_000.0]))[0] == 20_000_000


def test_cv_is_scale_free_but_variance_is_not():
    x = [10.0, 12.0, 14.0, 9.0]
    scaled = [v * 1_000 for v in x]
    assert describe(scaled)["cv"] == pytest.approx(describe(x)["cv"])
    assert describe(scaled)["variance"] == pytest.approx(describe(x)["variance"] * 1e6)


def test_describe_edge_cases():
    with pytest.raises(ValueError):
        describe([5.0])          # too few values
    with pytest.raises(ValueError):
        describe([-1.0, 1.0])    # zero mean -> CV undefined


def test_risk_classification_boundaries():
    ra = RiskAssessor()
    assert ra.classify(0.05) == "Low"
    assert ra.classify(0.15) == "Medium"
    assert ra.classify(0.30) == "High"
    with pytest.raises(ValueError):
        ra.classify(-0.1)


def test_value_at_risk_on_known_distribution():
    revenue = np.arange(1, 101, dtype=float)  # 1..100
    assert RiskAssessor.value_at_risk(revenue, 0.05) == pytest.approx(5.95)
    assert RiskAssessor.value_at_risk(np.full(50, 7.0)) == 7.0   # no risk -> VaR = value


def test_value_at_risk_invalid():
    with pytest.raises(ValueError):
        RiskAssessor.value_at_risk(np.array([]))
    with pytest.raises(ValueError):
        RiskAssessor.value_at_risk(np.array([1.0, 2.0]), alpha=1.5)


def test_run_scenario_structure_and_reproducibility():
    from src.fish_model import run_scenario

    args = dict(stock_model=FishStock(), h=0.1, assessor=RiskAssessor(), n_paths=200)
    a = run_scenario(price_model=PriceModel(seed=3), **args)
    b = run_scenario(price_model=PriceModel(seed=3), **args)
    assert a["total_revenue"].shape == (200,)
    assert np.array_equal(a["total_revenue"], b["total_revenue"])
    assert a["risk"] in {"Low", "Medium", "High"}
    assert a["var_quantile"] <= a["mean"]
