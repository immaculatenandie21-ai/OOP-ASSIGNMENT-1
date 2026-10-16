"""Unit tests for Mini-Project 1 (run from the repo root: python -m pytest)."""
import numpy as np
import pytest

from src.population_model import (
    DistrictPopulation,
    ExponentialGrowth,
    FibonacciRatio,
    LinearTrend,
    bootstrap_prediction_intervals,
    classrooms_needed,
    evaluate_models,
    mae,
    mape,
    rmse,
)

YEARS = np.arange(2015, 2025)
KAMPALA = [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800]


# ---------------- DistrictPopulation ----------------
def test_district_basics_repr_len_and_split():
    d = DistrictPopulation("Kampala", YEARS, KAMPALA)
    assert len(d) == 10
    assert "Kampala" in repr(d) and "2015-2024" in repr(d)
    train, test = d.split(2021)
    assert len(train) == 7 and len(test) == 3 and test.years[0] == 2022


def test_district_validation():
    with pytest.raises(ValueError):
        DistrictPopulation("bad", [2015, 2016], [100.0])               # unequal lengths
    with pytest.raises(ValueError):
        DistrictPopulation("neg", [2015, 2016], [100.0, -5.0])         # negative value
    with pytest.raises(ValueError):
        DistrictPopulation("gap", [2015, 2017], [100.0, 120.0])        # non-consecutive years
    with pytest.raises(ValueError):
        DistrictPopulation("one", [2015], [100.0])                     # too short


def test_variance_ddof_relationship():
    d = DistrictPopulation("Kampala", YEARS, KAMPALA)
    stat, np0, np1 = d.describe_statistics(), d.describe_numpy(0), d.describe_numpy(1)
    assert stat["variance"] == pytest.approx(np1["variance"])          # statistics.variance == ddof=1
    assert stat["variance"] > np0["variance"]                          # n-1 divisor is larger
    assert np0["variance"] * 10 / 9 == pytest.approx(np1["variance"])


def test_growth_and_cagr_hand_calculation():
    d = DistrictPopulation("x", [2020, 2021, 2022], [100.0, 110.0, 121.0])
    assert np.allclose(d.yoy_growth(), [0.10, 0.10])
    assert d.cagr() == pytest.approx(0.10)


def test_growth_edge_case_zero_population():
    d = DistrictPopulation("z", [2020, 2021], [0.0, 10.0])
    with pytest.raises(ValueError):
        d.cagr()
    with pytest.raises(ValueError):
        d.yoy_growth()


# ---------------- Forecasters ----------------
def test_linear_trend_recovers_line():
    m = LinearTrend().fit([2020, 2021, 2022, 2023], [10, 20, 30, 40])
    assert np.allclose(m.predict(2), [50, 60])
    assert np.allclose(m.fitted(), [10, 20, 30, 40])


def test_exponential_growth_doubling():
    m = ExponentialGrowth().fit([2020, 2021, 2022], [100, 200, 400])
    assert m.g_ == pytest.approx(1.0)
    assert np.allclose(m.predict(2), [800, 1600])


def test_fibonacci_ratio_factors():
    m = FibonacciRatio().fit([2020, 2021], [100, 100])
    assert np.allclose(m.ratios(3), [3 / 2, 5 / 3, 8 / 5])
    assert m.predict(1)[0] == pytest.approx(150)
    assert m.predict(3)[2] == pytest.approx(100 * 3 / 2 * 5 / 3 * 8 / 5)
    assert np.all(np.isnan(m.fitted()))


def test_forecaster_edge_cases():
    with pytest.raises(RuntimeError):
        LinearTrend().predict(3)                       # predict before fit
    with pytest.raises(ValueError):
        LinearTrend().fit([2020], [1.0])               # too few points
    with pytest.raises(ValueError):
        LinearTrend().fit([2020, 2021], [1.0, 2.0]).predict(0)


# ---------------- Metrics ----------------
def test_metrics_hand_values():
    a, p = [100, 200], [110, 180]
    assert mae(a, p) == pytest.approx(15.0)
    assert rmse(a, p) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert mape(a, p) == pytest.approx((10 + 10) / 2)


def test_metrics_edge_cases():
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])
    with pytest.raises(ValueError):
        mae([1, 2], [1])
    assert rmse([5, 5], [5, 5]) == 0.0


def test_evaluate_models_selects_exponential_on_exponential_data():
    d = DistrictPopulation("exp", np.arange(2015, 2025), 100 * 1.05 ** np.arange(10))
    rows = evaluate_models(d, [LinearTrend(), ExponentialGrowth(), FibonacciRatio()])
    best = min(rows, key=lambda r: r["RMSE"])
    assert best["model"] == "Exponential (CAGR)"
    assert best["RMSE"] < 1e-9


# ---------------- Bootstrap ----------------
def test_bootstrap_intervals_order_and_reproducibility():
    kw = dict(model=LinearTrend(), years=YEARS, values=KAMPALA, horizon=5, n_boot=1000, seed=7)
    lo, hi, med = bootstrap_prediction_intervals(**kw)
    lo2, hi2, _ = bootstrap_prediction_intervals(**kw)
    assert np.all(lo <= med) and np.all(med <= hi)
    assert np.array_equal(lo, lo2) and np.array_equal(hi, hi2)


def test_bootstrap_invalid_inputs():
    with pytest.raises(ValueError):
        bootstrap_prediction_intervals(LinearTrend(), YEARS, KAMPALA, 5, n_boot=10)
    with pytest.raises(ValueError):
        bootstrap_prediction_intervals(FibonacciRatio(), YEARS, KAMPALA, 5)   # no fitted values


# ---------------- Classrooms ----------------
def test_classrooms_needed_rounds_up():
    assert classrooms_needed(1.0) == 4            # 180 pupils / 53 = 3.4 -> 4
    assert classrooms_needed(53 / 0.18 / 1000) == 1
    assert classrooms_needed(0) == 0


def test_classrooms_invalid_inputs():
    with pytest.raises(ValueError):
        classrooms_needed(-1)
    with pytest.raises(ValueError):
        classrooms_needed(100, school_age_share=1.5)
    with pytest.raises(ValueError):
        classrooms_needed(100, pupils_per_classroom=0)
