"""Unit tests for Mini-Project 5 (run from the repo root: python -m pytest)."""
import numpy as np
import pytest

from src.taxi_model import (
    ExponentialSmoothing,
    LinearTrend,
    MarketEquilibrium,
    MovingAverage,
    Route,
    SeasonalNaive,
    rolling_origin_backtest,
    simulate_weekly_demand,
    tune_alpha,
    vehicles_needed,
)

NTINDA = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]


# ---------------- Route ----------------
def test_route_revenue_and_statistics():
    r = Route("Ntinda", NTINDA, 2000)
    assert r.daily_revenue()[0] == 70_000
    assert r.total_revenue() == sum(NTINDA) * 2000
    d = r.describe()
    assert d["mean"] == pytest.approx(47.4)
    assert d["std"] == pytest.approx(d["variance"] ** 0.5)
    assert len(r) == 10


def test_route_validation():
    with pytest.raises(ValueError):
        Route("empty", [], 2000)
    with pytest.raises(ValueError):
        Route("neg", [10, -1], 2000)
    with pytest.raises(ValueError):
        Route("free", [10, 20], 0)
    with pytest.raises(ValueError):
        Route("one", [10], 2000).describe()      # variance needs 2+ points


# ---------------- Equilibrium ----------------
def test_equilibrium_matches_hand_calculation():
    # 120 - 0.02P = 10 + 0.03P  ->  P* = 110 / 0.05 = 2200,  Q* = 76
    q, p = MarketEquilibrium(120, 0.02, 10, 0.03).solve()
    assert p == pytest.approx(2200)
    assert q == pytest.approx(76)


def test_fare_below_equilibrium_means_shortage():
    m = MarketEquilibrium(120, 0.02, 10, 0.03)
    assert m.excess_demand(2000) == pytest.approx(10)      # 80 wanted vs 70 supplied
    assert m.excess_demand(2200) == pytest.approx(0)
    assert m.excess_demand(2500) < 0                        # surplus above equilibrium


def test_equilibrium_invalid_inputs():
    with pytest.raises(ValueError):
        MarketEquilibrium(120, -0.02, 10, 0.03)
    with pytest.raises(ValueError):
        MarketEquilibrium(10, 0.02, 120, 0.03).solve()      # negative P*: not meaningful


# ---------------- Forecasters ----------------
def test_moving_average_known_value():
    assert MovingAverage(3).fit([10, 20, 30, 40]).predict(1)[0] == pytest.approx(30.0)


def test_ses_alpha_one_is_naive_and_alpha_small_is_smooth():
    y = [10, 20, 30]
    assert ExponentialSmoothing(1.0).fit(y).predict(1)[0] == 30
    assert ExponentialSmoothing(0.5).fit(y).predict(1)[0] == pytest.approx(22.5)


def test_linear_trend_recovers_a_straight_line():
    assert np.allclose(LinearTrend().fit([2, 4, 6, 8]).predict(2), [10, 12])


def test_seasonal_naive_repeats_last_season():
    y = [1, 2, 3, 4, 5, 6, 7, 10, 20]
    assert np.allclose(SeasonalNaive(7).fit(y).predict(3), [3, 4, 5])


def test_forecaster_edge_cases():
    with pytest.raises(ValueError):
        MovingAverage(3).fit([1, 2])             # history too short
    with pytest.raises(ValueError):
        ExponentialSmoothing(0)
    with pytest.raises(RuntimeError):
        MovingAverage(3).predict(1)              # predict before fit
    assert LinearTrend().fit([5, 3, 1]).predict(5).min() == 0.0   # never negative


# ---------------- Backtesting ----------------
def test_rolling_origin_uses_only_past_data():
    # constant series: every model should forecast it perfectly
    res = rolling_origin_backtest(MovingAverage(3), [50] * 10, first_target_day=4)
    assert len(res["pred"]) == 7 and res["mae"] == 0.0


def test_backtest_mae_hand_check():
    # naive (alpha=1): predictions are the previous value
    res = rolling_origin_backtest(ExponentialSmoothing(1.0), [10, 20, 40], first_target_day=2)
    assert res["mae"] == pytest.approx((10 + 20) / 2)


def test_tune_alpha_and_invalid_backtest_start():
    best, maes = tune_alpha(NTINDA, 4)
    assert best == min(maes, key=maes.get)
    with pytest.raises(ValueError):
        rolling_origin_backtest(MovingAverage(3), NTINDA, first_target_day=1)
    with pytest.raises(ValueError):
        tune_alpha(NTINDA, 4, grid=[])


# ---------------- Fleet planning ----------------
def test_vehicles_needed_rounding_and_buffer():
    assert vehicles_needed(50) == 1                  # 57.5 / 112
    assert vehicles_needed(97) == 1                  # 111.55 / 112
    assert vehicles_needed(98) == 2                  # 112.7 / 112 -> 2
    assert vehicles_needed(0) == 0


def test_vehicles_needed_invalid():
    with pytest.raises(ValueError):
        vehicles_needed(-5)
    with pytest.raises(ValueError):
        vehicles_needed(50, seats=0)


# ---------------- Extension ----------------
def test_weekly_simulation_pattern_and_seed():
    a = simulate_weekly_demand(50, days=60, seed=1)
    assert np.array_equal(a, simulate_weekly_demand(50, days=60, seed=1))
    fridays, sundays = a[4::7], a[6::7]
    assert fridays.mean() > a.mean() > sundays.mean()
    with pytest.raises(ValueError):
        simulate_weekly_demand(0)


def test_seasonal_naive_beats_moving_average_on_weekly_data():
    y = simulate_weekly_demand(60, days=60, seed=3)
    sn = rolling_origin_backtest(SeasonalNaive(7), y, 15)["mae"]
    ma = rolling_origin_backtest(MovingAverage(3), y, 15)["mae"]
    assert sn < ma
