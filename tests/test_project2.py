"""Unit tests for Mini-Project 2 (run from the repo root: python -m pytest)."""
import numpy as np
import pytest

from src.microgrid_model import (
    HybridMicroGrid,
    MicroGrid,
    energy_cost,
    generate_demand_csv,
    load_demand_csv,
    monte_carlo_sensitivity,
    read_demand,
)


# ---------------- MicroGrid: well-posedness and solving ----------------
def test_determinant_condition_and_rank():
    g = MicroGrid()
    assert g.determinant == pytest.approx(-5.0)       # 3*1 - 2*4
    assert g.rank == 2 and g.well_posed
    assert g.condition_number == pytest.approx(np.linalg.cond(g.A))
    assert g.condition_number > 1


def test_solve_day_hand_calculation():
    # x = 2, y = 3  ->  D1 = 3*2 + 2*3 = 12,  D2 = 4*2 + 3 = 11
    assert np.allclose(MicroGrid().solve_day(12, 11), [2.0, 3.0])
    # closed form: x = (2 D2 - D1)/5, y = (4 D1 - 3 D2)/5
    d1, d2 = 50.0, 40.0
    assert np.allclose(MicroGrid().solve_day(d1, d2), [(2 * d2 - d1) / 5, (4 * d1 - 3 * d2) / 5])


def test_loop_and_vectorised_agree():
    rng = np.random.default_rng(0)
    D = rng.uniform(20, 60, size=(30, 2))
    g = MicroGrid()
    assert np.allclose(g.solve_loop(D), g.solve_vectorised(D))
    assert g.solve_vectorised(D).shape == (30, 2)


def test_invalid_demands_and_singular_matrix():
    g = MicroGrid()
    with pytest.raises(ValueError):
        g.solve_day(-1, 5)
    with pytest.raises(ValueError):
        g.solve_day(1, 2, 3)                            # wrong number of demands
    singular = MicroGrid([[1, 2], [2, 4]])
    assert not singular.well_posed
    with pytest.raises(ValueError):
        singular.solve_day(3, 6)


# ---------------- Infeasible days ----------------
def test_infeasible_detection_and_nnls_handling():
    g = MicroGrid()
    D = np.array([[12.0, 11.0], [10.0, 2.0]])          # day 2: x = (4-10)/5 = -1.2
    X = g.solve_vectorised(D)
    assert list(g.infeasible_mask(X)) == [False, True]
    Xn, R = g.solve_nonneg(D)
    assert np.all(Xn >= 0)
    assert np.allclose(Xn[0], [2, 3]) and np.allclose(R[0], 0)     # feasible day unchanged
    assert np.linalg.norm(R[1]) > 0                                # infeasible day has a shortfall


def test_nnls_residual_not_worse_than_clipping():
    g = MicroGrid()
    D = np.array([[10.0, 2.0], [57.0, 29.0]])
    _, Rn = g.solve_nonneg(D)
    _, Rc = g.solve_clipped(D)
    assert np.all(np.linalg.norm(Rn, axis=1) <= np.linalg.norm(Rc, axis=1) + 1e-9)


# ---------------- Input modes ----------------
def test_read_demand_reprompts_until_valid():
    answers = iter(["abc", "", "-5", "nan", " 12.5 "])
    messages = []
    value = read_demand("D1? ", input_fn=lambda _: next(answers), print_fn=messages.append)
    assert value == 12.5
    assert len(messages) == 4                           # one error per bad entry


def test_read_demand_gives_up_after_max_attempts():
    with pytest.raises(ValueError):
        read_demand("D? ", input_fn=lambda _: "oops", print_fn=lambda _: None, max_attempts=3)


def test_csv_roundtrip_and_reproducibility(tmp_path):
    f1, f2 = tmp_path / "a.csv", tmp_path / "b.csv"
    D1 = generate_demand_csv(f1, days=30, seed=5)
    D2 = generate_demand_csv(f2, days=30, seed=5)
    assert D1.shape == (30, 2) and np.array_equal(D1, D2)
    assert np.allclose(load_demand_csv(f1), D1)
    assert f1.read_text() == f2.read_text()


def test_load_csv_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("day,d1_kwh,d2_kwh\n1,50,abc\n")
    with pytest.raises(ValueError):
        load_demand_csv(bad)
    neg = tmp_path / "neg.csv"
    neg.write_text("day,d1_kwh,d2_kwh\n1,-50,30\n")
    with pytest.raises(ValueError):
        load_demand_csv(neg)
    empty = tmp_path / "empty.csv"
    empty.write_text("day,d1_kwh,d2_kwh\n")
    with pytest.raises(ValueError):
        load_demand_csv(empty)
    with pytest.raises(FileNotFoundError):
        load_demand_csv(tmp_path / "missing.csv")


# ---------------- Cost ----------------
def test_energy_cost():
    assert energy_cost([[2, 3]])[0] == 2 * 150 + 3 * 450
    assert np.allclose(energy_cost([[0, 0], [1, 1]]), [0, 600])
    with pytest.raises(ValueError):
        energy_cost([[1, 2, 3]])                        # one price per source


# ---------------- Hybrid and sensitivity ----------------
def test_hybrid_solves_3x3():
    h = HybridMicroGrid()
    s = h.solve_day(51.0, 30.0, 27.0)
    assert np.allclose(h.A @ s, [51.0, 30.0, 27.0])
    assert isinstance(h, MicroGrid) and h.n == 3


def test_hybrid_linearly_dependent_equation():
    # third equation = first + second -> singular
    dep = HybridMicroGrid([[3, 2, 1], [4, 1, 1], [7, 3, 2]])
    assert dep.rank == 2 and not dep.well_posed
    with pytest.raises(ValueError):
        dep.solve_day(49, 30, 79)
    s = dep.solve_least_squares([49, 30, 79])           # consistent: infinitely many solutions
    assert np.allclose(dep.A @ s, [49, 30, 79])


def test_sensitivity_amplification_bounded_by_condition_number():
    g = MicroGrid()
    out = monte_carlo_sensitivity(g, [51.3, 30.5], pct=0.05, n_draws=1000, seed=1)
    assert out["X"].shape == (1000, 2)
    assert out["amplification"].max() <= g.condition_number + 1e-9
    again = monte_carlo_sensitivity(g, [51.3, 30.5], pct=0.05, n_draws=1000, seed=1)
    assert np.array_equal(out["X"], again["X"])
    with pytest.raises(ValueError):
        monte_carlo_sensitivity(g, [51.3, 30.5], pct=0)
