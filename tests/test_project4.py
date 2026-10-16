"""Unit tests for Mini-Project 4 (run from the repo root: python -m pytest)."""
import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine

from src.rainfall_model import (
    CropRule,
    Region,
    cosine_similarity,
    euclidean_distance,
    pairwise_matrix,
    pearson_correlation,
)

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
GULU = [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]
MBARARA = [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]


# ---------------- Region ----------------
def test_region_basic_statistics():
    r = Region("Kampala", KAMPALA)
    assert r.annual_total == 1600
    assert r.wettest_month == "May"
    assert r.driest_month == "Sep"
    assert len(r) == 12
    assert r.mean == pytest.approx(1600 / 12)


def test_region_validation():
    with pytest.raises(ValueError):
        Region("short", [1, 2, 3])
    with pytest.raises(ValueError):
        Region("neg", [-1] + [10] * 11)
    with pytest.raises(ValueError):
        Region("nan", [float("nan")] + [10] * 11)


def test_cv_zero_for_flat_rainfall_and_error_for_zero_rain():
    assert Region("flat", [50] * 12).cv == pytest.approx(0.0)
    with pytest.raises(ValueError):
        Region("dry", [0] * 12).cv


# ---------------- CropRule ----------------
def test_crop_rule_classification_and_boundaries():
    maize = CropRule("maize", 125, 200)
    assert maize.classify(150) == "Good for maize"
    assert maize.classify(125) == "Good for maize"      # inclusive lower bound
    assert maize.classify(200) == "Good for maize"      # inclusive upper bound
    assert maize.classify(60) == "Drought risk"
    assert maize.classify(250) == "Waterlogging risk"


def test_crop_rule_invalid_inputs():
    with pytest.raises(ValueError):
        CropRule("bad", 200, 100)
    with pytest.raises(ValueError):
        CropRule("maize", 100, 200).classify(-5)


def test_planting_window_wraps_past_december():
    # wet only in Nov-Feb: a 4-month crop should start in November
    wet = Region("wrap", [150, 150, 0, 0, 0, 0, 0, 0, 0, 0, 150, 150])
    start, score = CropRule("x", 100, 200, season_months=4).planting_window(wet)
    assert start == "Nov" and score == 1.0
    assert CropRule("coffee", 80, 180).planting_window(wet) is None   # perennial


# ---------------- Similarity measures ----------------
def test_cosine_matches_scipy():
    mine = cosine_similarity(KAMPALA, GULU)
    assert mine == pytest.approx(1 - scipy_cosine(KAMPALA, GULU))


def test_cosine_ignores_magnitude_but_euclidean_does_not():
    a = np.array(KAMPALA, dtype=float)
    assert cosine_similarity(a, 3 * a) == pytest.approx(1.0)
    assert euclidean_distance(a, 3 * a) > 500


def test_pearson_matches_numpy_and_zero_vector_edge_cases():
    assert pearson_correlation(KAMPALA, MBARARA) == pytest.approx(np.corrcoef(KAMPALA, MBARARA)[0, 1])
    with pytest.raises(ValueError):
        cosine_similarity([0, 0, 0], [1, 2, 3])
    with pytest.raises(ValueError):
        pearson_correlation([5, 5, 5], [1, 2, 3])


def test_pairwise_matrix_is_symmetric_with_unit_diagonal():
    regions = [Region("K", KAMPALA), Region("G", GULU), Region("M", MBARARA)]
    m = pairwise_matrix(regions, cosine_similarity)
    assert m.shape == (3, 3)
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 1.0)


# ---------------- Seasonality ----------------
def test_modality_detection_on_synthetic_shapes():
    one_peak = Region("uni", [0, 10, 50, 100, 50, 10, 0, 0, 0, 0, 0, 0])
    two_peaks = Region("bi", [0, 100, 20, 0, 0, 0, 0, 0, 100, 20, 0, 0])
    assert one_peak.modality() == "unimodal"
    assert two_peaks.modality() == "bimodal"
    assert Region("dry", [0] * 12).peaks() == []


def test_peak_found_across_december_january_boundary():
    r = Region("wrap", [100, 40, 0, 0, 0, 0, 0, 0, 0, 0, 40, 100])
    assert r.modality() == "unimodal"       # Dec/Jan is one season, not two
