"""Metrics: overlap, boundary distance, calibration, uncertainty quality, stats."""

from __future__ import annotations

import numpy as np
import pytest

from evissl.metrics import (
    adaptive_calibration_error,
    aggregate,
    assd,
    ause,
    bootstrap_ci,
    boundary_f1,
    brier_score,
    compare,
    compute_all,
    compute_batch,
    compute_calibration,
    dice_score,
    expected_calibration_error,
    hd95,
    holm_bonferroni,
    iou_score,
    negative_log_likelihood,
    sparsification_curve,
    surface_mask,
    uncertainty_error_auroc,
)


def square(size=64, lo=20, hi=44):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[lo:hi, lo:hi] = 1
    return mask


# --------------------------------------------------------------------------- #
# Overlap
# --------------------------------------------------------------------------- #


def test_identical_masks_score_perfectly():
    mask = square()
    metrics = compute_all(mask, mask)
    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["hd95"] == 0.0
    assert metrics["assd"] == 0.0
    assert metrics["boundary_f1"] == 1.0


def test_disjoint_masks_score_zero_overlap():
    a = square(64, 5, 20)
    b = square(64, 40, 60)
    assert dice_score(a, b) == 0.0
    assert iou_score(a, b) == 0.0


def test_dice_and_iou_are_consistent():
    """IoU = Dice / (2 - Dice) is an algebraic identity, so it must hold exactly."""
    a, b = square(), square(64, 24, 50)
    d, i = dice_score(a, b), iou_score(a, b)
    assert i == pytest.approx(d / (2 - d))


def test_both_empty_is_perfect_not_undefined():
    empty = np.zeros((32, 32), dtype=np.uint8)
    metrics = compute_all(empty, empty)
    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["hd95"] == 0.0


def test_one_empty_leaves_distances_undefined():
    """There is no surface to measure to, so NaN - not a made-up large number.

    Substituting the image diagonal would reward a model for failing completely
    instead of recording that it failed.
    """
    metrics = compute_all(np.zeros((32, 32), np.uint8), square(32, 8, 20))
    assert metrics["dice"] == 0.0
    assert np.isnan(metrics["hd95"])
    assert np.isnan(metrics["assd"])
    assert metrics["boundary_f1"] == 0.0


def test_undefined_ratios_are_nan_not_one():
    """An empty prediction must not report perfect precision."""
    metrics = compute_all(np.zeros((32, 32), np.uint8), square(32, 8, 20))
    assert np.isnan(metrics["precision"])


# --------------------------------------------------------------------------- #
# Boundary
# --------------------------------------------------------------------------- #


def test_surface_mask_is_one_pixel_thick():
    mask = square(32, 8, 24)
    surface = surface_mask(mask)
    assert surface.sum() < mask.sum()
    # A 16x16 filled square has a 4*16 - 4 = 60 pixel inner border.
    assert surface.sum() == 60


def test_surface_of_an_empty_mask_is_empty():
    assert surface_mask(np.zeros((8, 8), np.uint8)).sum() == 0


def test_hd95_recovers_a_known_translation():
    """Shifting a square by k pixels puts its boundary k pixels away."""
    for shift in (1, 2, 4):
        a = square(64, 20, 44)
        b = square(64, 20 + shift, 44 + shift)
        assert hd95(a, b) == pytest.approx(float(shift), abs=0.5)


def test_assd_is_at_most_hd95():
    a, b = square(), square(64, 24, 50)
    assert assd(a, b) <= hd95(a, b) + 1e-9


def test_boundary_f1_tolerance_is_monotone():
    a, b = square(64, 20, 44), square(64, 24, 48)
    scores = [boundary_f1(a, b, tol) for tol in (0.0, 1.0, 2.0, 8.0)]
    assert scores == sorted(scores)
    assert scores[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_aggregate_excludes_nan_and_reports_the_count():
    good = square()
    per_image = [
        compute_all(good, good),
        compute_all(np.zeros_like(good), good),  # undefined distances
        compute_all(square(64, 22, 46), good),
    ]
    agg = aggregate(per_image)
    assert agg["dice_n"] == 3
    assert agg["hd95_n"] == 2, "the undefined distance must be excluded"
    assert np.isfinite(agg["hd95"])


def test_aggregate_of_nothing_is_empty():
    assert aggregate([]) == {}


def test_compute_batch_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_batch(np.zeros((2, 8, 8), np.uint8), np.zeros((3, 8, 8), np.uint8))


def test_fast_mode_omits_only_the_boundary_metrics():
    mask = square(32, 8, 20)
    fast = compute_all(mask, mask, boundary_metrics=False)
    full = compute_all(mask, mask, boundary_metrics=True)
    assert set(full) - set(fast) == {"hd95", "assd", "boundary_f1"}
    for key in fast:
        assert fast[key] == full[key]


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_perfect_calibration_has_near_zero_error():
    rng = np.random.default_rng(0)
    target = (rng.random(20000) > 0.6).astype(float)
    probability = np.where(target > 0.5, 0.999, 0.001)
    out = compute_calibration(probability, target)
    assert out["ece"] < 0.01
    assert out["brier"] < 0.01


def test_overconfidence_is_detected_at_the_right_magnitude():
    """A model 80% accurate but claiming certainty must show ECE near 0.20."""
    rng = np.random.default_rng(1)
    target = (rng.random(20000) > 0.5).astype(float)
    wrong = rng.random(20000) < 0.2
    probability = np.where(wrong, 1.0 - target, target).astype(float)
    out = compute_calibration(probability, target)
    assert out["ece"] == pytest.approx(0.20, abs=0.02)
    assert out["pixel_accuracy"] == pytest.approx(0.80, abs=0.02)


def test_equal_width_and_equal_mass_agree_on_a_clean_case():
    rng = np.random.default_rng(2)
    confidence = rng.uniform(0.5, 1.0, 20000)
    correct = (rng.random(20000) < confidence).astype(float)
    ece = expected_calibration_error(confidence, correct, bins=15)["ece"]
    ace = adaptive_calibration_error(confidence, correct, bins=15)
    assert abs(ece - ace) < 0.03


def test_reliability_curve_bins_are_populated_and_ordered():
    rng = np.random.default_rng(3)
    confidence = rng.uniform(0.0, 1.0, 5000)
    correct = (rng.random(5000) < confidence).astype(float)
    curve = expected_calibration_error(confidence, correct, bins=10)
    assert len(curve["bin_confidence"]) > 5
    assert np.all(np.diff(curve["bin_confidence"]) > 0)
    assert curve["bin_count"].sum() == 5000


def test_calibration_of_an_empty_input_is_nan_not_an_error():
    out = expected_calibration_error(np.array([]), np.array([]))
    assert np.isnan(out["ece"])


def test_brier_and_nll_reward_the_truth():
    target = np.array([0.0, 1.0, 1.0, 0.0])
    good = np.array([0.02, 0.98, 0.97, 0.03])
    bad = np.array([0.98, 0.02, 0.03, 0.97])
    assert brier_score(good, target) < brier_score(bad, target)
    assert negative_log_likelihood(good, target) < negative_log_likelihood(bad, target)


def test_nll_is_finite_at_the_extremes():
    """Clipping must keep log(0) out of the estimator."""
    assert np.isfinite(negative_log_likelihood(np.array([0.0, 1.0]), np.array([1.0, 0.0])))


# --------------------------------------------------------------------------- #
# Uncertainty quality
# --------------------------------------------------------------------------- #


def test_ause_spans_zero_to_one():
    rng = np.random.default_rng(4)
    error = (rng.random(4000) < 0.3).astype(float)
    assert ause(error, error) == pytest.approx(0.0, abs=1e-6)          # oracle
    assert ause(rng.random(4000), error) == pytest.approx(1.0, abs=0.1)  # uninformative


def test_auroc_spans_half_to_one():
    rng = np.random.default_rng(5)
    error = (rng.random(4000) < 0.3).astype(float)
    assert uncertainty_error_auroc(error, error) == pytest.approx(1.0)
    assert uncertainty_error_auroc(rng.random(4000), error) == pytest.approx(0.5, abs=0.05)


def test_auroc_handles_ties_exactly():
    """All-tied uncertainty carries no information, so AUROC must be exactly 0.5."""
    error = np.array([0.0, 1.0, 0.0, 1.0])
    assert uncertainty_error_auroc(np.ones(4), error) == pytest.approx(0.5)


def test_auroc_is_undefined_without_both_classes():
    assert np.isnan(uncertainty_error_auroc(np.random.rand(10), np.zeros(10)))
    assert np.isnan(uncertainty_error_auroc(np.random.rand(10), np.ones(10)))


def test_sparsification_curves_are_ordered():
    """Oracle must dominate the model, which must beat nothing at fraction 0."""
    rng = np.random.default_rng(6)
    error = (rng.random(3000) < 0.25).astype(float)
    uncertainty = error * 0.8 + rng.random(3000) * 0.2  # informative but noisy
    curve = sparsification_curve(uncertainty, error, steps=10)
    assert np.all(curve["oracle"] <= curve["model"] + 1e-9)
    assert curve["model"][0] == pytest.approx(curve["random"][0], abs=1e-9)
    assert curve["model"][-1] < curve["model"][0]


def test_sparsification_of_empty_input_is_empty():
    curve = sparsification_curve(np.array([]), np.array([]))
    assert curve["fraction"].size == 0


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def test_bootstrap_interval_brackets_the_mean():
    rng = np.random.default_rng(7)
    values = rng.normal(0.85, 0.05, 200)
    interval = bootstrap_ci(values, n_resamples=1000, seed=0)
    assert interval.lower < interval.estimate < interval.upper
    assert interval.estimate == pytest.approx(values.mean())
    assert interval.n == 200


def test_bootstrap_is_reproducible_given_a_seed():
    values = np.random.default_rng(8).normal(0.5, 0.1, 100)
    a = bootstrap_ci(values, 500, seed=3)
    b = bootstrap_ci(values, 500, seed=3)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_bootstrap_drops_nan_and_survives_degenerate_input():
    values = np.array([0.5, np.nan, np.nan])
    assert bootstrap_ci(values).n == 1
    assert bootstrap_ci(np.array([np.nan, np.nan])).n == 0


def test_paired_comparison_detects_a_real_difference():
    rng = np.random.default_rng(9)
    baseline = rng.normal(0.80, 0.05, 150)
    better = baseline + rng.normal(0.04, 0.01, 150)  # paired improvement
    result = compare(better, baseline, "ours", "baseline")
    assert result.p_value < 0.001
    assert result.difference.lower > 0
    assert result.effect_size > 1.0


def test_paired_comparison_keeps_the_false_positive_rate_near_alpha():
    """No-difference data must rarely be flagged significant.

    Asserted over many seeds rather than one, because a single draw can legally
    land just inside the rejection region - a test that demands otherwise is
    testing the seed, not the estimator.
    """
    flagged = 0
    trials = 60
    for seed in range(trials):
        rng = np.random.default_rng(1000 + seed)
        a = rng.normal(0.8, 0.05, 150)
        b = rng.normal(0.8, 0.05, 150)
        if compare(a, b, n_resamples=400, seed=seed).significant:
            flagged += 1
    # Nominal alpha is 0.05; allow generous slack for 60 trials.
    assert flagged / trials < 0.20, f"flagged {flagged}/{trials} null comparisons"


def test_identical_inputs_leave_the_test_undefined():
    values = np.random.default_rng(11).normal(0.8, 0.05, 50)
    result = compare(values, values.copy())
    assert np.isnan(result.p_value)
    assert result.significant is False


def test_mismatched_pairs_are_rejected():
    from evissl.metrics import paired_bootstrap_difference

    with pytest.raises(ValueError, match="match in shape"):
        paired_bootstrap_difference(np.zeros(5), np.zeros(6))


def test_holm_correction_is_monotone_and_conservative():
    rng = np.random.default_rng(12)
    baseline = rng.normal(0.8, 0.05, 120)
    comparisons = [
        compare(baseline + shift, baseline, f"m{i}", "base")
        for i, shift in enumerate([0.0005, 0.01, 0.03, 0.06])
    ]
    raw = [c.p_value for c in comparisons]
    holm_bonferroni(comparisons)
    adjusted = [c.p_adjusted for c in comparisons]

    for r, a in zip(raw, adjusted, strict=True):
        assert a >= r - 1e-12, "correction must never lower a p-value"
        assert a <= 1.0
    # Monotone in the original ordering of p-values.
    order = np.argsort(raw)
    ordered = [adjusted[i] for i in order]
    assert ordered == sorted(ordered)


def test_holm_ignores_undefined_tests():
    values = np.random.default_rng(13).normal(0.8, 0.05, 40)
    comparisons = [compare(values, values.copy(), "same", "base")]
    holm_bonferroni(comparisons)
    assert comparisons[0].p_adjusted is None
