from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evaluation.bootstrap import paired_bootstrap_mean_difference
from evaluation.effect_size import (
    cliffs_delta,
    cohens_d_paired,
    interpret_cliffs_delta,
    interpret_cohens_d,
)
from evaluation.io import load_benchmark_folder
from evaluation.significance import pairwise_significance
from evaluation.statistics import descriptive_statistics


def test_descriptive_statistics_have_expected_mean_and_ci(
    benchmark_dir: Path,
) -> None:
    results = load_benchmark_folder(benchmark_dir).results
    frame = descriptive_statistics(results)
    row = frame.loc[
        (frame["method"] == "smart_certified") & (frame["metric"] == "accuracy")
    ].iloc[0]
    expected = results.loc[
        results["method"] == "smart_certified", "accuracy"
    ].mean()
    assert row["mean"] == pytest.approx(expected)
    assert row["ci_lower"] <= row["mean"] <= row["ci_upper"]
    assert row["variance"] >= 0.0


def test_bootstrap_is_paired_seeded_and_deterministic() -> None:
    left = np.array([1.0, 2.0, 3.0, 4.0])
    right = left + 0.5
    first = paired_bootstrap_mean_difference(
        left, right, resamples=500, seed=17
    )
    second = paired_bootstrap_mean_difference(
        left, right, resamples=500, seed=17
    )
    assert first == second
    assert first.observed == pytest.approx(0.5)
    assert first.lower == pytest.approx(0.5)
    assert first.upper == pytest.approx(0.5)


def test_effect_sizes_and_interpretations() -> None:
    low = np.array([1.0, 2.0, 3.0])
    high = np.array([4.0, 5.0, 6.0])
    assert cliffs_delta(high, low) == pytest.approx(1.0)
    assert interpret_cliffs_delta(1.0) == "large"
    paired_d = cohens_d_paired(low, high)
    assert np.isinf(paired_d)
    assert interpret_cohens_d(paired_d) == "large"


def test_wilcoxon_pipeline_is_paired_and_complete(benchmark_dir: Path) -> None:
    results = load_benchmark_folder(benchmark_dir).results
    frame = pairwise_significance(results, resamples=100, seed=9)
    assert len(frame) == 15
    assert set(frame["pairs"]) == {9}
    assert (frame["p_value"] >= 0.0).all()
    assert (frame["p_value"] <= 1.0).all()
    accuracy = frame.loc[
        (frame["left_method"] == "smart_certified")
        & (frame["right_method"] == "cals")
        & (frame["metric"] == "accuracy")
    ].iloc[0]
    assert accuracy["mean_difference_right_minus_left"] == pytest.approx(0.03)
