from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.bootstrap import paired_bootstrap_mean_difference
from evaluation.effect_size import (
    cliffs_delta,
    cohens_d_paired,
    interpret_cliffs_delta,
    interpret_cohens_d,
)
from evaluation.io import load_benchmark_folder
from evaluation.significance import (
    _holm_adjust,
    paired_metric_values,
    pairwise_significance,
)
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
        results["method"] == "smart_certified"
    ].groupby("dataset")["accuracy"].mean().mean()
    assert row["mean"] == pytest.approx(expected)
    assert row["n"] == results["dataset"].nunique()
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
    assert set(frame["pairs"]) == {3}
    assert (frame["p_value"] >= 0.0).all()
    assert (frame["p_value"] <= 1.0).all()
    assert (frame["p_value_adjusted_holm"] >= frame["p_value"]).all()
    assert (
        frame["significant_holm_0_05"]
        == (frame["p_value_adjusted_holm"] < 0.05)
    ).all()
    accuracy = frame.loc[
        (frame["left_method"] == "smart_certified")
        & (frame["right_method"] == "cals")
        & (frame["metric"] == "accuracy")
    ].iloc[0]
    assert accuracy["mean_difference_right_minus_left"] == pytest.approx(0.03)


def test_holm_adjustment_is_monotone_in_step_down_order() -> None:
    adjusted = _holm_adjust(np.array([0.04, 0.01, 0.03]))
    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_pairing_rejects_unmatched_repeated_observations(
    benchmark_dir: Path,
) -> None:
    results = load_benchmark_folder(benchmark_dir).results
    missing = results.drop(
        results.loc[
            (results["dataset"] == "alpha")
            & (results["run"] == 0)
            & (results["depth"] == 5)
            & (results["method"] == "cals")
        ].index
    )
    with pytest.raises(ValueError, match="incomplete repeated observations"):
        paired_metric_values(missing, "smart_certified", "cals", "accuracy")


def test_repeated_runs_and_depths_do_not_create_pseudoreplication() -> None:
    rows: list[dict[str, object]] = []
    dataset_differences = {"alpha": 0.1, "beta": 0.1, "gamma": -0.1}
    for dataset, difference in dataset_differences.items():
        for run in range(20):
            for depth in (5, 7):
                rows.extend(
                    (
                        {
                            "dataset": dataset,
                            "run": run,
                            "depth": depth,
                            "method": "smart_certified",
                            "accuracy": 0.5,
                            "tree_nodes": 10.0,
                            "predicate_literals": 5.0,
                            "mean_axp_length": 2.0,
                            "fit_time_seconds": 1.0,
                        },
                        {
                            "dataset": dataset,
                            "run": run,
                            "depth": depth,
                            "method": "cals",
                            "accuracy": 0.5 + difference,
                            "tree_nodes": 10.0 + difference,
                            "predicate_literals": 5.0 + difference,
                            "mean_axp_length": 2.0 + difference,
                            "fit_time_seconds": 1.0 + difference,
                        },
                    )
                )
    results = pd.DataFrame.from_records(rows)

    left, right = paired_metric_values(
        results, "smart_certified", "cals", "accuracy"
    )
    assert left.size == 3
    assert right - left == pytest.approx([0.1, 0.1, -0.1])

    significance = pairwise_significance(results, resamples=100, seed=7)
    accuracy = significance.loc[significance["metric"] == "accuracy"].iloc[0]
    assert accuracy["pairs"] == 3
    assert accuracy["p_value"] == pytest.approx(1.0)


def test_descriptive_statistics_weight_each_dataset_once() -> None:
    rows: list[dict[str, object]] = []
    for dataset, repetitions, value in (("alpha", 100, 0.0), ("beta", 1, 1.0)):
        for run in range(repetitions):
            rows.append(
                {
                    "dataset": dataset,
                    "run": run,
                    "depth": 5,
                    "method": "smart_certified",
                    "accuracy": value,
                    "tree_nodes": value,
                    "predicate_literals": value,
                    "mean_axp_length": value,
                    "fit_time_seconds": value,
                }
            )
    statistics = descriptive_statistics(pd.DataFrame.from_records(rows))
    accuracy = statistics.loc[statistics["metric"] == "accuracy"].iloc[0]
    assert accuracy["n"] == 2
    assert accuracy["mean"] == pytest.approx(0.5)
