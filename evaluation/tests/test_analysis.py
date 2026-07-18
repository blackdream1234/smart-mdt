from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.certification import (
    axp_summary,
    certification_summary,
    pruning_summary,
    search_summary,
    warning_summary,
)
from evaluation.dataset_analysis import analyse_datasets
from evaluation.io import load_benchmark_folder


def test_dataset_summary_wins_ranks_and_critical_difference(
    benchmark_dir: Path,
) -> None:
    data = load_benchmark_folder(benchmark_dir)
    analysis = analyse_datasets(data.results)
    assert len(analysis.dataset_summary) == 3
    assert set(analysis.dataset_summary["best_method"]) == {
        "cals_compact_explain"
    }
    assert analysis.method_ranks["wins"].sum() == 3
    assert analysis.method_ranks.iloc[0]["method"] == "cals_compact_explain"
    assert analysis.critical_difference > 0.0


def test_certification_and_optional_diagnostic_summaries(
    benchmark_dir: Path,
) -> None:
    data = load_benchmark_folder(benchmark_dir)
    certification = certification_summary(data).set_index("audit_item")
    assert certification.loc["theorem_certified_rows", "count"] == 27
    for item in (
        "empirical_rows",
        "forbidden_predicates",
        "feature_label_leakage",
        "path_violations",
        "cached_subtree_violations",
        "empirical_fallbacks",
    ):
        assert certification.loc[item, "count"] == 0

    warnings = warning_summary(data)
    assert warnings["percentage"].sum() == pytest.approx(100.0)
    search = search_summary(data)
    assert search["greedy_nodes"].sum() == 81
    assert search["cache_hits"].sum() > 0
    pruning = pruning_summary(data)
    assert set(pruning["tree_reduction_percentage"]) == {50.0}
    axp = axp_summary(data)
    assert axp["family_usage"].str.contains("Horn").all()
