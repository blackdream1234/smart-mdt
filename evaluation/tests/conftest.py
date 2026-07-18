from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


METHODS = ("smart_certified", "cals", "cals_compact_explain")


def create_benchmark(root: Path, *, optional: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for dataset_index, dataset in enumerate(("alpha", "beta", "gamma")):
        for run in range(3):
            for method_index, method in enumerate(METHODS):
                accuracy = (
                    0.74
                    + 0.03 * method_index
                    + 0.01 * dataset_index
                    + 0.001 * run
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "run": run,
                        "depth": 5,
                        "method": method,
                        "accuracy": accuracy,
                        "tree_nodes": 25 - 6 * method_index + dataset_index,
                        "literals_after_prune": 18 - 4 * method_index + dataset_index,
                        "mean_axp_length": 3.2 - 0.25 * method_index,
                        "total_fit_time": 0.8 + 0.5 * method_index + 0.02 * run,
                        "theorem_certified": True,
                        "path_certified": True,
                        "path_violation_count": 0,
                        "empirical_fallback_used": False,
                        "incompatible_cached_subtree_reused": False,
                        "all_predicates_backend_allowed": True,
                        "selected_family_counts": "{'Unary': 2, 'Horn': 1}",
                    }
                )
    results = pd.DataFrame(rows)
    results.to_csv(root / "full_results.csv", index=False)
    if not optional:
        return root

    results.to_csv(root / "theorem_certified_results.csv", index=False)
    results.iloc[0:0].to_csv(root / "empirical_results.csv", index=False)
    (
        results.groupby("method", sort=True)
        .agg(
            rows=("method", "size"),
            accuracy_mean=("accuracy", "mean"),
            tree_nodes_mean=("tree_nodes", "mean"),
            mean_axp_length_mean=("mean_axp_length", "mean"),
        )
        .reset_index()
        .to_csv(root / "summary_by_method.csv", index=False)
    )
    pd.DataFrame(
        {
            "dataset": ["alpha", "beta", "gamma"],
            "skipped": [False, False, False],
            "feature_equal_to_label_count": [0, 0, 0],
        }
    ).to_csv(root / "dataset_metadata.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["alpha", "beta"],
            "method": ["cals", "cals_compact_explain"],
            "warning_type": ["tiny_tree", "constant_tree"],
            "affected_rows": [2, 1],
        }
    ).to_csv(root / "benchmark_warnings.csv", index=False)

    diagnostics = results[["dataset", "run", "depth", "method"]].copy()
    diagnostics["nodes_using_greedy_selection"] = 3
    diagnostics["nodes_using_selective_lookahead"] = 1
    diagnostics["branch_and_bound_activation_count"] = 0
    diagnostics["branch_and_bound_avoided_count"] = 2
    diagnostics["cache_activation_count"] = 4
    diagnostics["estimated_work_saved"] = 10
    diagnostics["search_time"] = 0.5
    diagnostics.to_csv(root / "search_diagnostics.csv", index=False)

    cache = results[["dataset", "run", "depth", "method"]].copy()
    cache["predicate_mask_hits"] = 5
    cache["predicate_mask_misses"] = 3
    cache["candidate_hits"] = 2
    cache["candidate_misses"] = 1
    cache["subtree_hits"] = 0
    cache["subtree_misses"] = 1
    cache.to_csv(root / "cache_diagnostics.csv", index=False)

    pruning = results[["dataset", "run", "depth", "method"]].copy()
    pruning["validation_accuracy_before"] = 0.80
    pruning["validation_accuracy_after"] = 0.80
    pruning["validation_balanced_accuracy_before"] = 0.75
    pruning["validation_balanced_accuracy_after"] = 0.76
    pruning["validation_minority_recall_before"] = 0.65
    pruning["validation_minority_recall_after"] = 0.66
    pruning["nodes_before"] = 30
    pruning["nodes_after"] = 15
    pruning.to_csv(root / "pruning_diagnostics.csv", index=False)

    family = results[["dataset", "run", "depth", "method"]].copy()
    family["compatible_family_count"] = 3
    family.to_csv(root / "family_budget_diagnostics.csv", index=False)

    beam = results[["dataset", "run", "depth", "method"]].copy()
    beam["candidate_beam_width"] = 8
    beam["tree_beam_width"] = 4
    beam["lookahead_depth"] = 2
    beam["node_budget"] = 100
    beam["total_fit_time"] = results["total_fit_time"]
    beam.to_csv(root / "beam_diagnostics.csv", index=False)

    results.to_csv(root / "axp_metadata.csv", index=False)
    results.to_csv(root / "tuning_diagnostics.csv", index=False)
    (root / "README_RESULTS.md").write_text("# Synthetic benchmark\n", encoding="utf-8")
    return root


@pytest.fixture
def benchmark_dir(tmp_path: Path) -> Path:
    return create_benchmark(tmp_path / "benchmark")
