"""Certification, warning, search, pruning, and AXp audits."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

from .io import BenchmarkData
from .utils import method_label, ordered_methods, parse_bool_series


def _true_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(parse_bool_series(frame[column], column=column).sum())


def _false_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    values = parse_bool_series(frame[column], column=column)
    return int((~values).sum())


def certification_summary(data: BenchmarkData) -> pd.DataFrame:
    """Summarise every hard certification and data-integrity boundary."""

    results = data.results
    metadata = data.optional("dataset_metadata.csv")
    theorem = data.optional("theorem_certified_results.csv")
    empirical = data.optional("empirical_results.csv")

    dataset_count = int(results["dataset"].nunique())
    skipped = 0
    leakage = 0
    if metadata is not None:
        if "skipped" in metadata:
            skipped = _true_count(metadata, "skipped")
        if "feature_equal_to_label_count" in metadata:
            leakage = int(
                pd.to_numeric(
                    metadata["feature_equal_to_label_count"].fillna(0), errors="coerce"
                )
                .fillna(0)
                .sum()
            )
        elif "suspicious_feature_label_leakage" in metadata:
            leakage = _true_count(metadata, "suspicious_feature_label_leakage")

    theorem_rows = (
        len(theorem)
        if theorem is not None
        else (
            _true_count(results, "theorem_certified")
            if "theorem_certified" in results
            else 0
        )
    )
    empirical_rows = len(empirical) if empirical is not None else 0
    path_violations = (
        int(pd.to_numeric(results["path_violation_count"], errors="coerce").fillna(0).sum())
        if "path_violation_count" in results
        else _false_count(results, "path_certified")
    )
    forbidden = _false_count(results, "all_predicates_backend_allowed")
    cached_violations = _true_count(results, "incompatible_cached_subtree_reused")
    empirical_fallbacks = _true_count(results, "empirical_fallback_used")

    counts = (
        ("datasets", dataset_count, dataset_count),
        ("rows", len(results), len(results)),
        ("theorem_certified_rows", theorem_rows, len(results)),
        ("empirical_rows", empirical_rows, len(results)),
        ("forbidden_predicates", forbidden, len(results)),
        ("feature_label_leakage", leakage, max(dataset_count, 1)),
        ("path_violations", path_violations, len(results)),
        ("cached_subtree_violations", cached_violations, len(results)),
        ("empirical_fallbacks", empirical_fallbacks, len(results)),
        ("skipped_datasets", skipped, max(dataset_count + skipped, 1)),
    )
    records = [
        {
            "audit_item": name,
            "count": int(count),
            "denominator": int(denominator),
            "percentage": 100.0 * count / denominator if denominator else 0.0,
        }
        for name, count, denominator in counts
    ]
    return pd.DataFrame.from_records(records)


def warning_summary(data: BenchmarkData) -> pd.DataFrame:
    warnings = data.optional("benchmark_warnings.csv")
    columns = [
        "dataset",
        "method",
        "warning_type",
        "warning_records",
        "affected_rows",
        "percentage",
    ]
    if warnings is None or warnings.empty:
        return pd.DataFrame(columns=columns)
    required = {"dataset", "method", "warning_type"}
    if not required.issubset(warnings.columns):
        return pd.DataFrame(columns=columns)

    frame = warnings.copy()
    frame["affected_rows"] = pd.to_numeric(
        frame.get("affected_rows", pd.Series(1, index=frame.index)),
        errors="coerce",
    ).fillna(0)
    grouped = (
        frame.groupby(["dataset", "method", "warning_type"], dropna=False, sort=True)
        .agg(
            warning_records=("warning_type", "size"),
            affected_rows=("affected_rows", "sum"),
        )
        .reset_index()
    )
    grouped["percentage"] = 100.0 * grouped["warning_records"] / len(frame)
    return grouped[columns]


def search_summary(data: BenchmarkData) -> pd.DataFrame:
    search = data.optional("search_diagnostics.csv")
    if search is None or search.empty or "method" not in search:
        return pd.DataFrame(
            columns=[
                "method",
                "method_label",
                "rows",
                "greedy_nodes",
                "lookahead_nodes",
                "branch_and_bound_activations",
                "branch_and_bound_avoided",
                "cache_activations",
                "candidate_savings",
                "cache_hits",
                "search_time_seconds",
            ]
        )

    sums = {
        "nodes_using_greedy_selection": "greedy_nodes",
        "nodes_using_selective_lookahead": "lookahead_nodes",
        "branch_and_bound_activation_count": "branch_and_bound_activations",
        "branch_and_bound_avoided_count": "branch_and_bound_avoided",
        "cache_activation_count": "cache_activations",
        "estimated_work_saved": "candidate_savings",
        "search_time": "search_time_seconds",
    }
    records: list[dict[str, Any]] = []
    cache = data.optional("cache_diagnostics.csv")
    for method in ordered_methods(search["method"].astype(str).unique()):
        rows = search.loc[search["method"].astype(str) == method]
        record: dict[str, Any] = {
            "method": method,
            "method_label": method_label(method),
            "rows": len(rows),
        }
        for source, target in sums.items():
            record[target] = (
                float(pd.to_numeric(rows[source], errors="coerce").fillna(0).sum())
                if source in rows
                else 0.0
            )
        cache_hits = 0.0
        if cache is not None and not cache.empty and "method" in cache:
            cache_rows = cache.loc[cache["method"].astype(str) == method]
            hit_columns = [
                column
                for column in (
                    "predicate_mask_hits",
                    "candidate_hits",
                    "subtree_hits",
                )
                if column in cache_rows
            ]
            cache_hits = sum(
                float(pd.to_numeric(cache_rows[column], errors="coerce").fillna(0).sum())
                for column in hit_columns
            )
        record["cache_hits"] = cache_hits
        records.append(record)
    return pd.DataFrame.from_records(records)


def pruning_summary(data: BenchmarkData) -> pd.DataFrame:
    pruning = data.optional("pruning_diagnostics.csv")
    if pruning is None or pruning.empty or "method" not in pruning:
        return pd.DataFrame()

    mean_columns = (
        "validation_accuracy_before",
        "validation_accuracy_after",
        "validation_balanced_accuracy_before",
        "validation_balanced_accuracy_after",
        "validation_minority_recall_before",
        "validation_minority_recall_after",
        "nodes_before",
        "nodes_after",
    )
    records: list[dict[str, Any]] = []
    for method in ordered_methods(pruning["method"].astype(str).unique()):
        rows = pruning.loc[pruning["method"].astype(str) == method]
        record: dict[str, Any] = {
            "method": method,
            "method_label": method_label(method),
            "rows": len(rows),
        }
        for column in mean_columns:
            record[column] = (
                float(pd.to_numeric(rows[column], errors="coerce").fillna(0).mean())
                if column in rows
                else 0.0
            )
        before = record["nodes_before"]
        after = record["nodes_after"]
        record["tree_reduction_percentage"] = (
            100.0 * (before - after) / before if before else 0.0
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _family_counts(rows: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if "selected_family_counts" not in rows:
        return counts
    for value in rows["selected_family_counts"].dropna().astype(str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        for family, count in parsed.items():
            counts[str(family)] = counts.get(str(family), 0) + int(count)
    return dict(sorted(counts.items()))


def axp_summary(data: BenchmarkData) -> pd.DataFrame:
    results = data.results
    records: list[dict[str, Any]] = []
    for method in ordered_methods(results["method"].unique()):
        rows = results.loc[results["method"] == method]
        values = rows["mean_axp_length"].to_numpy(dtype=float)
        family_counts = _family_counts(rows)
        records.append(
            {
                "method": method,
                "method_label": method_label(method),
                "rows": len(rows),
                "mean_axp": float(np.mean(values)),
                "median_axp": float(np.median(values)),
                "std_axp": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "minimum_axp": float(np.min(values)),
                "maximum_axp": float(np.max(values)),
                "family_usage": " | ".join(
                    f"{family}:{count}" for family, count in family_counts.items()
                )
                or "not reported",
            }
        )
    return pd.DataFrame.from_records(records)
