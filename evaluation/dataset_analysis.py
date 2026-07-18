"""Per-dataset winners, ranks, and critical-difference information."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import studentized_range

from .config import METRICS
from .utils import method_label, ordered_methods


@dataclass(frozen=True)
class DatasetAnalysis:
    dataset_summary: pd.DataFrame
    method_ranks: pd.DataFrame
    dataset_metric_means: pd.DataFrame
    critical_difference: float


def _nemenyi_critical_difference(
    method_count: int,
    dataset_count: int,
    alpha: float = 0.05,
) -> float:
    if method_count < 2 or dataset_count < 1:
        return 0.0
    q_alpha = float(studentized_range.ppf(1.0 - alpha, method_count, np.inf))
    q_alpha /= math.sqrt(2.0)
    return q_alpha * math.sqrt(
        method_count * (method_count + 1) / (6.0 * dataset_count)
    )


def analyse_datasets(results: pd.DataFrame) -> DatasetAnalysis:
    """Aggregate repetitions and select a deterministic best method per dataset."""

    metric_columns = [metric.column for metric in METRICS]
    grouped = (
        results.groupby(["dataset", "method"], sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    methods = ordered_methods(grouped["method"].unique())
    datasets = sorted(grouped["dataset"].unique())

    ranks: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        pivot = grouped.pivot(index="dataset", columns="method", values=metric.column)
        pivot = pivot.reindex(index=datasets, columns=methods)
        ranks[metric.column] = pivot.rank(
            axis=1,
            method="average",
            ascending=not metric.higher_is_better,
        )

    accuracy_ranks = ranks["accuracy"]
    critical_difference = _nemenyi_critical_difference(
        len(methods), len(datasets)
    )

    summary_records: list[dict[str, object]] = []
    wins = {method: 0 for method in methods}
    for dataset in datasets:
        rows = grouped.loc[grouped["dataset"] == dataset].copy()
        rows["_method_order"] = rows["method"].map(
            {method: index for index, method in enumerate(methods)}
        )
        rows = rows.sort_values(
            [
                "accuracy",
                "tree_nodes",
                "mean_axp_length",
                "fit_time_seconds",
                "_method_order",
            ],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        best = rows.iloc[0]
        method = str(best["method"])
        wins[method] += 1
        summary_records.append(
            {
                "dataset": dataset,
                "best_method": method,
                "best_method_label": method_label(method),
                "accuracy": float(best["accuracy"]),
                "tree_nodes": float(best["tree_nodes"]),
                "mean_axp_length": float(best["mean_axp_length"]),
                "fit_time_seconds": float(best["fit_time_seconds"]),
            }
        )

    rank_records: list[dict[str, object]] = []
    for method in methods:
        record: dict[str, object] = {
            "method": method,
            "method_label": method_label(method),
            "wins": wins[method],
            "average_rank": float(accuracy_ranks[method].mean()),
            "critical_difference": critical_difference,
            "datasets": len(datasets),
        }
        for metric in METRICS:
            record[f"{metric.column}_average_rank"] = float(
                ranks[metric.column][method].mean()
            )
        rank_records.append(record)

    method_ranks = pd.DataFrame.from_records(rank_records).sort_values(
        ["average_rank", "method"], kind="mergesort"
    )
    dataset_summary = pd.DataFrame.from_records(summary_records)
    return DatasetAnalysis(
        dataset_summary=dataset_summary,
        method_ranks=method_ranks.reset_index(drop=True),
        dataset_metric_means=grouped,
        critical_difference=critical_difference,
    )
