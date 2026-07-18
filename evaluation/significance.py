"""Paired significance tests for the three primary certified learners."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .bootstrap import paired_bootstrap_mean_difference
from .config import METRICS, PAIRWISE_COMPARISONS
from .effect_size import (
    cliffs_delta,
    cohens_d_paired,
    interpret_cliffs_delta,
    interpret_cohens_d,
)
from .utils import method_label, stable_seed


PAIR_KEYS = ["dataset", "run", "depth"]


def paired_metric_values(
    results: pd.DataFrame,
    left_method: str,
    right_method: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align two methods by dataset, run, and depth."""

    subset = results.loc[
        results["method"].isin((left_method, right_method)),
        [*PAIR_KEYS, "method", metric],
    ]
    pivot = subset.pivot(index=PAIR_KEYS, columns="method", values=metric)
    if left_method not in pivot or right_method not in pivot:
        raise ValueError(f"cannot pair {left_method} and {right_method} for {metric}")
    if pivot[[left_method, right_method]].isna().any().any():
        missing = int(pivot[[left_method, right_method]].isna().any(axis=1).sum())
        raise ValueError(
            f"{left_method} vs {right_method} has {missing} incomplete pairs for {metric}"
        )
    pivot = pivot.sort_index()
    return (
        pivot[left_method].to_numpy(dtype=float),
        pivot[right_method].to_numpy(dtype=float),
    )


def _wilcoxon(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    differences = right - left
    if np.all(differences == 0.0):
        return 0.0, 1.0
    result = wilcoxon(
        right,
        left,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def pairwise_significance(
    results: pd.DataFrame,
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute paired Wilcoxon, bootstrap, and effect-size results."""

    available = set(results["method"].unique())
    records: list[dict[str, object]] = []
    for left_method, right_method in PAIRWISE_COMPARISONS:
        if left_method not in available or right_method not in available:
            continue
        for metric in METRICS:
            left, right = paired_metric_values(
                results, left_method, right_method, metric.column
            )
            statistic, p_value = _wilcoxon(left, right)
            local_seed = stable_seed(seed, left_method, right_method, metric.column)
            bootstrap = paired_bootstrap_mean_difference(
                left,
                right,
                resamples=resamples,
                confidence_level=confidence_level,
                seed=local_seed,
            )
            cliff = cliffs_delta(right, left)
            cohen = cohens_d_paired(left, right)
            records.append(
                {
                    "comparison": f"{method_label(left_method)} vs "
                    f"{method_label(right_method)}",
                    "left_method": left_method,
                    "right_method": right_method,
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "pairs": int(left.size),
                    "left_mean": float(np.mean(left)),
                    "right_mean": float(np.mean(right)),
                    "mean_difference_right_minus_left": bootstrap.observed,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "bootstrap_mean": bootstrap.mean,
                    "bootstrap_ci_lower": bootstrap.lower,
                    "bootstrap_ci_upper": bootstrap.upper,
                    "bootstrap_resamples": bootstrap.resamples,
                    "bootstrap_seed": bootstrap.seed,
                    "cliffs_delta_right_vs_left": cliff,
                    "cliffs_interpretation": interpret_cliffs_delta(cliff),
                    "cohens_d_paired": cohen,
                    "cohens_interpretation": interpret_cohens_d(cohen),
                }
            )
    return pd.DataFrame.from_records(records)
