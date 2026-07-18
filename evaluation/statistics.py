"""Descriptive statistics for benchmark metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .config import METRICS
from .utils import method_label, ordered_methods


def mean_confidence_interval(
    values: np.ndarray | pd.Series,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Student-t confidence interval for the arithmetic mean."""

    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("confidence interval requires at least one value")
    mean = float(np.mean(data))
    if data.size == 1:
        return mean, mean
    standard_error = float(scipy_stats.sem(data))
    radius = float(
        scipy_stats.t.ppf((1.0 + confidence_level) / 2.0, data.size - 1)
        * standard_error
    )
    return mean - radius, mean + radius


def descriptive_statistics(
    results: pd.DataFrame,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Compute publication-ready descriptive statistics by method and metric."""

    records: list[dict[str, object]] = []
    for method in ordered_methods(results["method"].unique()):
        method_rows = results.loc[results["method"] == method]
        for metric in METRICS:
            values = method_rows[metric.column].to_numpy(dtype=float)
            lower, upper = mean_confidence_interval(values, confidence_level)
            records.append(
                {
                    "method": method,
                    "method_label": method_label(method),
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "variance": (
                        float(np.var(values, ddof=1)) if values.size > 1 else 0.0
                    ),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": confidence_level,
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    if not np.isfinite(
        frame[
            [
                "mean",
                "median",
                "variance",
                "std",
                "minimum",
                "maximum",
                "ci_lower",
                "ci_upper",
            ]
        ].to_numpy()
    ).all():
        raise ValueError("descriptive statistics contain non-finite values")
    return frame
