"""Effect-size estimators and conventional interpretations."""

from __future__ import annotations

import numpy as np


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Cliff's delta as P(left > right) - P(left < right)."""

    first = np.asarray(left, dtype=float)
    second = np.sort(np.asarray(right, dtype=float))
    if first.ndim != 1 or second.ndim != 1 or first.size == 0 or second.size == 0:
        raise ValueError("Cliff's delta requires non-empty one-dimensional inputs")
    less = np.searchsorted(second, first, side="left").sum(dtype=np.int64)
    greater = (second.size - np.searchsorted(second, first, side="right")).sum(
        dtype=np.int64
    )
    return float((less - greater) / (first.size * second.size))


def cohens_d_paired(left: np.ndarray, right: np.ndarray) -> float:
    """Compute paired Cohen's d for ``right - left``."""

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Cohen's d inputs must be one-dimensional")
    if first.size == 0 or first.size != second.size:
        raise ValueError("paired Cohen's d requires equal non-zero lengths")
    differences = second - first
    if differences.size == 1:
        return 0.0
    standard_deviation = float(np.std(differences, ddof=1))
    if standard_deviation == 0.0:
        return 0.0 if float(np.mean(differences)) == 0.0 else float(
            np.sign(np.mean(differences)) * np.inf
        )
    return float(np.mean(differences) / standard_deviation)


def interpret_cliffs_delta(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.330:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


def interpret_cohens_d(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"
