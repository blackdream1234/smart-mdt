"""Deterministic paired bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    lower: float
    upper: float
    mean: float
    observed: float
    resamples: int
    seed: int


def paired_bootstrap_mean_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 0,
    chunk_size: int = 512,
) -> BootstrapResult:
    """Bootstrap the paired mean of ``right - left``."""

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("paired bootstrap inputs must be one-dimensional")
    if first.size == 0 or first.size != second.size:
        raise ValueError("paired bootstrap inputs must have equal non-zero length")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    differences = second - first
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    offset = 0
    while offset < resamples:
        count = min(chunk_size, resamples - offset)
        indices = rng.integers(0, differences.size, size=(count, differences.size))
        estimates[offset : offset + count] = differences[indices].mean(axis=1)
        offset += count

    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return BootstrapResult(
        lower=float(lower),
        upper=float(upper),
        mean=float(np.mean(estimates)),
        observed=float(np.mean(differences)),
        resamples=resamples,
        seed=seed,
    )
