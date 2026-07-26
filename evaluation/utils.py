"""Shared deterministic utilities."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import METHOD_LABELS, METHOD_ORDER


TRUE_VALUES = frozenset({"1", "true", "yes", "y"})
FALSE_VALUES = frozenset({"0", "false", "no", "n", ""})


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a process-independent NumPy seed from stable text."""

    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def ordered_methods(methods: Iterable[str]) -> list[str]:
    present = set(methods)
    known = [method for method in METHOD_ORDER if method in present]
    return known + sorted(present.difference(known))


def parse_bool_series(series: pd.Series, *, column: str) -> pd.Series:
    """Parse common CSV boolean representations and reject ambiguity."""

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    invalid = ~normalized.isin(TRUE_VALUES | FALSE_VALUES)
    if invalid.any():
        values = sorted(normalized[invalid].unique().tolist())
        raise ValueError(f"column {column!r} contains invalid booleans: {values}")
    return normalized.isin(TRUE_VALUES)


def safe_numeric(series: pd.Series, *, column: str) -> pd.Series:
    try:
        converted = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"column {column!r} must be numeric") from error
    if converted.isna().any():
        raise ValueError(f"column {column!r} contains missing numeric values")
    values = converted.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"column {column!r} contains non-finite values")
    return converted


def safe_nonnegative_integers(
    series: pd.Series,
    *,
    column: str,
    maximum: int = np.iinfo(np.int64).max,
) -> pd.Series:
    """Parse bounded counts without lossy floating-point range checks."""

    parsed: list[int] = []
    for raw in series:
        if pd.isna(raw) or isinstance(raw, (bool, np.bool_)):
            raise ValueError(
                f"column {column!r} must contain non-negative integers"
            )
        try:
            value = Decimal(str(raw).strip())
        except (InvalidOperation, ValueError):
            raise ValueError(
                f"column {column!r} must contain non-negative integers"
            ) from None
        if (
            not value.is_finite()
            or value != value.to_integral_value()
            or value < 0
            or value > maximum
        ):
            raise ValueError(
                f"column {column!r} must contain non-negative integers"
            )
        parsed.append(int(value))
    dtype = np.int64 if maximum <= np.iinfo(np.int64).max else np.uint64
    return pd.Series(parsed, index=series.index, dtype=dtype, name=series.name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to JSON-safe records with deterministic nulls."""

    cleaned = frame.astype(object).where(pd.notna(frame), None)
    return cleaned.to_dict(orient="records")
