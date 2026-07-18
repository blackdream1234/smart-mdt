"""Validated loading of Smart-MDT benchmark folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from .config import METHOD_LABELS, OPTIONAL_FILES, OPTIONAL_TEXT_FILES, REQUIRED_FILES
from .utils import safe_numeric


class EvaluationDataError(ValueError):
    """Raised when benchmark data cannot support a valid evaluation."""


@dataclass(frozen=True)
class BenchmarkData:
    """Validated input frames and canonical full results."""

    input_dir: Path
    frames: Mapping[str, pd.DataFrame]
    results: pd.DataFrame
    missing_optional_files: tuple[str, ...]

    def optional(self, filename: str) -> pd.DataFrame | None:
        return self.frames.get(filename)


RESULT_REQUIRED_COLUMNS = (
    "dataset",
    "run",
    "depth",
    "method",
    "accuracy",
    "tree_nodes",
    "mean_axp_length",
)

RESULT_NUMERIC_COLUMNS = (
    "run",
    "depth",
    "accuracy",
    "tree_nodes",
    "mean_axp_length",
)

RESULT_KEY = ("dataset", "run", "depth", "method")

OPTIONAL_SCHEMAS: dict[str, dict[str, tuple[str, ...]]] = {
    "summary_by_method.csv": {
        "required": (
            "method",
            "rows",
            "accuracy_mean",
            "tree_nodes_mean",
            "mean_axp_length_mean",
        ),
        "numeric": (
            "rows",
            "accuracy_mean",
            "tree_nodes_mean",
            "mean_axp_length_mean",
        ),
        "key": ("method",),
    },
    "theorem_certified_results.csv": {
        "required": RESULT_REQUIRED_COLUMNS,
        "numeric": RESULT_NUMERIC_COLUMNS,
        "key": RESULT_KEY,
    },
    "empirical_results.csv": {
        "required": RESULT_REQUIRED_COLUMNS,
        "numeric": RESULT_NUMERIC_COLUMNS,
        "key": RESULT_KEY,
    },
    "axp_metadata.csv": {
        "required": RESULT_REQUIRED_COLUMNS,
        "numeric": RESULT_NUMERIC_COLUMNS,
        "key": RESULT_KEY,
    },
    "tuning_diagnostics.csv": {
        "required": RESULT_REQUIRED_COLUMNS,
        "numeric": RESULT_NUMERIC_COLUMNS,
        "key": RESULT_KEY,
    },
    "benchmark_warnings.csv": {
        "required": ("dataset", "method", "warning_type", "affected_rows"),
        "numeric": ("affected_rows",),
        "key": (),
    },
    "search_diagnostics.csv": {
        "required": (
            *RESULT_KEY,
            "nodes_using_greedy_selection",
            "nodes_using_selective_lookahead",
            "branch_and_bound_activation_count",
            "branch_and_bound_avoided_count",
            "cache_activation_count",
            "estimated_work_saved",
            "search_time",
        ),
        "numeric": (
            "run",
            "depth",
            "nodes_using_greedy_selection",
            "nodes_using_selective_lookahead",
            "branch_and_bound_activation_count",
            "branch_and_bound_avoided_count",
            "cache_activation_count",
            "estimated_work_saved",
            "search_time",
        ),
        "key": RESULT_KEY,
    },
    "cache_diagnostics.csv": {
        "required": (
            *RESULT_KEY,
            "predicate_mask_hits",
            "predicate_mask_misses",
            "candidate_hits",
            "candidate_misses",
            "subtree_hits",
            "subtree_misses",
        ),
        "numeric": (
            "run",
            "depth",
            "predicate_mask_hits",
            "predicate_mask_misses",
            "candidate_hits",
            "candidate_misses",
            "subtree_hits",
            "subtree_misses",
        ),
        "key": RESULT_KEY,
    },
    "family_budget_diagnostics.csv": {
        "required": (*RESULT_KEY, "compatible_family_count"),
        "numeric": ("run", "depth", "compatible_family_count"),
        "key": RESULT_KEY,
    },
    "pruning_diagnostics.csv": {
        "required": (
            *RESULT_KEY,
            "nodes_before",
            "nodes_after",
            "validation_accuracy_before",
            "validation_accuracy_after",
            "validation_balanced_accuracy_before",
            "validation_balanced_accuracy_after",
            "validation_minority_recall_before",
            "validation_minority_recall_after",
        ),
        "numeric": (
            "run",
            "depth",
            "nodes_before",
            "nodes_after",
            "validation_accuracy_before",
            "validation_accuracy_after",
            "validation_balanced_accuracy_before",
            "validation_balanced_accuracy_after",
            "validation_minority_recall_before",
            "validation_minority_recall_after",
        ),
        "key": RESULT_KEY,
    },
    "beam_diagnostics.csv": {
        "required": (
            *RESULT_KEY,
            "candidate_beam_width",
            "tree_beam_width",
            "lookahead_depth",
            "node_budget",
            "total_fit_time",
        ),
        "numeric": (
            "run",
            "depth",
            "candidate_beam_width",
            "tree_beam_width",
            "lookahead_depth",
            "node_budget",
            "total_fit_time",
        ),
        "key": RESULT_KEY,
    },
}


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=True)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise EvaluationDataError(f"failed to read {path}: {error}") from error


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], filename: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise EvaluationDataError(
            f"{filename} is missing required columns: {', '.join(missing)}"
        )


def _select_alias(
    frame: pd.DataFrame,
    target: str,
    aliases: tuple[str, ...],
    filename: str,
) -> pd.Series:
    for alias in aliases:
        if alias in frame.columns:
            return safe_numeric(frame[alias], column=alias)
    raise EvaluationDataError(
        f"{filename} requires one of these columns for {target}: {', '.join(aliases)}"
    )


def _validate_results(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    _require_columns(frame, RESULT_REQUIRED_COLUMNS, filename)
    if frame.empty:
        raise EvaluationDataError(f"{filename} contains no benchmark rows")

    result = frame.copy()
    for column in ("dataset", "method"):
        if result[column].isna().any() or (result[column].astype(str).str.strip() == "").any():
            raise EvaluationDataError(f"{filename} column {column!r} contains missing values")
        result[column] = result[column].astype(str).str.strip()

    invalid_methods = sorted(set(result["method"]).difference(METHOD_LABELS))
    if invalid_methods:
        raise EvaluationDataError(
            f"{filename} contains invalid method names: {', '.join(invalid_methods)}"
        )

    for column in RESULT_NUMERIC_COLUMNS:
        try:
            result[column] = safe_numeric(result[column], column=column)
        except ValueError as error:
            raise EvaluationDataError(f"{filename}: {error}") from error

    duplicate_mask = result.duplicated(list(RESULT_KEY), keep=False)
    if duplicate_mask.any():
        example = result.loc[duplicate_mask, list(RESULT_KEY)].iloc[0].to_dict()
        raise EvaluationDataError(
            f"{filename} contains duplicate dataset/run/depth/method rows; "
            f"first duplicate: {example}"
        )

    try:
        result["predicate_literals"] = _select_alias(
            result,
            "predicate_literals",
            ("predicate_literals", "literals_after_prune", "literals_before_prune"),
            filename,
        )
        result["fit_time_seconds"] = _select_alias(
            result,
            "fit_time_seconds",
            ("fit_time_seconds", "total_fit_time", "train_time"),
            filename,
        )
    except ValueError as error:
        raise EvaluationDataError(str(error)) from error

    for column in ("accuracy", "tree_nodes", "predicate_literals", "mean_axp_length"):
        if result[column].isna().any():
            raise EvaluationDataError(f"{filename} column {column!r} contains missing values")

    if ((result["accuracy"] < 0.0) | (result["accuracy"] > 1.0)).any():
        raise EvaluationDataError(f"{filename} accuracy values must lie in [0, 1]")
    if (result[["tree_nodes", "predicate_literals", "mean_axp_length"]] < 0).any().any():
        raise EvaluationDataError(f"{filename} contains negative complexity values")
    if (result["fit_time_seconds"] < 0).any():
        raise EvaluationDataError(f"{filename} contains negative fit times")

    return result.sort_values(list(RESULT_KEY), kind="mergesort").reset_index(drop=True)


def _validate_dataset_metadata(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    _require_columns(frame, ("dataset",), filename)
    if frame["dataset"].isna().any():
        raise EvaluationDataError(f"{filename} contains a missing dataset name")
    if frame["dataset"].duplicated().any():
        raise EvaluationDataError(f"{filename} contains duplicate dataset rows")
    return frame.sort_values("dataset", kind="mergesort").reset_index(drop=True)


def _validate_optional_frame(
    frame: pd.DataFrame,
    filename: str,
) -> pd.DataFrame:
    schema = OPTIONAL_SCHEMAS.get(filename)
    if schema is None:
        return frame
    required = schema["required"]
    _require_columns(frame, required, filename)
    if frame.empty:
        return frame

    result = frame.copy()
    for column in ("dataset", "method"):
        if column not in result:
            continue
        if result[column].isna().any() or (result[column].astype(str).str.strip() == "").any():
            raise EvaluationDataError(f"{filename} column {column!r} contains missing values")
        result[column] = result[column].astype(str).str.strip()

    if "method" in result:
        allowed = set(METHOD_LABELS)
        if filename == "benchmark_warnings.csv":
            allowed.add("all")
        invalid_methods = sorted(set(result["method"]).difference(allowed))
        if invalid_methods:
            raise EvaluationDataError(
                f"{filename} contains invalid method names: "
                + ", ".join(invalid_methods)
            )

    for column in schema["numeric"]:
        try:
            result[column] = safe_numeric(result[column], column=column)
        except ValueError as error:
            raise EvaluationDataError(f"{filename}: {error}") from error

    key = schema["key"]
    if key and result.duplicated(list(key), keep=False).any():
        example = result.loc[
            result.duplicated(list(key), keep=False), list(key)
        ].iloc[0]
        raise EvaluationDataError(
            f"{filename} contains duplicate key rows; first duplicate: "
            f"{example.to_dict()}"
        )
    if key:
        return result.sort_values(list(key), kind="mergesort").reset_index(drop=True)
    return result


def load_benchmark_folder(input_dir: str | Path) -> BenchmarkData:
    """Load a benchmark directory, validating required and available data."""

    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise EvaluationDataError(f"benchmark input directory does not exist: {root}")

    missing_required = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing_required:
        raise EvaluationDataError(
            "benchmark folder is missing required files: " + ", ".join(missing_required)
        )

    frames: dict[str, pd.DataFrame] = {}
    for filename in (*REQUIRED_FILES, *OPTIONAL_FILES):
        path = root / filename
        if path.is_file():
            frames[filename] = _read_csv(path)

    results = _validate_results(frames["full_results.csv"], "full_results.csv")
    frames["full_results.csv"] = results

    metadata = frames.get("dataset_metadata.csv")
    if metadata is not None:
        frames["dataset_metadata.csv"] = _validate_dataset_metadata(
            metadata, "dataset_metadata.csv"
        )
    for filename, frame in tuple(frames.items()):
        if filename in OPTIONAL_SCHEMAS:
            frames[filename] = _validate_optional_frame(frame, filename)

    missing_optional = tuple(
        filename
        for filename in (*OPTIONAL_FILES, *OPTIONAL_TEXT_FILES)
        if not (root / filename).is_file()
    )
    return BenchmarkData(root, frames, results, missing_optional)
