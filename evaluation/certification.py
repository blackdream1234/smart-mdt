"""Certification, warning, search, pruning, and AXp audits."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

from .io import BenchmarkData, EvaluationDataError, RESULT_KEY
from .utils import (
    method_label,
    ordered_methods,
    parse_bool_series,
    safe_nonnegative_integers,
    safe_numeric,
)

CERTIFIED_BACKENDS = {
    "StructuralHorn",
    "StructuralAntiHorn",
    "TwoSat",
    "Gf2Gaussian",
    "PathCertified",
}
CERTIFIED_LANGUAGES = {
    "Unary",
    "Horn",
    "AntiHorn",
    "Square2Cnf",
    "Affine",
    "SmartCertified",
}
CERTIFIED_PATH_CERTIFICATES = {
    "HornCnf",
    "AntiHornCnf",
    "TwoCnf",
    "AffineGf2",
    "PathTheory",
}
SINGLE_FAMILY_CERTIFICATES = {
    "Unary": (
        "StructuralHorn",
        "StructuralHorn",
        "HornCnf",
        "uncommitted",
    ),
    "Horn": (
        "StructuralHorn",
        "StructuralHorn",
        "HornCnf",
        "horn",
    ),
    "AntiHorn": (
        "StructuralAntiHorn",
        "StructuralAntiHorn",
        "AntiHornCnf",
        "antihorn",
    ),
    "Square2Cnf": (
        "TwoSat",
        "TwoSat",
        "TwoCnf",
        "two_sat",
    ),
    "Affine": (
        "Gf2Gaussian",
        "Gf2Gaussian",
        "AffineGf2",
        "affine_gf2",
    ),
}
PATH_STATE_BACKENDS = {
    "uncommitted": "StructuralHorn",
    "horn": "StructuralHorn",
    "antihorn": "StructuralAntiHorn",
    "two_sat": "TwoSat",
    "affine_gf2": "Gf2Gaussian",
}
TEXTUAL_CERTIFICATE_COLUMNS = (
    "language_family",
    "backend",
    "axp_backend",
    "path_certificate",
    "path_theory_state",
    "path_backend",
)
BOOLEAN_CERTIFICATE_COLUMNS = (
    "theorem_certified",
    "path_certified",
    "all_predicates_backend_allowed",
    "incompatible_cached_subtree_reused",
    "empirical_fallback_used",
)
COUNT_CERTIFICATE_COLUMNS = ("path_violation_count",)


def _strict_bool_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series | None:
    if column not in frame:
        return None
    raw = frame[column]
    if raw.isna().any() or (raw.astype(str).str.strip() == "").any():
        raise EvaluationDataError(
            f"audit column {column!r} contains missing boolean evidence"
        )
    try:
        return parse_bool_series(raw, column=column)
    except ValueError as error:
        raise EvaluationDataError(str(error)) from error


def _strict_count_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series | None:
    if column not in frame:
        return None
    try:
        return safe_nonnegative_integers(frame[column], column=column)
    except ValueError as error:
        raise EvaluationDataError(str(error)) from error


def _textual_certificate_eligibility(
    results: pd.DataFrame,
) -> pd.Series | None:
    required_columns = set(TEXTUAL_CERTIFICATE_COLUMNS)
    textual_columns = required_columns.intersection(results.columns)
    if not textual_columns:
        return None
    if textual_columns != required_columns:
        missing = sorted(required_columns.difference(textual_columns))
        raise EvaluationDataError(
            "full_results.csv contains incomplete textual certificate evidence; "
            f"missing: {', '.join(missing)}"
        )

    def components(value: object, allowed: set[str]) -> frozenset[str] | None:
        parts = str(value).strip().split("|")
        if (
            not parts
            or any(not part or part not in allowed for part in parts)
            or len(parts) != len(set(parts))
        ):
            return None
        return frozenset(parts)

    def compatible(row: pd.Series) -> bool:
        language = (
            str(row["language_family"]).strip()
            if "language_family" in textual_columns
            else None
        )
        if language is not None and language not in CERTIFIED_LANGUAGES:
            return False

        if language in SINGLE_FAMILY_CERTIFICATES:
            expected = SINGLE_FAMILY_CERTIFICATES[language]
            for column, value in zip(
                (
                    "backend",
                    "axp_backend",
                    "path_certificate",
                ),
                expected[:3],
            ):
                if str(row[column]).strip() != value:
                    return False
            states = components(
                row["path_theory_state"], set(PATH_STATE_BACKENDS)
            )
            if states is None:
                return False
            committed_state = expected[3]
            allowed_states = (
                frozenset({"uncommitted"})
                if language == "Unary"
                else frozenset({"uncommitted", committed_state})
            )
            if not states.issubset(allowed_states):
                return False
            backends = components(row["path_backend"], CERTIFIED_BACKENDS)
            if backends is None:
                return False
            expected_backends = frozenset(
                PATH_STATE_BACKENDS[state] for state in states
            )
            return backends == expected_backends

        if language == "SmartCertified":
            for column, value in (
                ("backend", "PathCertified"),
                ("axp_backend", "PathCertified"),
                ("path_certificate", "PathTheory"),
            ):
                if column in textual_columns and str(row[column]).strip() != value:
                    return False
            states = (
                components(row["path_theory_state"], set(PATH_STATE_BACKENDS))
                if "path_theory_state" in textual_columns
                else None
            )
            if "path_theory_state" in textual_columns and states is None:
                return False
            backends = (
                components(row["path_backend"], CERTIFIED_BACKENDS)
                if "path_backend" in textual_columns
                else None
            )
            if "path_backend" in textual_columns and backends is None:
                return False
            if states is not None and backends is not None:
                expected_backends = frozenset(PATH_STATE_BACKENDS[state] for state in states)
                if backends != expected_backends:
                    return False
            return True

        return False

    return results.apply(compatible, axis=1)


def _result_eligibility(results: pd.DataFrame) -> pd.Series | None:
    theorem = _strict_bool_series(results, "theorem_certified")
    path = _strict_bool_series(results, "path_certified")
    violations = _strict_count_series(results, "path_violation_count")
    allowed = _strict_bool_series(results, "all_predicates_backend_allowed")
    cached = _strict_bool_series(results, "incompatible_cached_subtree_reused")
    fallback = _strict_bool_series(results, "empirical_fallback_used")
    evidence = (theorem, path, violations, allowed, cached, fallback)
    if any(value is None for value in evidence):
        return None
    assert theorem is not None
    assert path is not None
    assert violations is not None
    assert allowed is not None
    assert cached is not None
    assert fallback is not None
    if not (path.to_numpy() == (violations.to_numpy() == 0)).all():
        raise EvaluationDataError(
            "path_certified is inconsistent with path_violation_count"
        )
    eligible = theorem & path & violations.eq(0) & allowed & ~cached & ~fallback
    textual = _textual_certificate_eligibility(results)
    if textual is None:
        return None
    return eligible & textual


def _validate_result_partition(
    name: str,
    table: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    actual_keys = set(map(tuple, table.loc[:, RESULT_KEY].to_numpy()))
    expected_keys = set(map(tuple, expected.loc[:, RESULT_KEY].to_numpy()))
    if actual_keys != expected_keys:
        missing = len(expected_keys.difference(actual_keys))
        extra = len(actual_keys.difference(expected_keys))
        raise EvaluationDataError(
            f"{name} does not match full_results.csv certification partition "
            f"({missing} missing keys, {extra} extra keys)"
        )

    actual = table.sort_values(list(RESULT_KEY), kind="mergesort").reset_index(drop=True)
    canonical = expected.sort_values(list(RESULT_KEY), kind="mergesort").reset_index(
        drop=True
    )

    metric_aliases = {
        "accuracy": ("accuracy",),
        "tree_nodes": ("tree_nodes",),
        "mean_axp_length": ("mean_axp_length",),
        "predicate_literals": (
            "predicate_literals",
            "literals_after_prune",
            "literals_before_prune",
        ),
        "fit_time_seconds": (
            "fit_time_seconds",
            "total_fit_time",
            "train_time",
        ),
    }
    for metric, aliases in metric_aliases.items():
        actual_column = next(
            (column for column in aliases if column in actual.columns), None
        )
        if actual_column is None:
            raise EvaluationDataError(
                f"{name} is missing benchmark metric {metric!r}"
            )
        try:
            left = safe_numeric(
                actual[actual_column], column=actual_column
            ).to_numpy(dtype=float)
        except ValueError as error:
            raise EvaluationDataError(f"{name}: {error}") from error
        right = canonical[metric].to_numpy(dtype=float)
        if not np.array_equal(left, right):
            raise EvaluationDataError(
                f"{name} column {metric!r} disagrees with full_results.csv"
            )

    required_audit_columns = (
        *TEXTUAL_CERTIFICATE_COLUMNS,
        *BOOLEAN_CERTIFICATE_COLUMNS,
        *COUNT_CERTIFICATE_COLUMNS,
    )
    missing = sorted(set(required_audit_columns).difference(actual.columns))
    if missing:
        raise EvaluationDataError(
            f"{name} is missing certification audit columns: {', '.join(missing)}"
        )

    for column in TEXTUAL_CERTIFICATE_COLUMNS:
        left = actual[column].astype(str).str.strip().to_numpy()
        right = canonical[column].astype(str).str.strip().to_numpy()
        if not np.array_equal(left, right):
            raise EvaluationDataError(
                f"{name} audit column {column!r} disagrees with full_results.csv"
            )
    for column in BOOLEAN_CERTIFICATE_COLUMNS:
        left = _strict_bool_series(actual, column)
        right = _strict_bool_series(canonical, column)
        assert left is not None
        assert right is not None
        if not np.array_equal(left.to_numpy(), right.to_numpy()):
            raise EvaluationDataError(
                f"{name} audit column {column!r} disagrees with full_results.csv"
            )
    for column in COUNT_CERTIFICATE_COLUMNS:
        left = _strict_count_series(actual, column)
        right = _strict_count_series(canonical, column)
        assert left is not None
        assert right is not None
        if not np.array_equal(left.to_numpy(), right.to_numpy()):
            raise EvaluationDataError(
                f"{name} audit column {column!r} disagrees with full_results.csv"
            )


def _metadata_audit(
    metadata: pd.DataFrame | None,
    result_datasets: set[str],
) -> tuple[int | None, int | None]:
    if metadata is None:
        return None, None
    names = metadata["dataset"].astype(str).str.strip()
    if (names == "").any():
        raise EvaluationDataError("dataset_metadata.csv contains a blank dataset name")
    if names.duplicated().any():
        raise EvaluationDataError(
            "dataset_metadata.csv contains duplicate normalized dataset names"
        )
    if not result_datasets.issubset(set(names)):
        missing = sorted(result_datasets.difference(set(names)))
        raise EvaluationDataError(
            "dataset_metadata.csv is missing benchmark datasets: "
            + ", ".join(missing)
        )

    skipped = _strict_bool_series(metadata, "skipped")
    if skipped is None:
        raise EvaluationDataError(
            "dataset_metadata.csv is missing audit column 'skipped'"
        )
    evaluated_datasets = set(names.loc[~skipped])
    if evaluated_datasets != result_datasets:
        missing = sorted(evaluated_datasets.difference(result_datasets))
        extra = sorted(result_datasets.difference(evaluated_datasets))
        raise EvaluationDataError(
            "dataset_metadata.csv evaluated-dataset partition disagrees with "
            f"full_results.csv ({len(missing)} missing result datasets, "
            f"{len(extra)} result datasets marked skipped)"
        )

    suspicious = _strict_bool_series(
        metadata, "suspicious_feature_label_leakage"
    )
    leakage_counts = _strict_count_series(metadata, "feature_equal_to_label_count")
    if leakage_counts is not None:
        if suspicious is not None and not np.array_equal(
            suspicious.to_numpy(), leakage_counts.gt(0).to_numpy()
        ):
            raise EvaluationDataError(
                "dataset_metadata.csv feature-label leakage count and flag disagree"
            )
        leakage = sum(int(value) for value in leakage_counts)
    else:
        if suspicious is None:
            raise EvaluationDataError(
                "dataset_metadata.csv is missing feature-label leakage evidence"
            )
        leakage = int(suspicious.sum())
    return int(skipped.sum()), leakage


def _audit_record(
    name: str,
    count: int | None,
    denominator: int,
    *,
    violation: bool = False,
) -> dict[str, object]:
    available = count is not None
    if not available:
        status = "not_audited"
        percentage = np.nan
        reported_count: float | int = np.nan
    else:
        status = "fail" if violation and count else "pass"
        percentage = 100.0 * count / denominator if denominator else 0.0
        reported_count = int(count)
    return {
        "audit_item": name,
        "count": reported_count,
        "denominator": int(denominator),
        "percentage": percentage,
        "evidence_available": available,
        "status": status,
    }


def certification_summary(data: BenchmarkData) -> pd.DataFrame:
    """Summarise every hard certification and data-integrity boundary."""

    results = data.results
    metadata = data.optional("dataset_metadata.csv")
    theorem = data.optional("theorem_certified_results.csv")
    empirical = data.optional("empirical_results.csv")

    dataset_count = int(results["dataset"].nunique())
    skipped, leakage = _metadata_audit(
        metadata, set(results["dataset"].astype(str))
    )

    eligibility = _result_eligibility(results)
    if eligibility is None:
        if theorem is not None or empirical is not None:
            raise EvaluationDataError(
                "full_results.csv lacks hard audit fields required to validate "
                "theorem/empirical result tables"
            )
        theorem_rows = None
        empirical_rows = None
    else:
        certified_results = results.loc[eligibility]
        rejected_results = results.loc[~eligibility]
        if theorem is not None:
            _validate_result_partition(
                "theorem_certified_results.csv", theorem, certified_results
            )
        if empirical is not None:
            _validate_result_partition(
                "empirical_results.csv", empirical, rejected_results
            )
        theorem_rows = int(eligibility.sum())
        empirical_rows = int((~eligibility).sum())

    path = _strict_bool_series(results, "path_certified")
    violations = _strict_count_series(results, "path_violation_count")
    if path is not None and violations is not None:
        if not (path.to_numpy() == (violations.to_numpy() == 0)).all():
            raise EvaluationDataError(
                "path_certified is inconsistent with path_violation_count"
            )
        path_violations: int | None = int(violations.sum())
    elif path is not None:
        path_violations = int((~path).sum())
    elif violations is not None:
        path_violations = int(violations.sum())
    else:
        path_violations = None

    allowed = _strict_bool_series(results, "all_predicates_backend_allowed")
    cached = _strict_bool_series(results, "incompatible_cached_subtree_reused")
    fallback = _strict_bool_series(results, "empirical_fallback_used")
    _textual_certificate_eligibility(results)
    if allowed is not None:
        forbidden = int((~allowed).sum())
    else:
        forbidden = None
    cached_violations = int(cached.sum()) if cached is not None else None
    empirical_fallbacks = int(fallback.sum()) if fallback is not None else None

    records = [
        _audit_record("datasets", dataset_count, dataset_count),
        _audit_record("rows", len(results), len(results)),
        _audit_record("theorem_certified_rows", theorem_rows, len(results)),
        _audit_record("empirical_rows", empirical_rows, len(results), violation=True),
        _audit_record("forbidden_predicates", forbidden, len(results), violation=True),
        _audit_record(
            "feature_label_leakage",
            leakage,
            max(dataset_count, 1),
            violation=True,
        ),
        _audit_record(
            "path_violations", path_violations, len(results), violation=True
        ),
        _audit_record(
            "cached_subtree_violations",
            cached_violations,
            len(results),
            violation=True,
        ),
        _audit_record(
            "empirical_fallbacks",
            empirical_fallbacks,
            len(results),
            violation=True,
        ),
        _audit_record(
            "skipped_datasets",
            skipped,
            max(dataset_count + (skipped or 0), 1),
            violation=True,
        ),
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
