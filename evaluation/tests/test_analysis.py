from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation.certification import (
    axp_summary,
    certification_summary,
    pruning_summary,
    search_summary,
    warning_summary,
)
from evaluation.dataset_analysis import analyse_datasets
from evaluation.io import EvaluationDataError, load_benchmark_folder
from evaluation.report import _certification_sentence
from evaluation.tests.conftest import create_benchmark


def test_dataset_summary_wins_ranks_and_critical_difference(
    benchmark_dir: Path,
) -> None:
    data = load_benchmark_folder(benchmark_dir)
    analysis = analyse_datasets(data.results)
    assert len(analysis.dataset_summary) == 3
    assert set(analysis.dataset_summary["best_method"]) == {
        "cals_compact_explain"
    }
    assert analysis.method_ranks["wins"].sum() == 3
    assert analysis.method_ranks.iloc[0]["method"] == "cals_compact_explain"
    assert analysis.critical_difference > 0.0


def test_certification_and_optional_diagnostic_summaries(
    benchmark_dir: Path,
) -> None:
    data = load_benchmark_folder(benchmark_dir)
    certification = certification_summary(data).set_index("audit_item")
    assert certification.loc["theorem_certified_rows", "count"] == 27
    for item in (
        "empirical_rows",
        "forbidden_predicates",
        "feature_label_leakage",
        "path_violations",
        "cached_subtree_violations",
        "empirical_fallbacks",
    ):
        assert certification.loc[item, "count"] == 0

    warnings = warning_summary(data)
    assert warnings["percentage"].sum() == pytest.approx(100.0)
    search = search_summary(data)
    assert search["greedy_nodes"].sum() == 81
    assert search["cache_hits"].sum() > 0
    pruning = pruning_summary(data)
    assert set(pruning["tree_reduction_percentage"]) == {50.0}
    axp = axp_summary(data)
    assert axp["family_usage"].str.contains("Horn").all()


def test_certification_rejects_stale_theorem_partition(
    benchmark_dir: Path,
) -> None:
    path = benchmark_dir / "full_results.csv"
    results = pd.read_csv(path)
    results.loc[0, "theorem_certified"] = False
    results.to_csv(path, index=False)

    with pytest.raises(
        EvaluationDataError,
        match="theorem_certified_results.csv.*does not match",
    ):
        certification_summary(load_benchmark_folder(benchmark_dir))


def test_certification_rejects_theorem_metrics_that_disagree_with_full_results(
    benchmark_dir: Path,
) -> None:
    path = benchmark_dir / "theorem_certified_results.csv"
    theorem = pd.read_csv(path)
    theorem.loc[0, "accuracy"] += 0.01
    theorem.to_csv(path, index=False)

    with pytest.raises(
        EvaluationDataError,
        match="theorem_certified_results.csv.*accuracy.*disagrees",
    ):
        certification_summary(load_benchmark_folder(benchmark_dir))


def test_certification_rejects_empirical_backend_even_when_boolean_flags_claim_allowed(
    benchmark_dir: Path,
) -> None:
    for filename in ("full_results.csv", "theorem_certified_results.csv"):
        path = benchmark_dir / filename
        frame = pd.read_csv(path)
        frame["backend"] = "StructuralHorn"
        frame.loc[0, "backend"] = "EmpiricalMixed"
        frame.to_csv(path, index=False)

    with pytest.raises(
        EvaluationDataError,
        match="theorem_certified_results.csv.*does not match",
    ):
        certification_summary(load_benchmark_folder(benchmark_dir))


def test_certification_rejects_mismatched_whitelisted_certificate_tuple(
    benchmark_dir: Path,
) -> None:
    for filename in ("full_results.csv", "theorem_certified_results.csv"):
        path = benchmark_dir / filename
        frame = pd.read_csv(path)
        frame["language_family"] = "SmartCertified"
        frame["backend"] = "PathCertified"
        frame["path_theory_state"] = "uncommitted|horn"
        frame["path_backend"] = "StructuralHorn"
        frame["axp_backend"] = "PathCertified"
        frame["path_certificate"] = "PathTheory"
        frame.loc[
            0,
            [
                "language_family",
                "backend",
                "path_theory_state",
                "path_backend",
                "axp_backend",
                "path_certificate",
            ],
        ] = [
            "Affine",
            "TwoSat",
            "affine_gf2",
            "Gf2Gaussian",
            "Gf2Gaussian",
            "AffineGf2",
        ]
        frame.to_csv(path, index=False)

    with pytest.raises(
        EvaluationDataError,
        match="theorem_certified_results.csv.*does not match",
    ):
        certification_summary(load_benchmark_folder(benchmark_dir))


def test_certification_rejects_malformed_present_audit_evidence(
    benchmark_dir: Path,
) -> None:
    path = benchmark_dir / "full_results.csv"
    results = pd.read_csv(path)
    results["path_violation_count"] = results["path_violation_count"].astype(object)
    results.loc[0, "path_violation_count"] = "not-a-number"
    results.to_csv(path, index=False)

    with pytest.raises(
        EvaluationDataError,
        match="path_violation_count.*non-negative integers",
    ):
        certification_summary(load_benchmark_folder(benchmark_dir))


def test_missing_optional_audit_evidence_is_reported_as_unverified(
    tmp_path: Path,
) -> None:
    benchmark = create_benchmark(tmp_path / "minimal", optional=False)
    path = benchmark / "full_results.csv"
    results = pd.read_csv(path).drop(
        columns=[
            "theorem_certified",
            "path_certified",
            "path_violation_count",
            "empirical_fallback_used",
            "incompatible_cached_subtree_reused",
            "all_predicates_backend_allowed",
        ]
    )
    results.to_csv(path, index=False)

    summary = certification_summary(load_benchmark_folder(benchmark))
    audit = summary.set_index("audit_item")
    for item in (
        "theorem_certified_rows",
        "empirical_rows",
        "forbidden_predicates",
        "feature_label_leakage",
        "path_violations",
        "cached_subtree_violations",
        "empirical_fallbacks",
    ):
        assert not audit.loc[item, "evidence_available"]
        assert audit.loc[item, "status"] == "not_audited"
        assert pd.isna(audit.loc[item, "count"])
    sentence = _certification_sentence(summary)
    assert "unverified theorem-certified rows" in sentence
    assert "unverified feature-label leakage findings" in sentence


def test_present_metadata_without_leakage_evidence_is_rejected(
    benchmark_dir: Path,
) -> None:
    path = benchmark_dir / "dataset_metadata.csv"
    metadata = pd.read_csv(path).drop(columns=["feature_equal_to_label_count"])
    metadata.to_csv(path, index=False)

    with pytest.raises(
        EvaluationDataError,
        match="missing feature-label leakage evidence",
    ):
        certification_summary(load_benchmark_folder(benchmark_dir))


def test_boolean_claims_without_certificate_tuple_are_not_audited(
    tmp_path: Path,
) -> None:
    benchmark = create_benchmark(tmp_path / "missing-tuple", optional=False)
    path = benchmark / "full_results.csv"
    results = pd.read_csv(path).drop(
        columns=[
            "language_family",
            "backend",
            "axp_backend",
            "path_certificate",
            "path_theory_state",
            "path_backend",
        ]
    )
    results.to_csv(path, index=False)

    audit = certification_summary(load_benchmark_folder(benchmark)).set_index(
        "audit_item"
    )
    assert not audit.loc["theorem_certified_rows", "evidence_available"]
    assert audit.loc["theorem_certified_rows", "status"] == "not_audited"


def test_partial_certificate_tuple_is_rejected(
    benchmark_dir: Path,
) -> None:
    path = benchmark_dir / "full_results.csv"
    results = pd.read_csv(path).drop(columns=["path_certificate"])
    results.to_csv(path, index=False)

    with pytest.raises(
        EvaluationDataError,
        match="incomplete textual certificate evidence.*path_certificate",
    ):
        certification_summary(load_benchmark_folder(benchmark_dir))


def test_leakage_count_and_boolean_flag_must_agree(
    benchmark_dir: Path,
) -> None:
    path = benchmark_dir / "dataset_metadata.csv"
    metadata = pd.read_csv(path)
    metadata["suspicious_feature_label_leakage"] = False
    metadata.loc[0, "suspicious_feature_label_leakage"] = True
    metadata.to_csv(path, index=False)

    with pytest.raises(
        EvaluationDataError,
        match="feature-label leakage count and flag disagree",
    ):
        certification_summary(load_benchmark_folder(benchmark_dir))


def test_leakage_audit_counts_every_leaking_feature(
    benchmark_dir: Path,
) -> None:
    path = benchmark_dir / "dataset_metadata.csv"
    metadata = pd.read_csv(path)
    metadata.loc[0, "feature_equal_to_label_count"] = 3
    metadata.to_csv(path, index=False)

    audit = certification_summary(load_benchmark_folder(benchmark_dir)).set_index(
        "audit_item"
    )
    assert audit.loc["feature_label_leakage", "count"] == 3
    assert audit.loc["feature_label_leakage", "status"] == "fail"


def test_leakage_count_above_int64_is_rejected(
    benchmark_dir: Path,
) -> None:
    path = benchmark_dir / "dataset_metadata.csv"
    metadata = pd.read_csv(
        path, dtype={"feature_equal_to_label_count": "object"}
    )
    metadata.loc[0, "feature_equal_to_label_count"] = "9223372036854775808"
    metadata.to_csv(path, index=False)

    with pytest.raises(
        EvaluationDataError,
        match="feature_equal_to_label_count.*non-negative integers",
    ):
        certification_summary(load_benchmark_folder(benchmark_dir))


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("path_certificate", "EmpiricalHeuristic"),
        ("path_theory_state", "invalid"),
        ("path_violation_count", 7),
        ("total_fit_time", 999.0),
        ("literals_after_prune", 999),
    ),
)
def test_theorem_partition_crossvalidates_all_critical_fields(
    benchmark_dir: Path,
    column: str,
    value: object,
) -> None:
    path = benchmark_dir / "theorem_certified_results.csv"
    theorem = pd.read_csv(path)
    theorem.loc[0, column] = value
    theorem.to_csv(path, index=False)

    with pytest.raises(
        EvaluationDataError,
        match="theorem_certified_results.csv.*disagrees",
    ):
        certification_summary(load_benchmark_folder(benchmark_dir))


def test_empirical_best_certified_is_rejected_from_theorem_without_being_forbidden(
    tmp_path: Path,
) -> None:
    benchmark = create_benchmark(tmp_path / "empirical-best", optional=False)
    full_path = benchmark / "full_results.csv"
    results = pd.read_csv(full_path)
    empirical = results["method"] == "smart_certified"
    results.loc[empirical, "method"] = "best-certified"
    results.loc[empirical, "theorem_certified"] = False
    results.loc[empirical, "language_family"] = "EmpiricalMixed"
    results.loc[empirical, "backend"] = "EmpiricalMixed"
    results.loc[empirical, "axp_backend"] = "EmpiricalMixed"
    results.loc[empirical, "path_certificate"] = "EmpiricalMixed"
    results.to_csv(full_path, index=False)
    results.loc[~empirical].to_csv(
        benchmark / "theorem_certified_results.csv", index=False
    )
    results.loc[empirical].to_csv(
        benchmark / "empirical_results.csv", index=False
    )

    audit = certification_summary(load_benchmark_folder(benchmark)).set_index(
        "audit_item"
    )
    assert audit.loc["theorem_certified_rows", "count"] == 18
    assert audit.loc["empirical_rows", "count"] == 9
    assert audit.loc["forbidden_predicates", "count"] == 0


@pytest.mark.parametrize(
    (
        "language",
        "backend",
        "path_certificate",
        "path_theory_state",
        "path_backend",
    ),
    (
        (
            "Horn",
            "StructuralHorn",
            "HornCnf",
            "uncommitted",
            "StructuralHorn",
        ),
        (
            "Horn",
            "StructuralHorn",
            "HornCnf",
            "uncommitted|horn",
            "StructuralHorn",
        ),
        (
            "AntiHorn",
            "StructuralAntiHorn",
            "AntiHornCnf",
            "uncommitted|antihorn",
            "StructuralHorn|StructuralAntiHorn",
        ),
        (
            "Square2Cnf",
            "TwoSat",
            "TwoCnf",
            "uncommitted|two_sat",
            "StructuralHorn|TwoSat",
        ),
        (
            "Affine",
            "Gf2Gaussian",
            "AffineGf2",
            "uncommitted|affine_gf2",
            "StructuralHorn|Gf2Gaussian",
        ),
    ),
)
def test_single_family_certificate_accepts_uncommitted_path_states(
    benchmark_dir: Path,
    language: str,
    backend: str,
    path_certificate: str,
    path_theory_state: str,
    path_backend: str,
) -> None:
    for filename in ("full_results.csv", "theorem_certified_results.csv"):
        path = benchmark_dir / filename
        frame = pd.read_csv(path)
        frame["language_family"] = language
        frame["backend"] = backend
        frame["axp_backend"] = backend
        frame["path_certificate"] = path_certificate
        frame["path_theory_state"] = path_theory_state
        frame["path_backend"] = path_backend
        frame.to_csv(path, index=False)

    audit = certification_summary(load_benchmark_folder(benchmark_dir)).set_index(
        "audit_item"
    )
    assert audit.loc["theorem_certified_rows", "count"] == 27
    assert audit.loc["empirical_rows", "count"] == 0


def test_single_family_certificate_rejects_state_backend_mismatch(
    benchmark_dir: Path,
) -> None:
    for filename in ("full_results.csv", "theorem_certified_results.csv"):
        path = benchmark_dir / filename
        frame = pd.read_csv(path)
        frame["language_family"] = "AntiHorn"
        frame["backend"] = "StructuralAntiHorn"
        frame["axp_backend"] = "StructuralAntiHorn"
        frame["path_certificate"] = "AntiHornCnf"
        frame["path_theory_state"] = "uncommitted|antihorn"
        frame["path_backend"] = "StructuralAntiHorn"
        frame.to_csv(path, index=False)

    with pytest.raises(
        EvaluationDataError,
        match="theorem_certified_results.csv.*does not match",
    ):
        certification_summary(load_benchmark_folder(benchmark_dir))
