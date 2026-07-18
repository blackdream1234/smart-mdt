from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluation.io import EvaluationDataError, load_benchmark_folder
from evaluation.tests.conftest import create_benchmark


def test_loads_and_canonicalizes_metrics(benchmark_dir: Path) -> None:
    data = load_benchmark_folder(benchmark_dir)
    assert len(data.results) == 27
    assert {"predicate_literals", "fit_time_seconds"}.issubset(data.results.columns)
    assert data.missing_optional_files == ()
    assert data.results.duplicated(["dataset", "run", "depth", "method"]).sum() == 0


def test_missing_optional_files_are_detected(tmp_path: Path) -> None:
    root = create_benchmark(tmp_path / "minimal", optional=False)
    data = load_benchmark_folder(root)
    assert "search_diagnostics.csv" in data.missing_optional_files
    assert "README_RESULTS.md" in data.missing_optional_files


def test_invalid_optional_diagnostic_fails_informatively(
    benchmark_dir: Path,
) -> None:
    path = benchmark_dir / "search_diagnostics.csv"
    frame = pd.read_csv(path)
    frame["search_time"] = frame["search_time"].astype(object)
    frame.loc[0, "search_time"] = "not-a-number"
    frame.to_csv(path, index=False)
    with pytest.raises(
        EvaluationDataError,
        match="search_diagnostics.csv.*search_time.*numeric",
    ):
        load_benchmark_folder(benchmark_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("duplicate", "duplicate"),
        ("invalid_method", "invalid method names"),
        ("invalid_numeric", "must be numeric"),
        ("missing_column", "missing required columns"),
        ("missing_value", "missing numeric values"),
    ),
)
def test_invalid_full_results_fail_informatively(
    benchmark_dir: Path,
    mutation: str,
    message: str,
) -> None:
    path = benchmark_dir / "full_results.csv"
    frame = pd.read_csv(path)
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif mutation == "invalid_method":
        frame.loc[0, "method"] = "not_a_method"
    elif mutation == "invalid_numeric":
        frame["accuracy"] = frame["accuracy"].astype(object)
        frame.loc[0, "accuracy"] = "invalid"
    elif mutation == "missing_column":
        frame = frame.drop(columns=["tree_nodes"])
    elif mutation == "missing_value":
        frame.loc[0, "accuracy"] = None
    frame.to_csv(path, index=False)
    with pytest.raises(EvaluationDataError, match=message):
        load_benchmark_folder(benchmark_dir)
