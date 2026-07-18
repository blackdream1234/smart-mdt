from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from evaluation.config import EvaluationConfig, FIGURE_NAMES
from evaluation.report import build_report
from evaluation.tables import dataframe_to_latex
from evaluation.tests.conftest import create_benchmark


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_latex_generation_uses_booktabs_caption_label_and_three_decimals() -> None:
    frame = pd.DataFrame({"Method": ["CALS_MDT"], "Mean": [0.123456]})
    rendered = dataframe_to_latex(
        frame, caption="A caption", label="tab:test_table"
    )
    assert r"\toprule" in rendered
    assert r"\bottomrule" in rendered
    assert r"\caption{A caption}" in rendered
    assert r"\label{tab:test\_table}" in rendered
    assert "0.123" in rendered
    assert r"CALS\_MDT" in rendered


def test_end_to_end_report_generates_every_output_deterministically(
    benchmark_dir: Path,
    tmp_path: Path,
) -> None:
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first = build_report(
        EvaluationConfig(
            benchmark_dir,
            first_output,
            bootstrap_resamples=80,
            seed=123,
        )
    )
    second = build_report(
        EvaluationConfig(
            benchmark_dir,
            second_output,
            bootstrap_resamples=80,
            seed=123,
        )
    )
    assert len(first.figures) == 2 * len(FIGURE_NAMES)
    for name in FIGURE_NAMES:
        assert (first_output / "figures" / f"{name}.png").is_file()
        assert (first_output / "figures" / f"{name}.pdf").is_file()
    for filename in (
        "statistics.csv",
        "statistics.tex",
        "statistics.md",
        "significance.csv",
        "significance.tex",
        "dataset_summary.csv",
        "dataset_summary.tex",
        "certification_summary.tex",
        "warnings_summary.csv",
        "search_summary.tex",
        "pruning_summary.tex",
        "axp_summary.tex",
    ):
        assert (first_output / "tables" / filename).is_file()
    for filename in (
        "evaluation_report.md",
        "evaluation_report.tex",
        "executive_summary.md",
        "executive_summary.tex",
        "reproducibility_manifest.json",
    ):
        assert (first_output / "report" / filename).is_file()

    deterministic_paths = (
        "tables/statistics.csv",
        "tables/significance.csv",
        "figures/accuracy_boxplot.png",
        "figures/accuracy_boxplot.pdf",
        "report/evaluation_report.md",
        "report/evaluation_report.tex",
        "report/reproducibility_manifest.json",
    )
    for relative in deterministic_paths:
        assert _digest(first_output / relative) == _digest(second_output / relative)
    assert len(first.tables) == len(second.tables)


def test_end_to_end_report_handles_missing_optional_files(tmp_path: Path) -> None:
    benchmark = create_benchmark(tmp_path / "minimal", optional=False)
    output = tmp_path / "minimal-output"
    build_report(
        EvaluationConfig(
            benchmark,
            output,
            bootstrap_resamples=20,
            seed=5,
        )
    )
    report = (output / "report" / "evaluation_report.md").read_text(
        encoding="utf-8"
    )
    assert "search_diagnostics.csv" in report
    search_table = (output / "tables" / "search_summary.tex").read_text(
        encoding="utf-8"
    )
    assert "No data available" in search_table
    assert (output / "figures" / "search_diagnostics.pdf").is_file()
