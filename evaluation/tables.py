"""Deterministic CSV, Markdown, and booktabs LaTeX table generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: object) -> str:
    text = str(value)
    return "".join(LATEX_REPLACEMENTS.get(character, character) for character in text)


def _format_value(value: object, *, latex: bool = False) -> str:
    if value is None or pd.isna(value):
        return "--"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isinf(number):
            return (
                (r"$+\infty$" if number > 0 else r"$-\infty$")
                if latex
                else ("+inf" if number > 0 else "-inf")
            )
        return f"{number:.3f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    text = str(value)
    return latex_escape(text) if latex else text


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        frame = pd.DataFrame({"Status": ["No data available"]})
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [_format_value(value).replace("|", r"\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def dataframe_to_latex(
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
    longtable: bool = False,
) -> str:
    """Render a frame without relying on optional pandas formatting packages."""

    if frame.empty:
        frame = pd.DataFrame({"Status": ["No data available"]})
    columns = list(frame.columns)
    numeric = [
        pd.api.types.is_numeric_dtype(frame[column].dtype) for column in columns
    ]
    alignment = "".join("r" if is_numeric else "l" for is_numeric in numeric)
    header = " & ".join(latex_escape(column) for column in columns) + r" \\"
    rows = [
        " & ".join(_format_value(value, latex=True) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]

    if longtable:
        lines = [
            r"\begingroup",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\begin{{longtable}}{{{alignment}}}",
            rf"\caption{{{latex_escape(caption)}}}\label{{{latex_escape(label)}}}\\",
            r"\toprule",
            header,
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            header,
            r"\midrule",
            r"\endhead",
            *rows,
            r"\bottomrule",
            r"\end{longtable}",
            r"\endgroup",
        ]
    else:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{latex_escape(label)}}}",
            r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    return "\n".join(lines) + "\n"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def _select(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    selected = [column for column in columns if column in frame.columns]
    return frame.loc[:, selected].copy()


def write_evaluation_tables(
    output_dir: Path,
    *,
    statistics: pd.DataFrame,
    significance: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    method_ranks: pd.DataFrame,
    certification: pd.DataFrame,
    warnings: pd.DataFrame,
    search: pd.DataFrame,
    pruning: pd.DataFrame,
    axp: pd.DataFrame,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    statistics_display = _select(
        statistics,
        (
            "method_label",
            "metric_label",
            "n",
            "mean",
            "median",
            "variance",
            "std",
            "minimum",
            "maximum",
            "ci_lower",
            "ci_upper",
        ),
    ).rename(
        columns={
            "method_label": "Method",
            "metric_label": "Metric",
            "n": "N",
            "mean": "Mean",
            "median": "Median",
            "variance": "Variance",
            "std": "Std.",
            "minimum": "Min.",
            "maximum": "Max.",
            "ci_lower": "95% CI lower",
            "ci_upper": "95% CI upper",
        }
    )
    statistics_csv = output_dir / "statistics.csv"
    statistics_tex = output_dir / "statistics.tex"
    statistics_md = output_dir / "statistics.md"
    _write_csv(statistics, statistics_csv)
    statistics_central = statistics_display[
        ["Method", "Metric", "N", "Mean", "Median", "Variance", "Std."]
    ]
    statistics_interval = statistics_display[
        ["Method", "Metric", "Min.", "Max.", "95% CI lower", "95% CI upper"]
    ]
    statistics_tex.write_text(
        dataframe_to_latex(
            statistics_central,
            caption="Central tendency and dispersion by method and metric.",
            label="tab:descriptive-statistics-central",
            longtable=True,
        )
        + "\n"
        + dataframe_to_latex(
            statistics_interval,
            caption="Ranges and 95 percent mean confidence intervals.",
            label="tab:descriptive-statistics-intervals",
            longtable=True,
        ),
        encoding="utf-8",
    )
    statistics_md.write_text(
        dataframe_to_markdown(statistics_display), encoding="utf-8"
    )
    written.extend((statistics_csv, statistics_tex, statistics_md))

    significance_display = _select(
        significance,
        (
            "comparison",
            "metric_label",
            "pairs",
            "mean_difference_right_minus_left",
            "wilcoxon_statistic",
            "p_value",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
            "cliffs_delta_right_vs_left",
            "cliffs_interpretation",
            "cohens_d_paired",
            "cohens_interpretation",
        ),
    ).rename(
        columns={
            "comparison": "Comparison",
            "metric_label": "Metric",
            "pairs": "Pairs",
            "mean_difference_right_minus_left": "Mean diff.",
            "wilcoxon_statistic": "Wilcoxon W",
            "p_value": "p-value",
            "bootstrap_ci_lower": "Bootstrap lower",
            "bootstrap_ci_upper": "Bootstrap upper",
            "cliffs_delta_right_vs_left": "Cliff's delta",
            "cliffs_interpretation": "Cliff magnitude",
            "cohens_d_paired": "Cohen's d",
            "cohens_interpretation": "d magnitude",
        }
    )
    significance_csv = output_dir / "significance.csv"
    significance_tex = output_dir / "significance.tex"
    _write_csv(significance, significance_csv)
    significance_tests = significance_display[
        [
            "Comparison",
            "Metric",
            "Pairs",
            "Mean diff.",
            "Wilcoxon W",
            "p-value",
            "Bootstrap lower",
            "Bootstrap upper",
        ]
    ]
    significance_effects = significance_display[
        [
            "Comparison",
            "Metric",
            "Cliff's delta",
            "Cliff magnitude",
            "Cohen's d",
            "d magnitude",
        ]
    ]
    significance_tex.write_text(
        dataframe_to_latex(
            significance_tests,
            caption="Paired Wilcoxon tests and bootstrap confidence intervals.",
            label="tab:significance-tests",
        )
        + "\n"
        + dataframe_to_latex(
            significance_effects,
            caption="Cliff's delta and paired Cohen's d effect sizes.",
            label="tab:significance-effects",
        ),
        encoding="utf-8",
    )
    written.extend((significance_csv, significance_tex))

    dataset_display = _select(
        dataset_summary,
        (
            "dataset",
            "best_method_label",
            "accuracy",
            "tree_nodes",
            "mean_axp_length",
            "fit_time_seconds",
        ),
    ).rename(
        columns={
            "dataset": "Dataset",
            "best_method_label": "Best method",
            "accuracy": "Accuracy",
            "tree_nodes": "Nodes",
            "mean_axp_length": "AXp",
            "fit_time_seconds": "Runtime (s)",
        }
    )
    ranks_display = _select(
        method_ranks,
        ("method_label", "wins", "average_rank", "critical_difference"),
    ).rename(
        columns={
            "method_label": "Method",
            "wins": "Wins",
            "average_rank": "Average rank",
            "critical_difference": "Critical difference",
        }
    )
    dataset_csv = output_dir / "dataset_summary.csv"
    dataset_tex = output_dir / "dataset_summary.tex"
    _write_csv(dataset_summary, dataset_csv)
    dataset_tex.write_text(
        dataframe_to_latex(
            dataset_display,
            caption="Best method and associated metrics for every dataset.",
            label="tab:dataset-summary",
            longtable=True,
        )
        + "\n"
        + dataframe_to_latex(
            ranks_display,
            caption="Dataset wins and average accuracy ranks. The critical difference "
            "uses a two-sided Nemenyi comparison at alpha 0.05.",
            label="tab:dataset-ranks",
        ),
        encoding="utf-8",
    )
    written.extend((dataset_csv, dataset_tex))

    certification_display = certification.rename(
        columns={
            "audit_item": "Audit item",
            "count": "Count",
            "denominator": "Denominator",
            "percentage": "Percentage",
        }
    )
    certification_tex = output_dir / "certification_summary.tex"
    certification_tex.write_text(
        dataframe_to_latex(
            certification_display,
            caption="Certification and data-integrity audit.",
            label="tab:certification-summary",
        ),
        encoding="utf-8",
    )
    written.append(certification_tex)

    warnings_csv = output_dir / "warnings_summary.csv"
    _write_csv(warnings, warnings_csv)
    written.append(warnings_csv)

    search_display = search.rename(
        columns={
            "method_label": "Method",
            "rows": "Rows",
            "greedy_nodes": "Greedy nodes",
            "lookahead_nodes": "Lookahead nodes",
            "branch_and_bound_activations": "B&B active",
            "branch_and_bound_avoided": "B&B avoided",
            "cache_activations": "Cache active",
            "candidate_savings": "Candidate savings",
            "cache_hits": "Cache hits",
            "search_time_seconds": "Search time (s)",
        }
    ).drop(columns=["method"], errors="ignore")
    search_tex = output_dir / "search_summary.tex"
    search_selection = _select(
        search_display,
        (
            "Method",
            "Rows",
            "Greedy nodes",
            "Lookahead nodes",
            "B&B active",
            "B&B avoided",
        ),
    )
    search_cache = _select(
        search_display,
        (
            "Method",
            "Cache active",
            "Candidate savings",
            "Cache hits",
            "Search time (s)",
        ),
    )
    search_tex.write_text(
        dataframe_to_latex(
            search_selection,
            caption="Search-selection diagnostics.",
            label="tab:search-selection",
        )
        + "\n"
        + dataframe_to_latex(
            search_cache,
            caption="Cache activation, candidate savings, and search time.",
            label="tab:search-cache",
        ),
        encoding="utf-8",
    )
    written.append(search_tex)

    pruning_display = pruning.rename(
        columns={
            "method_label": "Method",
            "rows": "Rows",
            "validation_accuracy_before": "Accuracy before",
            "validation_accuracy_after": "Accuracy after",
            "validation_balanced_accuracy_before": "Balanced before",
            "validation_balanced_accuracy_after": "Balanced after",
            "validation_minority_recall_before": "Minority recall before",
            "validation_minority_recall_after": "Minority recall after",
            "nodes_before": "Nodes before",
            "nodes_after": "Nodes after",
            "tree_reduction_percentage": "Tree reduction (%)",
        }
    ).drop(columns=["method"], errors="ignore")
    pruning_tex = output_dir / "pruning_summary.tex"
    pruning_validation = _select(
        pruning_display,
        (
            "Method",
            "Accuracy before",
            "Accuracy after",
            "Balanced before",
            "Balanced after",
            "Minority recall before",
            "Minority recall after",
        ),
    )
    pruning_size = _select(
        pruning_display,
        ("Method", "Rows", "Nodes before", "Nodes after", "Tree reduction (%)"),
    )
    pruning_tex.write_text(
        dataframe_to_latex(
            pruning_validation,
            caption="Validation metrics before and after pruning.",
            label="tab:pruning-validation",
        )
        + "\n"
        + dataframe_to_latex(
            pruning_size,
            caption="Tree-size reduction during pruning.",
            label="tab:pruning-size",
        ),
        encoding="utf-8",
    )
    written.append(pruning_tex)

    axp_display = axp.rename(
        columns={
            "method_label": "Method",
            "rows": "Rows",
            "mean_axp": "Mean AXp",
            "median_axp": "Median AXp",
            "std_axp": "AXp std.",
            "minimum_axp": "Min. AXp",
            "maximum_axp": "Max. AXp",
            "family_usage": "Family usage",
        }
    ).drop(columns=["method"], errors="ignore")
    axp_tex = output_dir / "axp_summary.tex"
    axp_distribution = _select(
        axp_display,
        (
            "Method",
            "Rows",
            "Mean AXp",
            "Median AXp",
            "AXp std.",
            "Min. AXp",
            "Max. AXp",
        ),
    )
    axp_families = _select(axp_display, ("Method", "Family usage"))
    axp_tex.write_text(
        dataframe_to_latex(
            axp_distribution,
            caption="AXp distribution by method.",
            label="tab:axp-distribution",
        )
        + "\n"
        + dataframe_to_latex(
            axp_families,
            caption="Selected predicate-family usage.",
            label="tab:axp-families",
        ),
        encoding="utf-8",
    )
    written.append(axp_tex)

    return written
