"""Command-line orchestration for the complete reproducible evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smart-mdt-matplotlib")
)

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import matplotlib
import numpy as np
import pandas as pd
import scipy
import jinja2

from .certification import (
    axp_summary,
    certification_summary,
    pruning_summary,
    search_summary,
    warning_summary,
)
from .config import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_SEED,
    EvaluationConfig,
    OPTIONAL_FILES,
    OPTIONAL_TEXT_FILES,
    REQUIRED_FILES,
)
from .dataset_analysis import analyse_datasets
from .io import EvaluationDataError, load_benchmark_folder
from .plots import generate_all_figures
from .significance import pairwise_significance
from .statistics import descriptive_statistics
from .tables import (
    dataframe_to_markdown,
    latex_escape,
    write_evaluation_tables,
)
from .utils import method_label, ordered_methods, sha256_file, write_json


@dataclass(frozen=True)
class EvaluationArtifacts:
    output_dir: Path
    figures: tuple[Path, ...]
    tables: tuple[Path, ...]
    reports: tuple[Path, ...]


def _best_method(
    means: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    order = {method: index for index, method in enumerate(ordered_methods(means.index))}
    rows = [
        (str(method), float(value))
        for method, value in means[metric].items()
    ]
    rows.sort(
        key=lambda item: (
            -item[1] if higher_is_better else item[1],
            order[item[0]],
        )
    )
    method, value = rows[0]
    label = method_label(method)
    return {
        "method": method,
        "method_label": label,
        "method_label_tex": latex_escape(label),
        "value": value,
    }


def _significance_display(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "comparison",
        "metric_label",
        "pairs",
        "mean_difference_right_minus_left",
        "p_value",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "cliffs_delta_right_vs_left",
        "cliffs_interpretation",
        "cohens_d_paired",
        "cohens_interpretation",
    ]
    selected = [column for column in columns if column in frame]
    return frame.loc[:, selected].rename(
        columns={
            "comparison": "Comparison",
            "metric_label": "Metric",
            "pairs": "Pairs",
            "mean_difference_right_minus_left": "Mean diff.",
            "p_value": "p-value",
            "bootstrap_ci_lower": "CI lower",
            "bootstrap_ci_upper": "CI upper",
            "cliffs_delta_right_vs_left": "Cliff's delta",
            "cliffs_interpretation": "Cliff magnitude",
            "cohens_d_paired": "Cohen's d",
            "cohens_interpretation": "d magnitude",
        }
    )


def _significant_conclusions(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No configured paired comparison reached p < 0.05."]
    significant = frame.loc[frame["p_value"] < 0.05].copy()
    if significant.empty:
        return ["No configured paired comparison reached p < 0.05."]
    significant["_accuracy_first"] = (significant["metric"] != "accuracy").astype(int)
    significant = significant.sort_values(
        ["_accuracy_first", "p_value", "comparison", "metric"], kind="mergesort"
    )
    conclusions: list[str] = []
    for _, row in significant.head(6).iterrows():
        difference = float(row["mean_difference_right_minus_left"])
        direction = "higher" if difference > 0 else "lower"
        conclusions.append(
            f"For {row['comparison']} on {row['metric_label']}, the right-hand "
            f"method was {direction} by {abs(difference):.4f} "
            f"(95% bootstrap CI [{row['bootstrap_ci_lower']:.4f}, "
            f"{row['bootstrap_ci_upper']:.4f}], p={row['p_value']:.4g}, "
            f"Cliff magnitude {row['cliffs_interpretation']})."
        )
    return conclusions


def _certification_sentence(certification: pd.DataFrame) -> str:
    counts = certification.set_index("audit_item")["count"].to_dict()
    return (
        f"{int(counts.get('theorem_certified_rows', 0))} rows are theorem-certified; "
        f"the audit found {int(counts.get('empirical_rows', 0))} empirical rows, "
        f"{int(counts.get('forbidden_predicates', 0))} forbidden predicates, "
        f"{int(counts.get('feature_label_leakage', 0))} feature-label leakage "
        f"findings, {int(counts.get('path_violations', 0))} path violations, and "
        f"{int(counts.get('empirical_fallbacks', 0))} empirical fallbacks."
    )


def _render_reports(
    report_dir: Path,
    *,
    context: dict[str, Any],
) -> list[Path]:
    template_dir = Path(__file__).resolve().parent / "templates"
    environment = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("evaluation_report.md.j2", "evaluation_report.md"),
        ("evaluation_report.tex.j2", "evaluation_report.tex"),
        ("executive_summary.md.j2", "executive_summary.md"),
        ("executive_summary.tex.j2", "executive_summary.tex"),
    )
    written: list[Path] = []
    for template_name, output_name in specifications:
        path = report_dir / output_name
        rendered = environment.get_template(template_name).render(**context)
        path.write_text(rendered.rstrip() + "\n", encoding="utf-8")
        written.append(path)
    return written


def _manifest(
    config: EvaluationConfig,
    *,
    data_files: list[Path],
    generated_files: list[Path],
) -> dict[str, Any]:
    return {
        "framework": "Smart-MDT research evaluation framework",
        "framework_version": "1.0.0",
        "configuration": {
            "bootstrap_resamples": config.bootstrap_resamples,
            "confidence_level": config.confidence_level,
            "seed": config.seed,
        },
        "dependencies": {
            "jinja2": jinja2.__version__,
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
            "scipy": scipy.__version__,
        },
        "inputs": {
            path.name: sha256_file(path)
            for path in sorted(data_files, key=lambda item: item.name)
        },
        "outputs": {
            str(path.relative_to(config.output_dir)): sha256_file(path)
            for path in sorted(generated_files, key=lambda item: str(item))
        },
    }


def build_report(config: EvaluationConfig) -> EvaluationArtifacts:
    """Run the full evaluation and return every generated artifact."""

    config.validate()
    data = load_benchmark_folder(config.input_dir)
    figures_dir = config.output_dir / "figures"
    tables_dir = config.output_dir / "tables"
    report_dir = config.output_dir / "report"

    stats_frame = descriptive_statistics(data.results, config.confidence_level)
    significance_frame = pairwise_significance(
        data.results,
        resamples=config.bootstrap_resamples,
        confidence_level=config.confidence_level,
        seed=config.seed,
    )
    dataset = analyse_datasets(data.results)
    certification_frame = certification_summary(data)
    warnings_frame = warning_summary(data)
    search_frame = search_summary(data)
    pruning_frame = pruning_summary(data)
    axp_frame = axp_summary(data)

    table_paths = write_evaluation_tables(
        tables_dir,
        statistics=stats_frame,
        significance=significance_frame,
        dataset_summary=dataset.dataset_summary,
        method_ranks=dataset.method_ranks,
        certification=certification_frame,
        warnings=warnings_frame,
        search=search_frame,
        pruning=pruning_frame,
        axp=axp_frame,
    )
    figure_paths = generate_all_figures(
        figures_dir,
        results=data.results,
        dataset_analysis=dataset,
        certification=certification_frame,
        warnings=warnings_frame,
        search=search_frame,
        pruning=pruning_frame,
    )

    means = data.results.groupby("method", sort=True)[
        [
            "accuracy",
            "tree_nodes",
            "predicate_literals",
            "mean_axp_length",
            "fit_time_seconds",
        ]
    ].mean()
    best_accuracy = _best_method(means, "accuracy", higher_is_better=True)
    smallest_tree = _best_method(means, "tree_nodes", higher_is_better=False)
    fastest_runtime = _best_method(
        means, "fit_time_seconds", higher_is_better=False
    )
    best_axp = _best_method(means, "mean_axp_length", higher_is_better=False)
    conclusions = _significant_conclusions(significance_frame)
    certification_sentence = _certification_sentence(certification_frame)

    main_figures = [
        {"name": "accuracy_boxplot", "caption_tex": "Accuracy boxplot."},
        {"name": "accuracy_violin", "caption_tex": "Accuracy violin plot."},
        {"name": "pareto", "caption_tex": "Accuracy--complexity Pareto frontier."},
        {"name": "accuracy_vs_nodes", "caption_tex": "Accuracy versus tree nodes."},
        {"name": "accuracy_vs_runtime", "caption_tex": "Accuracy versus runtime."},
        {"name": "accuracy_vs_axp", "caption_tex": "Accuracy versus AXp length."},
        {"name": "runtime_boxplot", "caption_tex": "Runtime boxplot."},
        {
            "name": "runtime_log_boxplot",
            "caption_tex": "Runtime boxplot on a logarithmic scale.",
        },
        {"name": "nodes_boxplot", "caption_tex": "Tree-node boxplot."},
        {"name": "axp_boxplot", "caption_tex": "AXp-length boxplot."},
        {
            "name": "predicate_literals_boxplot",
            "caption_tex": "Predicate-literal boxplot.",
        },
    ]
    context: dict[str, Any] = {
        "input_name": data.input_dir.name,
        "input_name_tex": latex_escape(data.input_dir.name),
        "row_count": len(data.results),
        "dataset_count": int(data.results["dataset"].nunique()),
        "method_count": int(data.results["method"].nunique()),
        "bootstrap_resamples": config.bootstrap_resamples,
        "seed": config.seed,
        "confidence_percent": 100.0 * config.confidence_level,
        "missing_optional": (
            ", ".join(data.missing_optional_files)
            if data.missing_optional_files
            else "none"
        ),
        "best_accuracy": best_accuracy,
        "smallest_tree": smallest_tree,
        "fastest_runtime": fastest_runtime,
        "best_axp": best_axp,
        "key_conclusions": conclusions,
        "key_conclusions_tex": [latex_escape(value) for value in conclusions],
        "statistics_markdown": dataframe_to_markdown(
            stats_frame[
                [
                    "method_label",
                    "metric_label",
                    "n",
                    "mean",
                    "median",
                    "std",
                    "ci_lower",
                    "ci_upper",
                ]
            ].rename(
                columns={
                    "method_label": "Method",
                    "metric_label": "Metric",
                    "n": "N",
                    "mean": "Mean",
                    "median": "Median",
                    "std": "Std.",
                    "ci_lower": "95% CI lower",
                    "ci_upper": "95% CI upper",
                }
            )
        ),
        "significance_markdown": dataframe_to_markdown(
            _significance_display(significance_frame)
        ),
        "ranks_markdown": dataframe_to_markdown(
            dataset.method_ranks[
                ["method_label", "wins", "average_rank"]
            ].rename(
                columns={
                    "method_label": "Method",
                    "wins": "Wins",
                    "average_rank": "Average rank",
                }
            )
        ),
        "critical_difference": dataset.critical_difference,
        "certification_markdown": dataframe_to_markdown(
            certification_frame.rename(
                columns={
                    "audit_item": "Audit item",
                    "count": "Count",
                    "denominator": "Denominator",
                    "percentage": "Percentage",
                }
            )
        ),
        "warning_group_count": len(warnings_frame),
        "warnings_markdown": dataframe_to_markdown(warnings_frame.head(30)),
        "search_markdown": dataframe_to_markdown(
            search_frame.drop(columns=["method"], errors="ignore")
        ),
        "pruning_markdown": dataframe_to_markdown(
            pruning_frame.drop(columns=["method"], errors="ignore")
        ),
        "axp_markdown": dataframe_to_markdown(
            axp_frame.drop(columns=["method"], errors="ignore")
        ),
        "main_figures": main_figures,
        "certification_sentence": certification_sentence,
        "certification_sentence_tex": latex_escape(certification_sentence),
    }
    report_paths = _render_reports(report_dir, context=context)

    expected_inputs = [
        data.input_dir / filename
        for filename in (*REQUIRED_FILES, *OPTIONAL_FILES, *OPTIONAL_TEXT_FILES)
        if (data.input_dir / filename).is_file()
    ]
    generated = [*table_paths, *figure_paths, *report_paths]
    manifest_path = report_dir / "reproducibility_manifest.json"
    write_json(
        manifest_path,
        _manifest(config, data_files=expected_inputs, generated_files=generated),
    )
    report_paths.append(manifest_path)
    return EvaluationArtifacts(
        output_dir=config.output_dir,
        figures=tuple(figure_paths),
        tables=tuple(table_paths),
        reports=tuple(report_paths),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Smart-MDT research evaluation report."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Benchmark output directory containing full_results.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output root (default: the evaluation package directory).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help=f"Paired bootstrap resamples (default: {DEFAULT_BOOTSTRAP_RESAMPLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base seed for deterministic bootstrap streams (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help=f"Confidence level (default: {DEFAULT_CONFIDENCE_LEVEL}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = EvaluationConfig(
        input_dir=args.input.expanduser().resolve(),
        output_dir=args.output.expanduser().resolve(),
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        confidence_level=args.confidence_level,
    )
    try:
        artifacts = build_report(config)
    except (EvaluationDataError, ValueError) as error:
        parser.error(str(error))
    print(
        f"generated {len(artifacts.figures)} figure files, "
        f"{len(artifacts.tables)} table files, and "
        f"{len(artifacts.reports)} report files in {artifacts.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
