"""Configuration and schema constants for the evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetricSpec:
    """Canonical metric metadata."""

    column: str
    label: str
    higher_is_better: bool


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("accuracy", "Accuracy", True),
    MetricSpec("tree_nodes", "Tree nodes", False),
    MetricSpec("predicate_literals", "Predicate literals", False),
    MetricSpec("mean_axp_length", "Mean AXp length", False),
    MetricSpec("fit_time_seconds", "Fit time (seconds)", False),
)

METRIC_BY_COLUMN = {metric.column: metric for metric in METRICS}

METHOD_LABELS: dict[str, str] = {
    "unary": "Unary",
    "horn": "Horn",
    "antihorn": "AntiHorn",
    "square2cnf": "Square2CNF",
    "affine": "Boolean Affine/GF(2)",
    "smart_certified": "SmartCertified",
    "cals": "CALS-MDT",
    "cals_compact_explain": "CompactExplain",
}

METHOD_ORDER: tuple[str, ...] = tuple(METHOD_LABELS)

PAIRWISE_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("smart_certified", "cals"),
    ("smart_certified", "cals_compact_explain"),
    ("cals", "cals_compact_explain"),
)

REQUIRED_FILES: tuple[str, ...] = ("full_results.csv",)

OPTIONAL_FILES: tuple[str, ...] = (
    "summary_by_method.csv",
    "theorem_certified_results.csv",
    "dataset_metadata.csv",
    "benchmark_warnings.csv",
    "search_diagnostics.csv",
    "cache_diagnostics.csv",
    "family_budget_diagnostics.csv",
    "pruning_diagnostics.csv",
    "beam_diagnostics.csv",
    "axp_metadata.csv",
    "empirical_results.csv",
    "tuning_diagnostics.csv",
)

OPTIONAL_TEXT_FILES: tuple[str, ...] = ("README_RESULTS.md",)

FIGURE_NAMES: tuple[str, ...] = (
    "accuracy_boxplot",
    "accuracy_violin",
    "runtime_boxplot",
    "runtime_log_boxplot",
    "nodes_boxplot",
    "axp_boxplot",
    "predicate_literals_boxplot",
    "pareto",
    "accuracy_vs_nodes",
    "accuracy_vs_runtime",
    "accuracy_vs_axp",
    "dataset_win_histogram",
    "average_rank",
    "certification_overview",
    "warning_distribution",
    "search_diagnostics",
    "pruning_diagnostics",
)

DEFAULT_SEED = 20260718
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class EvaluationConfig:
    """Runtime settings for a reproducible report."""

    input_dir: Path
    output_dir: Path
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    seed: int = DEFAULT_SEED
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL

    def validate(self) -> None:
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be at least 1")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1")
