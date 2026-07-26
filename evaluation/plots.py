"""Publication-quality Matplotlib figures with deterministic metadata."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Callable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smart-mdt-matplotlib")
)

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURE_NAMES
from .dataset_analysis import DatasetAnalysis
from .utils import method_label, ordered_methods


COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)

PDF_METADATA = {
    "Title": "Smart-MDT reproducible research evaluation",
    "Author": "Smart-MDT evaluation framework",
    "Creator": "Smart-MDT evaluation framework",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}

PNG_METADATA = {"Software": "Smart-MDT evaluation framework"}


def configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, name: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"
    fig.savefig(png, dpi=300, metadata=PNG_METADATA)
    fig.savefig(pdf, metadata=PDF_METADATA)
    plt.close(fig)
    return [png, pdf]


def _empty(title: str, message: str = "No data available") -> plt.Figure:
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
    return fig


def _method_data(
    results: pd.DataFrame,
    column: str,
    *,
    positive_only: bool = False,
) -> tuple[list[str], list[np.ndarray]]:
    methods = ordered_methods(results["method"].unique())
    labels: list[str] = []
    data: list[np.ndarray] = []
    for method in methods:
        values = results.loc[results["method"] == method, column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if positive_only:
            values = values[values > 0.0]
        if values.size:
            labels.append(method_label(method))
            data.append(values)
    return labels, data


def _boxplot(
    results: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    *,
    log_scale: bool = False,
) -> plt.Figure:
    labels, data = _method_data(results, column, positive_only=log_scale)
    if not data:
        return _empty(title)
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    boxes = axis.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, color in zip(boxes["boxes"], COLORS, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    if log_scale:
        axis.set_yscale("log")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    return fig


def _violin(results: pd.DataFrame) -> plt.Figure:
    labels, data = _method_data(results, "accuracy")
    if not data:
        return _empty("Accuracy distribution")
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    violins = axis.violinplot(
        data,
        positions=np.arange(1, len(data) + 1),
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )
    for body, color in zip(violins["bodies"], COLORS, strict=False):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.68)
    axis.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=28)
    axis.set_ylabel("Accuracy")
    axis.set_title("Accuracy violin plot")
    fig.tight_layout()
    return fig


def _mean_tradeoff(
    results: pd.DataFrame,
    x_column: str,
    x_label: str,
    title: str,
    *,
    log_x: bool = False,
) -> plt.Figure:
    means = (
        results.groupby("method", sort=True)[["accuracy", x_column]]
        .mean()
        .reset_index()
    )
    if means.empty:
        return _empty(title)
    methods = ordered_methods(means["method"])
    means = means.set_index("method").reindex(methods).reset_index()
    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    for index, row in means.iterrows():
        axis.scatter(
            row[x_column],
            row["accuracy"],
            s=65,
            color=COLORS[index % len(COLORS)],
            edgecolor="black",
            linewidth=0.5,
            label=method_label(str(row["method"])),
            zorder=3,
        )
    if log_x:
        axis.set_xscale("log")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Mean accuracy")
    axis.set_title(title)
    axis.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


def _pareto(results: pd.DataFrame) -> plt.Figure:
    means = (
        results.groupby("method", sort=True)[["accuracy", "tree_nodes"]]
        .mean()
        .reset_index()
        .sort_values(["tree_nodes", "accuracy"], ascending=[True, False])
    )
    if means.empty:
        return _empty("Accuracy–complexity Pareto frontier")
    best_accuracy = -np.inf
    frontier_rows: list[pd.Series] = []
    for _, row in means.iterrows():
        if row["accuracy"] > best_accuracy:
            frontier_rows.append(row)
            best_accuracy = float(row["accuracy"])
    frontier = pd.DataFrame(frontier_rows)

    fig, axis = plt.subplots(figsize=(7.2, 5.0))
    method_colors = {
        method: COLORS[index % len(COLORS)]
        for index, method in enumerate(ordered_methods(means["method"]))
    }
    for _, row in means.iterrows():
        method = str(row["method"])
        axis.scatter(
            row["tree_nodes"],
            row["accuracy"],
            s=70,
            color=method_colors[method],
            edgecolor="black",
            linewidth=0.5,
        )
        axis.annotate(
            method_label(method),
            (row["tree_nodes"], row["accuracy"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.plot(
        frontier["tree_nodes"],
        frontier["accuracy"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Pareto frontier",
    )
    axis.set_xlabel("Mean tree nodes (lower is better)")
    axis.set_ylabel("Mean accuracy (higher is better)")
    axis.set_title("Accuracy–complexity Pareto frontier")
    axis.legend()
    fig.tight_layout()
    return fig


def _dataset_scatter(
    analysis: DatasetAnalysis,
    x_column: str,
    x_label: str,
    title: str,
    *,
    log_x: bool = False,
) -> plt.Figure:
    frame = analysis.dataset_metric_means
    if frame.empty:
        return _empty(title)
    methods = ordered_methods(frame["method"].unique())
    fig, axis = plt.subplots(figsize=(8.0, 5.2))
    for index, method in enumerate(methods):
        rows = frame.loc[frame["method"] == method]
        x = rows[x_column].to_numpy(dtype=float)
        y = rows["accuracy"].to_numpy(dtype=float)
        if log_x:
            keep = x > 0.0
            x, y = x[keep], y[keep]
        axis.scatter(
            x,
            y,
            s=22,
            alpha=0.55,
            color=COLORS[index % len(COLORS)],
            label=method_label(method),
            edgecolors="none",
        )
    if log_x:
        axis.set_xscale("log")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Dataset mean accuracy")
    axis.set_title(title)
    axis.legend(ncols=2)
    fig.tight_layout()
    return fig


def _wins(analysis: DatasetAnalysis) -> plt.Figure:
    ranks = analysis.method_ranks
    if ranks.empty:
        return _empty("Dataset wins")
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    labels = ranks["method_label"].tolist()
    positions = np.arange(len(ranks))
    axis.bar(
        positions,
        ranks["wins"],
        color=[COLORS[index % len(COLORS)] for index in positions],
        edgecolor="black",
        linewidth=0.5,
    )
    axis.set_xticks(positions, labels, rotation=28)
    axis.set_ylabel("Dataset wins")
    axis.set_title("Deterministic dataset winner count")
    fig.tight_layout()
    return fig


def _average_rank(analysis: DatasetAnalysis) -> plt.Figure:
    ranks = analysis.method_ranks.sort_values("average_rank", ascending=False)
    if ranks.empty:
        return _empty("Average accuracy ranks")
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    positions = np.arange(len(ranks))
    axis.barh(
        positions,
        ranks["average_rank"],
        color=[COLORS[index % len(COLORS)] for index in positions],
        edgecolor="black",
        linewidth=0.5,
    )
    axis.set_yticks(positions, ranks["method_label"])
    axis.set_xlabel("Average rank (lower is better)")
    axis.set_title(
        f"Average accuracy rank (Nemenyi CD = {analysis.critical_difference:.3f})"
    )
    fig.tight_layout()
    return fig


def _certification(frame: pd.DataFrame) -> plt.Figure:
    if frame.empty:
        return _empty("Certification overview")
    selected_names = [
        "theorem_certified_rows",
        "empirical_rows",
        "forbidden_predicates",
        "feature_label_leakage",
        "path_violations",
        "cached_subtree_violations",
        "empirical_fallbacks",
    ]
    selected = frame.set_index("audit_item").reindex(selected_names).dropna()
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    positions = np.arange(len(selected))
    axis.barh(
        positions,
        selected["percentage"],
        color=[
            "#2ca02c" if item == "theorem_certified_rows" else "#d62728"
            for item in selected.index
        ],
    )
    axis.set_yticks(
        positions, [item.replace("_", " ").title() for item in selected.index]
    )
    axis.set_xlabel("Percentage of applicable rows or datasets")
    axis.set_xlim(0, 105)
    axis.set_title("Certification and integrity overview")
    fig.tight_layout()
    return fig


def _warnings(frame: pd.DataFrame) -> plt.Figure:
    if frame.empty:
        return _empty("Benchmark warning distribution")
    grouped = (
        frame.groupby("warning_type", sort=True)["warning_records"].sum().sort_values()
    )
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    positions = np.arange(len(grouped))
    axis.barh(positions, grouped.values, color="#ff7f0e", edgecolor="black")
    axis.set_yticks(
        positions, [str(value).replace("_", " ") for value in grouped.index]
    )
    axis.set_xlabel("Warning records")
    axis.set_title("Benchmark warning distribution")
    fig.tight_layout()
    return fig


def _search(frame: pd.DataFrame) -> plt.Figure:
    if frame.empty:
        return _empty("Search diagnostics")
    positions = np.arange(len(frame))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.bar(
        positions - width / 2,
        frame["greedy_nodes"],
        width,
        label="Greedy nodes",
        color="#1f77b4",
    )
    axis.bar(
        positions + width / 2,
        frame["lookahead_nodes"],
        width,
        label="Selective-lookahead nodes",
        color="#ff7f0e",
    )
    axis.set_xticks(positions, frame["method_label"], rotation=28)
    axis.set_ylabel("Node selections")
    axis.set_title("Search strategy diagnostics")
    axis.legend()
    fig.tight_layout()
    return fig


def _pruning(frame: pd.DataFrame) -> plt.Figure:
    if frame.empty:
        return _empty("Pruning diagnostics")
    positions = np.arange(len(frame))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.bar(
        positions - width / 2,
        frame["validation_balanced_accuracy_before"],
        width,
        label="Balanced accuracy before",
        color="#1f77b4",
    )
    axis.bar(
        positions + width / 2,
        frame["validation_balanced_accuracy_after"],
        width,
        label="Balanced accuracy after",
        color="#2ca02c",
    )
    axis.set_xticks(positions, frame["method_label"], rotation=28)
    axis.set_ylabel("Validation balanced accuracy")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Class-aware pruning diagnostics")
    axis.legend()
    fig.tight_layout()
    return fig


def generate_all_figures(
    output_dir: Path,
    *,
    results: pd.DataFrame,
    dataset_analysis: DatasetAnalysis,
    certification: pd.DataFrame,
    warnings: pd.DataFrame,
    search: pd.DataFrame,
    pruning: pd.DataFrame,
) -> list[Path]:
    """Generate every required figure in both PNG and vector PDF formats."""

    configure_matplotlib()
    builders: dict[str, Callable[[], plt.Figure]] = {
        "accuracy_boxplot": lambda: _boxplot(
            results, "accuracy", "Accuracy boxplot", "Accuracy"
        ),
        "accuracy_violin": lambda: _violin(results),
        "runtime_boxplot": lambda: _boxplot(
            results, "fit_time_seconds", "Runtime boxplot", "Fit time (seconds)"
        ),
        "runtime_log_boxplot": lambda: _boxplot(
            results,
            "fit_time_seconds",
            "Runtime boxplot (log scale)",
            "Fit time (seconds, log scale)",
            log_scale=True,
        ),
        "nodes_boxplot": lambda: _boxplot(
            results, "tree_nodes", "Tree-node boxplot", "Tree nodes"
        ),
        "axp_boxplot": lambda: _boxplot(
            results, "mean_axp_length", "AXp-length boxplot", "Mean AXp length"
        ),
        "predicate_literals_boxplot": lambda: _boxplot(
            results,
            "predicate_literals",
            "Predicate-literal boxplot",
            "Predicate literals",
        ),
        "pareto": lambda: _pareto(results),
        "accuracy_vs_nodes": lambda: _dataset_scatter(
            dataset_analysis,
            "tree_nodes",
            "Dataset mean tree nodes",
            "Accuracy versus tree nodes",
        ),
        "accuracy_vs_runtime": lambda: _dataset_scatter(
            dataset_analysis,
            "fit_time_seconds",
            "Dataset mean fit time (seconds, log scale)",
            "Accuracy versus runtime",
            log_x=True,
        ),
        "accuracy_vs_axp": lambda: _dataset_scatter(
            dataset_analysis,
            "mean_axp_length",
            "Dataset mean AXp length",
            "Accuracy versus AXp length",
        ),
        "dataset_win_histogram": lambda: _wins(dataset_analysis),
        "average_rank": lambda: _average_rank(dataset_analysis),
        "certification_overview": lambda: _certification(certification),
        "warning_distribution": lambda: _warnings(warnings),
        "search_diagnostics": lambda: _search(search),
        "pruning_diagnostics": lambda: _pruning(pruning),
    }
    written: list[Path] = []
    for name in FIGURE_NAMES:
        written.extend(_save(builders[name](), output_dir, name))
    return written
