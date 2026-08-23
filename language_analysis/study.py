#!/usr/bin/env python3
"""Reproducible exact-language characterization for the frozen benchmark."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smart-mdt-language-mpl"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


FIXED = ["unary", "horn", "antihorn", "square2cnf", "affine"]
ADAPTIVE = ["smart_certified", "cals", "cals_compact_explain"]
DISPLAY = {
    "unary": "Unary",
    "horn": "Star-nested Horn",
    "antihorn": "Star-nested Anti-Horn",
    "square2cnf": "Square2CNF",
    "affine": "Boolean Affine",
    "smart_certified": "SmartCertified",
    "cals": "CALS",
    "cals_compact_explain": "CALS CompactExplain",
}
METRICS = {
    "accuracy": True,
    "tree_nodes": False,
    "literals_after_prune": False,
    "mean_axp_length": False,
    "total_fit_time": False,
}
CONTROLLED_FAMILY = {
    "unary": "unary",
    "horn_simple": "horn",
    "horn_chain": "horn",
    "antihorn_chain": "antihorn",
    "square_form_i": "square2cnf",
    "square_form_ii": "square2cnf",
    "square_form_iii": "square2cnf",
    "affine_xor3": "affine",
}
CONTROLLED_TARGET_ORDER = [
    "unary",
    "horn_simple",
    "horn_chain",
    "antihorn_chain",
    "square_form_i",
    "square_form_ii",
    "square_form_iii",
    "affine_xor3",
]
CONTROLLED_TARGET_DISPLAY = {
    "unary": "Unary target",
    "horn_simple": "Simple star-nested Horn",
    "horn_chain": "Chained star-nested Horn",
    "antihorn_chain": "Chained star-nested Anti-Horn",
    "square_form_i": "Square2CNF Form I",
    "square_form_ii": "Square2CNF Form II",
    "square_form_iii": "Square2CNF Form III",
    "affine_xor3": "XOR3",
}


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.grid": False,
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    metadata = None
    if path.suffix.lower() == ".pdf":
        metadata = {
            "Creator": "Smart-MDT exact-language study",
            "CreationDate": None,
            "ModDate": None,
        }
    fig.savefig(path, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def exact_bool(series: pd.Series, field: str) -> pd.Series:
    """Parse CSV booleans without treating a non-empty 'false' string as true."""
    normalized = series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower()
    )
    invalid = ~normalized.isin([True, False, "true", "false"])
    if invalid.any():
        values = sorted(series[invalid].astype(str).unique())
        raise SystemExit(f"refusing invalid {field} values: {values}")
    return normalized.map({True: True, False: False, "true": True, "false": False})


def heatmap(frame: pd.DataFrame, path: Path, title: str, fmt: str, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, max(4.0, 0.22 * len(frame))))
    image = ax.imshow(frame.to_numpy(float), aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(frame.columns)), [DISPLAY.get(c, c) for c in frame.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(frame.index)), frame.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    if len(frame) <= 12:
        for i in range(len(frame)):
            for j in range(len(frame.columns)):
                ax.text(j, i, format(frame.iloc[i, j], fmt), ha="center", va="center", fontsize=7)
    save(fig, path)


def controlled_heatmap(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    fmt: str,
    colorbar_label: str,
) -> None:
    ordered = frame.reindex(CONTROLLED_TARGET_ORDER)[FIXED]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    image = ax.imshow(ordered.to_numpy(float), aspect="auto", cmap="YlGn_r")
    ax.set_xticks(
        range(len(ordered.columns)),
        [DISPLAY[method] for method in ordered.columns],
        rotation=28,
        ha="right",
    )
    ax.set_yticks(
        range(len(ordered.index)),
        [CONTROLLED_TARGET_DISPLAY[target] for target in ordered.index],
    )
    ax.set_title(f"{title}\n↓ Smaller is better")
    ax.set_xlabel(
        "Accuracy alone can hide specialization; compactness shows when a "
        "language represents native structure directly."
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(colorbar_label)
    for i in range(len(ordered)):
        for j in range(len(ordered.columns)):
            ax.text(
                j,
                i,
                format(ordered.iloc[i, j], fmt),
                ha="center",
                va="center",
                fontsize=7,
            )
    save(fig, path)


def parse_controlled(raw_path: Path, output: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path)
    split = raw["dataset"].str.extract(r"^(.*)__noise_(\d+)$")
    raw["target"] = split[0]
    raw["noise_percent"] = split[1].astype(int)
    raw["target_family"] = raw["target"].map(CONTROLLED_FAMILY)
    grouped = (
        raw.groupby(["target", "target_family", "noise_percent", "method"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            nodes=("tree_nodes", "mean"),
            leaves=("leaves", "mean"),
            depth=("max_depth_reached", "mean"),
            predicate_literals=("literals_after_prune", "mean"),
            mean_axp_length=("mean_axp_length", "mean"),
            training_time=("total_fit_time", "mean"),
            repetitions=("run", "nunique"),
        )
        .sort_values(["target", "noise_percent", "method"])
    )
    grouped.to_csv(output / "controlled_expressive_power.csv", index=False)
    clean = grouped[grouped["noise_percent"] == 0].pivot(index="target", columns="method", values="accuracy")[FIXED]
    heatmap(
        clean,
        output / "controlled_expressive_power_heatmap.pdf",
        "Controlled exact-language accuracy (0% label noise)",
        ".3f",
        "viridis",
    )
    return grouped


def controlled_compactness(controlled: pd.DataFrame, output: Path) -> pd.DataFrame:
    """Export clean-target compactness evidence without rerunning experiments."""
    clean = controlled[controlled["noise_percent"] == 0].copy()
    expected_keys = {
        (target, method)
        for target in CONTROLLED_TARGET_ORDER
        for method in FIXED
    }
    actual_keys = set(zip(clean["target"], clean["method"]))
    if len(clean) != len(expected_keys) or actual_keys != expected_keys:
        raise SystemExit("refusing incomplete controlled 0%-noise family grid")

    clean["native_family"] = clean["target"].map(CONTROLLED_FAMILY)
    clean["accuracy_gap_from_best"] = clean.groupby("target")["accuracy"].transform("max") - clean["accuracy"]
    native = clean[clean["method"] == clean["native_family"]].set_index("target")
    clean["node_ratio_vs_native"] = clean.apply(
        lambda row: row["nodes"] / native.loc[row["target"], "nodes"], axis=1
    )
    clean["literal_ratio_vs_native"] = clean.apply(
        lambda row: row["predicate_literals"]
        / native.loc[row["target"], "predicate_literals"],
        axis=1,
    )
    summary = clean.rename(
        columns={
            "method": "family",
            "predicate_literals": "literals",
            "mean_axp_length": "mean_axp",
            "training_time": "fit_time",
        }
    )[
        [
            "target",
            "native_family",
            "family",
            "accuracy",
            "nodes",
            "literals",
            "mean_axp",
            "fit_time",
            "accuracy_gap_from_best",
            "node_ratio_vs_native",
            "literal_ratio_vs_native",
        ]
    ]
    target_rank = {target: index for index, target in enumerate(CONTROLLED_TARGET_ORDER)}
    family_rank = {family: index for index, family in enumerate(FIXED)}
    summary = summary.sort_values(
        ["target", "family"],
        key=lambda series: series.map(
            target_rank if series.name == "target" else family_rank
        ),
    ).reset_index(drop=True)

    xor_nodes = summary[summary["target"] == "affine_xor3"].set_index("family")["nodes"]
    expected_xor_nodes = {"affine": 3.0, "square2cnf": 15.0, "unary": 23.0}
    for family, expected in expected_xor_nodes.items():
        if not np.isclose(xor_nodes[family], expected, atol=1e-12):
            raise SystemExit(
                f"controlled source mismatch: XOR3 {family} nodes="
                f"{xor_nodes[family]} expected {expected}"
            )

    summary.to_csv(output / "controlled_compactness_summary.csv", index=False)
    write_controlled_compactness_latex(
        summary, output / "controlled_compactness_summary.tex"
    )

    clean_pivots = {
        "nodes": clean.pivot(index="target", columns="method", values="nodes"),
        "literals": clean.pivot(
            index="target", columns="method", values="predicate_literals"
        ),
        "mean_axp": clean.pivot(
            index="target", columns="method", values="mean_axp_length"
        ),
    }
    controlled_heatmap(
        clean_pivots["nodes"],
        output / "controlled_nodes_heatmap.pdf",
        "Mean final tree nodes at 0% label noise",
        ".1f",
        "Mean final tree nodes",
    )
    controlled_heatmap(
        clean_pivots["literals"],
        output / "controlled_literals_heatmap.pdf",
        "Mean predicate-literal count at 0% label noise",
        ".1f",
        "Mean predicate literals",
    )
    controlled_heatmap(
        clean_pivots["mean_axp"],
        output / "controlled_axp_heatmap.pdf",
        "Mean AXp length at 0% label noise",
        ".2f",
        "Mean AXp length",
    )
    return summary


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_controlled_compactness_latex(frame: pd.DataFrame, path: Path) -> None:
    columns = [
        "Target",
        "Native",
        "Family",
        "Acc.",
        "Nodes",
        "Literals",
        "AXp",
        "Fit time",
        "Acc. gap",
        "Node ratio",
        "Literal ratio",
    ]
    lines = [
        r"\begin{tabular}{lllrrrrrrrr}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        values = [
            CONTROLLED_TARGET_DISPLAY[row.target],
            DISPLAY[row.native_family],
            DISPLAY[row.family],
            f"{row.accuracy:.3f}",
            f"{row.nodes:.1f}",
            f"{row.literals:.1f}",
            f"{row.mean_axp:.2f}",
            f"{row.fit_time:.6f}",
            f"{row.accuracy_gap_from_best:.3f}",
            f"{row.node_ratio_vs_native:.2f}",
            f"{row.literal_ratio_vs_native:.2f}",
        ]
        important = row.family == row.native_family or (
            row.target == "affine_xor3"
            and row.family in {"unary", "square2cnf", "affine"}
        )
        escaped = [latex_escape(value) for value in values]
        if important:
            escaped = [rf"\textbf{{{value}}}" for value in escaped]
        lines.append(" & ".join(escaped) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% Native-family rows and the XOR3 Unary/Square2CNF/Affine comparison are bold.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixed_results(full: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed = full[full["method"].isin(FIXED)].copy()
    by_depth = (
        fixed.groupby(["dataset", "method", "depth"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            nodes=("tree_nodes", "mean"),
            predicate_literals=("literals_after_prune", "mean"),
            mean_axp_length=("mean_axp_length", "mean"),
            runtime=("total_fit_time", "mean"),
            runs=("run", "nunique"),
        )
    )
    overall = (
        fixed.groupby(["dataset", "method"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            nodes=("tree_nodes", "mean"),
            predicate_literals=("literals_after_prune", "mean"),
            mean_axp_length=("mean_axp_length", "mean"),
            runtime=("total_fit_time", "mean"),
            runs=("run", "nunique"),
            depths=("depth", "nunique"),
        )
    )
    by_depth.insert(2, "aggregation", "depth_mean_across_runs")
    overall_export = overall.copy()
    overall_export.insert(2, "aggregation", "overall_mean_across_runs_and_depths")
    overall_export["depth"] = "all"
    columns = list(by_depth.columns)
    combined = pd.concat([by_depth, overall_export[columns]], ignore_index=True)
    combined.to_csv(output / "dataset_language_results.csv", index=False)

    accuracy = overall.pivot(index="dataset", columns="method", values="accuracy")[FIXED]
    ranks = accuracy.rank(axis=1, ascending=False, method="average")
    rank_long = ranks.stack().rename("rank").reset_index().merge(
        overall[["dataset", "method", "accuracy"]], on=["dataset", "method"]
    )
    rank_long.to_csv(output / "language_ranks.csv", index=False)

    winners = []
    for dataset, values in accuracy.iterrows():
        ordered = sorted(FIXED, key=lambda method: (-values[method], FIXED.index(method)))
        winners.append(
            {
                "dataset": dataset,
                "winner": ordered[0],
                "winner_accuracy": values[ordered[0]],
                "runner_up": ordered[1],
                "runner_up_accuracy": values[ordered[1]],
                "winning_margin": values[ordered[0]] - values[ordered[1]],
                "tie_at_best": int(sum(np.isclose(values, values.max(), atol=1e-12)) > 1),
            }
        )
    winners_frame = pd.DataFrame(winners)
    winners_frame.to_csv(output / "language_winners.csv", index=False)

    heatmap(accuracy, output / "language_accuracy_heatmap.pdf", "Fixed-language accuracy", ".3f", "viridis")
    heatmap(ranks, output / "language_accuracy_rank_heatmap.pdf", "Fixed-language accuracy rank (1 is best)", ".2f", "viridis_r")

    win_counts = winners_frame["winner"].value_counts().reindex(FIXED, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar([DISPLAY[m] for m in FIXED], win_counts.values)
    ax.set_ylabel("Datasets won (deterministic tie break)")
    ax.tick_params(axis="x", rotation=25)
    save(fig, output / "language_wins.pdf")

    avg_ranks = ranks.mean().reindex(FIXED)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar([DISPLAY[m] for m in FIXED], avg_ranks.values)
    ax.set_ylabel("Average dataset rank (lower is better)")
    ax.tick_params(axis="x", rotation=25)
    save(fig, output / "language_average_ranks.pdf")
    return overall, ranks


def mix64(value: int) -> int:
    mask = (1 << 64) - 1
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & mask
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def binarize_labels(values: np.ndarray) -> np.ndarray:
    labels, counts = np.unique(values, return_counts=True)
    if len(labels) == 2:
        return (values == labels[1]).astype(np.uint8)
    majority = labels[np.argmax(counts)]
    return (values == majority).astype(np.uint8)


def load_dl8(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.int64, comments="#", ndmin=2)
    y = binarize_labels(data[:, 0])
    x = data[:, 1:].astype(np.uint8)
    keep = np.var(x.astype(float), axis=0) > 1e-12
    return x[:, keep], y


def train_indices(n: int, seed: int) -> np.ndarray:
    keyed = sorted(((mix64((seed ^ index) & ((1 << 64) - 1)), index) for index in range(n)))
    train_len = round(n * 0.7)
    train_len = min(max(train_len, 1), max(n - 1, 1))
    return np.asarray([index for _key, index in keyed[:train_len]], dtype=np.int64)


def entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def information_gain(y: np.ndarray, mask: np.ndarray) -> float:
    left = y[mask]
    right = y[~mask]
    if not len(left) or not len(right):
        return 0.0
    parent = entropy(np.bincount(y, minlength=2))
    return parent - len(left) / len(y) * entropy(np.bincount(left, minlength=2)) - len(right) / len(y) * entropy(np.bincount(right, minlength=2))


def literal_pool(x: np.ndarray, y: np.ndarray, beam: int = 8) -> list[tuple[int, bool, float]]:
    literals: list[tuple[int, bool, float]] = []
    for feature in range(x.shape[1]):
        gain = information_gain(y, x[:, feature].astype(bool))
        literals.append((feature, True, gain))
        literals.append((feature, False, gain))
    literals.sort(key=lambda item: -item[2])
    return literals[: max(beam, 2)]


def literal_mask(x: np.ndarray, literal: tuple[int, bool, float]) -> np.ndarray:
    feature, positive, _gain = literal
    values = x[:, feature].astype(bool)
    return values if positive else ~values


def root_signals(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    unary = max((information_gain(y, x[:, feature].astype(bool)) for feature in range(x.shape[1])), default=0.0)
    pool = literal_pool(x, y)
    masks = [literal_mask(x, literal) for literal in pool]
    horn = 0.0
    antihorn = 0.0
    clauses: list[tuple[int, int, np.ndarray]] = []
    for i, j in combinations(range(len(pool)), 2):
        if pool[i][0] == pool[j][0] and pool[i][1] != pool[j][1]:
            continue
        clause_mask = masks[i] | masks[j]
        clauses.append((i, j, clause_mask))
        positives = int(pool[i][1]) + int(pool[j][1])
        if positives <= 1:
            horn = max(horn, information_gain(y, clause_mask))
        if 2 - positives <= 1:
            antihorn = max(antihorn, information_gain(y, clause_mask))
    square = 0.0
    seen: set[bytes] = set()
    for index, (_i, _j, first) in enumerate(clauses):
        for second in [first, *(clause[2] for clause in clauses[index + 1 :])]:
            candidate = first & second
            signature = np.packbits(candidate).tobytes()
            if signature in seen:
                continue
            seen.add(signature)
            square = max(square, information_gain(y, candidate))

    ranked_features = sorted(
        range(x.shape[1]),
        key=lambda feature: (-information_gain(y, x[:, feature].astype(bool)), feature),
    )[:8]
    affine = 0.0
    for arity in (2, 3):
        for selected in combinations(ranked_features, arity):
            parity = np.bitwise_xor.reduce(x[:, selected].astype(bool), axis=1)
            affine = max(affine, information_gain(y, parity), information_gain(y, ~parity))
    return {
        "unary_signal": unary,
        "horn_signal": horn,
        "antihorn_signal": antihorn,
        "square2cnf_signal": square,
        "affine_signal": affine,
    }


def structural_signatures(data_dir: Path, output: Path) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for path in sorted(data_dir.glob("*.dl8")):
        x, y = load_dl8(path)
        fold_signals = []
        for run in range(10):
            indices = train_indices(len(y), 42 + run)
            fold_signals.append(root_signals(x[indices], y[indices]))
        record: dict[str, float | int | str] = {
            "dataset": path.stem,
            "n_samples": len(y),
            "n_features": x.shape[1],
            "n_over_p": len(y) / x.shape[1],
            "positive_rate": float(y.mean()),
            "imbalance_ratio": float(max(y.mean(), 1 - y.mean()) / max(min(y.mean(), 1 - y.mean()), 1 / len(y))),
            "feature_prevalence_mean": float(x.mean(axis=0).mean()),
            "feature_prevalence_std": float(x.mean(axis=0).std(ddof=0)),
            "feature_prevalence_min": float(x.mean(axis=0).min()),
            "feature_prevalence_max": float(x.mean(axis=0).max()),
            "feature_sparsity": float(1.0 - x.mean()),
            "signal_source": "training_fold_only_same_boolean_candidate_semantics",
            "folds": 10,
        }
        for signal in fold_signals[0]:
            values = np.asarray([fold[signal] for fold in fold_signals])
            record[signal] = float(values.mean())
            record[f"{signal}_std"] = float(values.std(ddof=1))
        unary = float(record["unary_signal"])
        for family in FIXED[1:]:
            signal_name = f"{family}_signal"
            record[f"interaction_advantage_{family}"] = float(record[signal_name]) - unary
        records.append(record)
    frame = pd.DataFrame(records).sort_values("dataset")
    frame.to_csv(output / "dataset_structural_signatures.csv", index=False)
    return frame


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int, resamples: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        selected = rng.integers(0, len(x), size=len(x))
        if np.unique(x[selected]).size < 2 or np.unique(y[selected]).size < 2:
            continue
        values.append(stats.spearmanr(x[selected], y[selected]).statistic)
    if not values:
        return math.nan, math.nan
    return tuple(np.quantile(values, [0.025, 0.975]))


def specialization(signatures: pd.DataFrame, overall: pd.DataFrame, output: Path) -> pd.DataFrame:
    accuracy = overall.pivot(index="dataset", columns="method", values="accuracy")[FIXED]
    joined = signatures.set_index("dataset").join(accuracy, how="inner")
    records = []
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for index, family in enumerate(FIXED):
        signal = f"{family}_signal"
        advantage = joined[family] - joined[[item for item in FIXED if item != family]].mean(axis=1)
        result = stats.spearmanr(joined[signal], advantage)
        lower, upper = bootstrap_spearman(joined[signal].to_numpy(), advantage.to_numpy(), 20260823 + index)
        records.append(
            {
                "analysis_label": "EXPLORATORY STRUCTURAL CHARACTERIZATION",
                "family": family,
                "signal": signal,
                "advantage_definition": f"accuracy_{family} - mean_accuracy_other_fixed",
                "n_datasets": len(joined),
                "spearman_rho": result.statistic,
                "p_value_uncorrected": result.pvalue,
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
                "bootstrap_resamples": 10_000,
                "statistical_unit": "dataset",
            }
        )
        ax = axes.flat[index]
        ax.scatter(joined[signal], advantage, s=18, alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"{DISPLAY[family]}: rho={result.statistic:.2f}")
        ax.set_xlabel(signal)
        ax.set_ylabel("Accuracy advantage")
    axes.flat[-1].axis("off")
    save(fig, output / "structural_signal_vs_language_advantage.pdf")
    frame = pd.DataFrame(records)
    frame.to_csv(output / "language_specialization_summary.csv", index=False)
    latex = frame[["family", "spearman_rho", "bootstrap_ci_lower", "bootstrap_ci_upper", "p_value_uncorrected"]].to_latex(
        index=False, float_format=lambda value: f"{value:.3f}", escape=True
    )
    (output / "language_specialization_summary.tex").write_text(latex, encoding="utf-8")
    return frame


def oracle_analysis(full: pd.DataFrame, overall: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed = overall.pivot(index="dataset", columns="method", values="accuracy")[FIXED]
    records = []
    for dataset, row in fixed.iterrows():
        ordered = sorted(FIXED, key=lambda method: (-row[method], FIXED.index(method)))
        records.append(
            {
                "dataset": dataset,
                "oracle_accuracy": row[ordered[0]],
                "oracle_language": ordered[0],
                "tie_count": int(np.isclose(row, row.max(), atol=1e-12).sum()),
            }
        )
    oracle = pd.DataFrame(records)
    oracle.to_csv(output / "oracle_fixed_language.csv", index=False)
    adaptive = (
        full[full["method"].isin(ADAPTIVE)]
        .groupby(["dataset", "method"], as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "adaptive_accuracy"})
        .merge(oracle, on="dataset", how="left")
    )
    adaptive["regret"] = adaptive["oracle_accuracy"] - adaptive["adaptive_accuracy"]
    adaptive["interpretation"] = np.select(
        [adaptive["regret"] < -1e-12, adaptive["regret"].abs() <= 1e-12],
        ["beats_every_fixed_language", "matches_best_fixed_language"],
        default="underperforms_best_fixed_language",
    )
    adaptive.to_csv(output / "oracle_vs_adaptive.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    values = [adaptive.loc[adaptive["method"] == method, "regret"] for method in ADAPTIVE]
    ax.boxplot(values, tick_labels=[DISPLAY[m] for m in ADAPTIVE], showmeans=True)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Fixed-language oracle regret")
    ax.tick_params(axis="x", rotation=20)
    save(fig, output / "oracle_regret.pdf")

    cals = adaptive[adaptive["method"] == "cals"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(cals["oracle_accuracy"], cals["adaptive_accuracy"], s=22)
    bounds = [min(cals["oracle_accuracy"].min(), cals["adaptive_accuracy"].min()), 1.0]
    ax.plot(bounds, bounds, "k--", linewidth=0.8)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel("Best fixed-language accuracy")
    ax.set_ylabel("CALS accuracy")
    save(fig, output / "oracle_vs_cals_scatter.pdf")
    return oracle, adaptive


def parse_counts(value: str) -> dict[str, int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)
    return {str(key): int(count) for key, count in parsed.items()}


def structural_group_by_dataset(signatures: pd.DataFrame) -> dict[str, str]:
    groups: dict[str, str] = {}
    advantage_columns = {
        "horn": "interaction_advantage_horn",
        "antihorn": "interaction_advantage_antihorn",
        "square2cnf": "interaction_advantage_square2cnf",
        "affine": "interaction_advantage_affine",
    }
    for row in signatures.itertuples(index=False):
        advantages = {
            family: float(getattr(row, column))
            for family, column in advantage_columns.items()
        }
        family, advantage = max(
            advantages.items(), key=lambda item: (item[1], -FIXED.index(item[0]))
        )
        groups[row.dataset] = (
            f"{family}_interaction" if advantage > 1e-12 else "unary_dominant"
        )
    return groups


def cals_usage(
    full: pd.DataFrame, signatures: pd.DataFrame, output: Path
) -> pd.DataFrame:
    records = []
    structural_groups = structural_group_by_dataset(signatures)
    rename = {
        "Unary": "unary",
        "Horn": "horn",
        "AntiHorn": "antihorn",
        "Square2Cnf": "square2cnf",
        "Affine": "affine",
    }
    for row in full[full["method"].isin(["cals", "cals_compact_explain"])].itertuples():
        counts = parse_counts(row.selected_family_counts)
        total = sum(counts.values())
        for raw_family, count in counts.items():
            records.append(
                {
                    "dataset": row.dataset,
                    "run": row.run,
                    "depth": row.depth,
                    "method": row.method,
                    "structural_dataset_group": structural_groups[row.dataset],
                    "node_depth": 0 if total == 1 else "unavailable_in_frozen_csv",
                    "family": rename.get(raw_family, raw_family),
                    "predicate_arity": "unavailable_in_frozen_csv",
                    "predicate_literals": "unavailable_in_frozen_csv",
                    "gain": "unavailable_in_frozen_csv",
                    "survived_pruning": True,
                    "is_root": True if total == 1 else "unavailable_in_frozen_csv",
                    "node_count": count,
                    "source_granularity": "exact_final_tree_family_counts",
                }
            )
    usage = pd.DataFrame(records)
    usage.to_csv(output / "cals_language_usage.csv", index=False)

    (
        usage.groupby(
            ["method", "structural_dataset_group", "family"], as_index=False
        )["node_count"]
        .sum()
        .sort_values(["method", "structural_dataset_group", "family"])
        .to_csv(output / "cals_language_usage_by_structural_group.csv", index=False)
    )

    total = usage.groupby("family")["node_count"].sum().reindex(FIXED, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar([DISPLAY[m] for m in FIXED], total.values)
    ax.set_ylabel("Final internal nodes")
    ax.tick_params(axis="x", rotation=25)
    save(fig, output / "cals_family_usage.pdf")

    depth = usage.groupby(["depth", "family"])["node_count"].sum().unstack(fill_value=0).reindex(columns=FIXED, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bottom = np.zeros(len(depth))
    for family in FIXED:
        ax.bar(depth.index.astype(str), depth[family], bottom=bottom, label=DISPLAY[family])
        bottom += depth[family].to_numpy()
    ax.set_xlabel("Configured maximum tree depth (node depth unavailable in frozen CSV)")
    ax.set_ylabel("Final internal nodes")
    ax.legend(fontsize=7, ncol=2)
    save(fig, output / "cals_family_usage_by_depth.pdf")

    dataset = usage.groupby(["dataset", "family"])["node_count"].sum().unstack(fill_value=0).reindex(columns=FIXED, fill_value=0)
    proportions = dataset.div(dataset.sum(axis=1), axis=0).fillna(0)
    heatmap(proportions, output / "cals_family_usage_by_dataset.pdf", "CALS/Compact final-node family proportions", ".2f", "magma")
    return usage


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    size = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (size - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def rank_biserial(left: np.ndarray, right: np.ndarray) -> float:
    difference = left - right
    nonzero = difference != 0
    if not nonzero.any():
        return 0.0
    ranks = stats.rankdata(np.abs(difference[nonzero]))
    signs = np.sign(difference[nonzero])
    return float((ranks[signs > 0].sum() - ranks[signs < 0].sum()) / ranks.sum())


def statistical_analysis(full: pd.DataFrame, output: Path) -> pd.DataFrame:
    aggregated = full[full["method"].isin(FIXED)].groupby(["dataset", "method"], as_index=False).agg(
        {metric: "mean" for metric in METRICS}
    )
    records: list[dict[str, object]] = []
    for metric, maximize in METRICS.items():
        pivot = aggregated.pivot(index="dataset", columns="method", values=metric)[FIXED]
        ranks = pivot.rank(axis=1, ascending=not maximize, method="average")
        best = pivot.max(axis=1) if maximize else pivot.min(axis=1)
        for method in FIXED:
            values = pivot[method]
            records.append(
                {
                    "analysis_type": "descriptive",
                    "metric": metric,
                    "method": method,
                    "comparison": "",
                    "n_datasets": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(ddof=1),
                    "average_rank": ranks[method].mean(),
                    "dataset_wins_including_ties": np.isclose(values, best, atol=1e-12).sum(),
                }
            )
        friedman = stats.friedmanchisquare(*(pivot[method] for method in FIXED))
        records.append(
            {
                "analysis_type": "friedman",
                "metric": metric,
                "method": "all_fixed",
                "comparison": "all_fixed",
                "n_datasets": len(pivot),
                "statistic": friedman.statistic,
                "p_value": friedman.pvalue,
            }
        )
        pair_rows = []
        p_values = []
        for left_method, right_method in combinations(FIXED, 2):
            left = pivot[left_method].to_numpy()
            right = pivot[right_method].to_numpy()
            if np.allclose(left, right, atol=0, rtol=0):
                statistic, p_value = 0.0, 1.0
            else:
                result = stats.wilcoxon(left, right, zero_method="pratt", alternative="two-sided")
                statistic, p_value = result.statistic, result.pvalue
            pair_rows.append(
                {
                    "analysis_type": "paired_wilcoxon",
                    "metric": metric,
                    "method": "",
                    "comparison": f"{left_method}_vs_{right_method}",
                    "n_datasets": len(pivot),
                    "statistic": statistic,
                    "p_value": p_value,
                    "median_paired_difference_left_minus_right": np.median(left - right),
                    "paired_rank_biserial": rank_biserial(left, right),
                }
            )
            p_values.append(p_value)
        adjusted = holm(p_values)
        for row, corrected in zip(pair_rows, adjusted):
            row["holm_adjusted_p"] = corrected
            records.append(row)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "language_statistics.csv", index=False)
    return frame


def write_readme(output: Path) -> None:
    (output / "README.md").write_text(
        """# Exact-language characterization artifacts

Generate the complete study from the repository root with:

```bash
python3 smart-mdt-rs/tools/theorem_reaudit.py \\
  --input rust_results_final_freeze_r10_1fbfa30 \\
  --output theorem_reaudit.csv

python3 language_analysis/generate_controlled_fixtures.py \\
  --output language_analysis/controlled_fixtures

cd smart-mdt-rs
cargo run --release -- benchmark \\
  --data ../language_analysis/controlled_fixtures \\
  --depths 4 --runs 5 \\
  --methods unary,horn,antihorn,square2cnf,affine \\
  --output ../language_analysis/controlled_raw --seed 314159
cd ..

python3 language_analysis/study.py \\
  --benchmark rust_results_final_freeze_r10_1fbfa30 \\
  --data data \\
  --controlled language_analysis/controlled_raw/full_results.csv \\
  --output language_analysis
```

The theorem reaudit is a mandatory gate. The study refuses non-Boolean,
non-certified, incomplete, or theorem-boundary-violating frozen inputs.

Statistical unit: dataset after averaging runs and depths. Structural signals
are derived independently on each deterministic 70% training fold and then
averaged; held-out labels are never used. Structural correlations are labeled
exploratory. Raw fit times are local-machine measurements.

The controlled compactness tables and the node, literal, and AXp heatmaps use
the already-frozen 0%-noise controlled rows; they do not rerun training.
Smaller values are better in these three figures. Their captions distinguish
perfect predictive accuracy from direct, compact representation of a target's
native logical structure.

The frozen CSV records exact final-tree family counts but not root identity,
per-node depth, per-node gain, or per-node arity. Accordingly,
`cals_language_usage.csv` preserves exact all-node counts and explicitly marks
the unavailable fields instead of inventing them. It also assigns the
training-fold-only structural dataset group, with the corresponding aggregate
in `cals_language_usage_by_structural_group.csv`. The Rust benchmark now
writes diagnostic-only `cals_language_usage.csv` records for each selected
final node (depth, family, arity/literals, recomputed training-node gain,
pruning survival, and root flag).
Reproducing historical per-node details would require retraining rather than
reinterpretation of the frozen artifact.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_plots()

    full = pd.read_csv(args.benchmark / "full_results.csv")
    if len(full) != 7_360 or full["dataset"].nunique() != 46:
        raise SystemExit("refusing incomplete frozen benchmark")
    if not exact_bool(full["theorem_certified"], "theorem_certified").all():
        raise SystemExit("refusing benchmark with non-certified rows")
    metadata = pd.read_csv(args.benchmark / "dataset_metadata.csv")
    if len(metadata) != 46 or not exact_bool(
        metadata["is_binary_features"], "is_binary_features"
    ).all():
        raise SystemExit("refusing non-Boolean or incomplete dataset metadata")

    controlled = parse_controlled(args.controlled, args.output)
    controlled_compactness(controlled, args.output)
    overall, _ranks = fixed_results(full, args.output)
    signatures = structural_signatures(args.data, args.output)
    specialization(signatures, overall, args.output)
    oracle_analysis(full, overall, args.output)
    cals_usage(full, signatures, args.output)
    statistical_analysis(full, args.output)
    write_readme(args.output)


if __name__ == "__main__":
    main()
