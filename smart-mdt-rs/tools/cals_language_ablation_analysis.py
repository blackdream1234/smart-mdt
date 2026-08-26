#!/usr/bin/env python3
"""Dataset-paired analysis for the CALS language ablation.

Runs and depths are averaged inside each dataset × method block. Statistical
tests therefore use datasets, never the repeated run/depth rows, as paired
independent units.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


FAMILIES = ("unary", "horn", "antihorn", "square2cnf", "affine")
PROFILES = (
    ("cals", "cals"),
    ("compact_explain", "cals_compact_explain"),
)
METRICS = (
    ("accuracy", False),
    ("tree_nodes", True),
    ("predicate_literals", True),
    ("mean_axp_length", True),
    ("fit_time", True),
)
REQUIRED_COLUMNS = {
    "dataset",
    "method",
    "allowed_language_set",
    "depth",
    "run",
    "accuracy",
    "tree_nodes",
    "predicate_literals",
    "mean_axp_length",
    "theorem_certified",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_seed(base: int, *parts: str) -> int:
    payload = "|".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def holm_adjust(values: pd.Series) -> pd.Series:
    order = np.argsort(values.to_numpy(dtype=float), kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * float(values.iloc[index]))
        adjusted[index] = min(running, 1.0)
    return pd.Series(adjusted, index=values.index)


def paired_cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    differences = right - left
    if len(differences) < 2:
        return 0.0
    standard_deviation = float(np.std(differences, ddof=1))
    if standard_deviation == 0.0:
        mean = float(np.mean(differences))
        return 0.0 if mean == 0.0 else float(np.copysign(np.inf, mean))
    return float(np.mean(differences) / standard_deviation)


def paired_bootstrap(
    differences: np.ndarray, resamples: int, seed: int
) -> tuple[float, float, float]:
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    offset = 0
    while offset < resamples:
        size = min(512, resamples - offset)
        indices = generator.integers(
            0, len(differences), size=(size, len(differences))
        )
        estimates[offset : offset + size] = differences[indices].mean(axis=1)
        offset += size
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return float(estimates.mean()), float(lower), float(upper)


def method_pairs(available: set[str]) -> list[tuple[str, str, str, str]]:
    output: list[tuple[str, str, str, str]] = []
    for family in FAMILIES:
        reference = family
        for profile, mixed in PROFILES:
            restricted = f"{mixed}_{family}"
            if {reference, restricted}.issubset(available):
                output.append(
                    ("optimisation", profile, reference, restricted)
                )
            if {restricted, mixed}.issubset(available):
                output.append(
                    ("adaptive_language", profile, restricted, mixed)
                )
    return output


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS.difference(raw.columns))
    if missing:
        raise ValueError(f"missing required columns: {','.join(missing)}")
    if "total_fit_time" in raw:
        raw["fit_time"] = pd.to_numeric(raw["total_fit_time"], errors="raise")
    elif "train_time" in raw:
        raw["fit_time"] = pd.to_numeric(raw["train_time"], errors="raise")
    else:
        raise ValueError("input requires total_fit_time or train_time")
    for metric, _ in METRICS:
        raw[metric] = pd.to_numeric(raw[metric], errors="raise")
    for column in ("run", "depth"):
        raw[column] = pd.to_numeric(raw[column], errors="raise").astype(int)
    key = ["dataset", "method", "run", "depth"]
    if raw.duplicated(key).any():
        raise ValueError("duplicate dataset/method/run/depth rows")
    certified = raw["theorem_certified"].astype(str).str.lower()
    if not certified.isin(("true", "false")).all():
        raise ValueError("theorem_certified is not Boolean")
    raw["theorem_certified"] = certified == "true"
    restricted = raw[raw["ablation_variant"] == "single_language"]
    if not restricted.empty:
        if restricted["allowed_language_set"].astype(str).str.contains(r"\|").any():
            raise ValueError("single-language row reports more than one allowed family")
        if not restricted["theorem_certified"].all():
            raise ValueError("one or more restricted rows failed theorem certification")
    return raw


def pilot_table(results: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "dataset": results["dataset"],
            "method": results["method"],
            "optimizer_profile": results["optimizer_profile"],
            "allowed_language_set": results["allowed_language_set"],
            "ablation_variant": results["ablation_variant"],
            "depth": results["depth"],
            "run": results["run"],
            "accuracy": results["accuracy"],
            "nodes": results["tree_nodes"],
            "predicate_literals": results["predicate_literals"],
            "mean_axp": results["mean_axp_length"],
            "fit_time": results["fit_time"],
            "theorem_certified": results["theorem_certified"],
        }
    )
    for source, target in (
        ("leaves", "leaves"),
        ("max_depth_reached", "reached_depth"),
        ("root_language", "root_language"),
        ("selected_family_counts", "family_counts_final_tree"),
    ):
        if source in results:
            table[target] = results[source]
    if {"nodes_before_prune", "nodes_after_prune"}.issubset(results.columns):
        table["pruned_nodes"] = (
            results["nodes_before_prune"] - results["nodes_after_prune"]
        )
    return table.sort_values(["dataset", "method", "depth", "run"])


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    columns = [metric for metric, _ in METRICS]
    return (
        results.groupby(["dataset", "method"], sort=True)[columns]
        .mean()
        .reset_index()
    )


def analyze_pairs(
    aggregated: pd.DataFrame, resamples: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict[str, object]] = []
    tests: list[dict[str, object]] = []
    available = set(aggregated["method"])
    for effect, profile, left_method, right_method in method_pairs(available):
        paired = aggregated[aggregated["method"].isin((left_method, right_method))]
        pivot = paired.pivot(index="dataset", columns="method")
        for metric, lower_is_better in METRICS:
            if (metric, left_method) not in pivot or (metric, right_method) not in pivot:
                raise ValueError(
                    f"incomplete comparison {left_method} vs {right_method} for {metric}"
                )
            block = pivot.loc[:, [(metric, left_method), (metric, right_method)]].dropna()
            left = block[(metric, left_method)].to_numpy(dtype=float)
            right = block[(metric, right_method)].to_numpy(dtype=float)
            if len(left) == 0:
                continue
            differences = right - left
            if np.all(differences == 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(
                    right,
                    left,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic, p_value = float(test.statistic), float(test.pvalue)
            local_seed = stable_seed(seed, effect, profile, left_method, right_method, metric)
            bootstrap_mean, ci_lower, ci_upper = paired_bootstrap(
                differences, resamples, local_seed
            )
            right_better = differences < 0 if lower_is_better else differences > 0
            left_better = differences > 0 if lower_is_better else differences < 0
            ties = np.isclose(differences, 0.0, rtol=0.0, atol=1e-12)
            common = {
                "effect": effect,
                "optimizer_profile": profile,
                "left_method": left_method,
                "right_method": right_method,
                "metric": metric,
                "lower_is_better": lower_is_better,
                "datasets": len(left),
                "left_mean": float(left.mean()),
                "right_mean": float(right.mean()),
                "mean_difference_right_minus_left": float(differences.mean()),
                "right_better_datasets": int(np.count_nonzero(right_better & ~ties)),
                "ties": int(np.count_nonzero(ties)),
                "left_better_datasets": int(np.count_nonzero(left_better & ~ties)),
            }
            comparisons.append(common)
            tests.append(
                {
                    **common,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "bootstrap_mean": bootstrap_mean,
                    "bootstrap_ci_lower": ci_lower,
                    "bootstrap_ci_upper": ci_upper,
                    "bootstrap_resamples": resamples,
                    "bootstrap_seed": local_seed,
                    "cohens_d_paired": paired_cohens_d(left, right),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    test_frame = pd.DataFrame(tests)
    if not test_frame.empty:
        test_frame["p_value_adjusted_holm"] = test_frame.groupby(
            "metric", group_keys=False
        )["p_value"].apply(holm_adjust)
        test_frame["significant_holm_0_05"] = (
            test_frame["p_value_adjusted_holm"] < 0.05
        )
    return comparison_frame, test_frame


def summary_markdown(
    results: pd.DataFrame, comparisons: pd.DataFrame, tests: pd.DataFrame
) -> str:
    dataset_count = results["dataset"].nunique()
    kind = "pilot/exploratory" if dataset_count < 46 else "full ablation"
    certified = int(results["theorem_certified"].sum())
    lines = [
        "# CALS language ablation analysis",
        "",
        f"Evidence status: **{kind}**.",
        "",
        f"- Rows: {len(results)}",
        f"- Datasets (independent paired units): {dataset_count}",
        f"- Methods: {results['method'].nunique()}",
        f"- Theorem-certified rows: {certified}/{len(results)}",
        "- Repeated runs and depths were averaged inside each dataset × method block.",
        "- Inference uses paired Wilcoxon tests, 10,000 paired bootstrap resamples by default, 95% CIs, Holm correction within each metric, and paired Cohen's d.",
        "",
        "Pilot statistics are descriptive and exploratory; they are not a substitute for the planned 46-dataset experiment.",
    ]
    if not comparisons.empty:
        lines.extend(
            [
                "",
                f"Computed {len(comparisons)} effect/metric summaries and {len(tests)} corrected paired tests.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    results = normalize(pd.read_csv(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pilot_table(results)
    aggregated = aggregate(results)
    comparisons, tests = analyze_pairs(
        aggregated, args.bootstrap_resamples, args.seed
    )
    table.to_csv(args.output_dir / "pilot_results_table.csv", index=False)
    aggregated.to_csv(args.output_dir / "dataset_method_means.csv", index=False)
    comparisons.to_csv(args.output_dir / "ablation_comparisons.csv", index=False)
    tests.to_csv(args.output_dir / "statistical_tests.csv", index=False)
    (args.output_dir / "ANALYSIS_SUMMARY.md").write_text(
        summary_markdown(results, comparisons, tests), encoding="utf-8"
    )
    print(
        f"analyzed {len(results)} rows, {results['dataset'].nunique()} datasets, "
        f"{results['method'].nunique()} methods"
    )


if __name__ == "__main__":
    main()
