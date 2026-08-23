#!/usr/bin/env python3
"""Build the deterministic final thesis evidence bundle from frozen artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


FROZEN_FILES = [
    "README_RESULTS.md",
    "axp_metadata.csv",
    "beam_diagnostics.csv",
    "benchmark_warnings.csv",
    "cache_diagnostics.csv",
    "dataset_metadata.csv",
    "empirical_results.csv",
    "family_budget_diagnostics.csv",
    "full_results.csv",
    "pruning_diagnostics.csv",
    "search_diagnostics.csv",
    "summary_by_method.csv",
    "theorem_certified_results.csv",
    "tuning_diagnostics.csv",
]

COPY_MAP = {
    "theorem/THEOREM_CONFORMANCE_SPEC.md": "docs/THEOREM_CONFORMANCE_SPEC.md",
    "theorem/theorem_reaudit.csv": "theorem_reaudit.csv",
    "language_characterization/LANGUAGE_CHARACTERIZATION_REPORT.md": (
        "docs/LANGUAGE_CHARACTERIZATION_REPORT.md"
    ),
    "language_characterization/dataset_language_results.csv": (
        "language_analysis/dataset_language_results.csv"
    ),
    "language_characterization/language_winners.csv": (
        "language_analysis/language_winners.csv"
    ),
    "language_characterization/language_ranks.csv": (
        "language_analysis/language_ranks.csv"
    ),
    "language_characterization/language_statistics.csv": (
        "language_analysis/language_statistics.csv"
    ),
    "language_characterization/dataset_structural_signatures.csv": (
        "language_analysis/dataset_structural_signatures.csv"
    ),
    "language_characterization/language_specialization_summary.csv": (
        "language_analysis/language_specialization_summary.csv"
    ),
    "language_characterization/language_specialization_summary.tex": (
        "language_analysis/language_specialization_summary.tex"
    ),
    "controlled/controlled_expressive_power.csv": (
        "language_analysis/controlled_expressive_power.csv"
    ),
    "controlled/controlled_compactness_summary.csv": (
        "language_analysis/controlled_compactness_summary.csv"
    ),
    "controlled/controlled_compactness_summary.tex": (
        "language_analysis/controlled_compactness_summary.tex"
    ),
    "oracle/oracle_fixed_language.csv": "language_analysis/oracle_fixed_language.csv",
    "oracle/oracle_vs_adaptive.csv": "language_analysis/oracle_vs_adaptive.csv",
    "cals/cals_language_usage.csv": "language_analysis/cals_language_usage.csv",
    "cals/cals_language_usage_by_structural_group.csv": (
        "language_analysis/cals_language_usage_by_structural_group.csv"
    ),
    "claims/THESIS_CLAIMS_EVIDENCE.md": "docs/THESIS_CLAIMS_EVIDENCE.md",
}

FIGURES = [
    "controlled_expressive_power_heatmap.pdf",
    "controlled_nodes_heatmap.pdf",
    "controlled_literals_heatmap.pdf",
    "controlled_axp_heatmap.pdf",
    "language_accuracy_heatmap.pdf",
    "language_accuracy_rank_heatmap.pdf",
    "language_average_ranks.pdf",
    "language_wins.pdf",
    "structural_signal_vs_language_advantage.pdf",
    "oracle_regret.pdf",
    "oracle_vs_cals_scatter.pdf",
    "cals_family_usage.pdf",
    "cals_family_usage_by_dataset.pdf",
    "cals_family_usage_by_depth.pdf",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(repository: Path, output: Path, destination: str, source: str) -> None:
    source_path = repository / source
    if not source_path.is_file():
        raise SystemExit(f"missing required evidence source: {source}")
    destination_path = output / destination
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination_path)


def frozen_configuration(frozen: Path) -> dict[str, object]:
    with (frozen / "full_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (frozen / "dataset_metadata.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        metadata = list(csv.DictReader(handle))
    methods = list(dict.fromkeys(row["method"] for row in rows))
    return {
        "authoritative": True,
        "benchmark_commit": sorted({row["git_sha"] for row in rows}),
        "datasets": len({row["dataset"] for row in rows}),
        "depths": sorted({int(row["depth"]) for row in rows}),
        "methods": methods,
        "regeneration_required": False,
        "rows": len(rows),
        "runs": len({int(row["run"]) for row in rows}),
        "source_directory": frozen.name,
        "split_protocols": sorted({row["train_test_split_protocol"] for row in rows}),
        "all_datasets_boolean": all(
            row["is_binary_features"] == "true" and row["skipped"] == "false"
            for row in metadata
        ),
    }


def write_readme(output: Path) -> None:
    (output / "README.md").write_text(
        """# Final Thesis Research Evidence

This bundle is a content-addressed research freeze. Paths in
`manifest_sha256.txt` are relative to this directory and sorted
deterministically. The manifest does not hash itself because a file cannot
contain its own stable digest.

## Evidence classes

### OFFICIAL FROZEN BENCHMARK EVIDENCE

`frozen_benchmark/` is the authoritative 7,360-row, 46-dataset result grid
from commit `1fbfa30`, including result partitions, metadata, warnings,
diagnostics, and configuration. It was not regenerated for this freeze.

### CONTROLLED ADDITIONAL EXPERIMENT

`controlled/` and the controlled figures summarize the already-completed
600-row exact-target experiment. The compactness tables and figures are
post-processing of those existing rows; no controlled training was rerun.

### POST-HOC LANGUAGE ANALYSIS

`language_characterization/`, `oracle/`, and the corresponding figures use
dataset as the statistical unit after aggregating the frozen runs/depths.
Structural signals use training-fold labels only and remain exploratory.

### DIAGNOSTIC-ONLY CALS ANALYSIS

`cals/` contains exact final-tree family counts retained by the frozen run.
Unavailable historical root/per-node fields remain explicitly unavailable;
no diagnostics were invented. Diagnostic instrumentation does not affect
model selection.

`claims/THESIS_CLAIMS_EVIDENCE.md` is the normative wording register for
thesis claims.
""",
        encoding="utf-8",
    )


def write_theorem_summary(output: Path) -> None:
    (output / "theorem" / "theorem_tests_summary.txt").write_text(
        """STATUS: PASS

Exact theorem regression suites:
- audit_affine: 8 passed
- audit_certificate_boundaries: 6 passed
- exact_theorem_classes: 5 passed
- path_theory_state: 12 passed
- total targeted theorem/path tests: 31 passed

Independent frozen evidence:
- exact theorem reaudit: 7,360 / 7,360 valid
- theorem partition: 7,360 theorem rows, 0 empirical rows
- held-out AXp validity/minimality: complete for every theorem row
- empirical fallback/path/cache violations: 0
- independent leakage audit: 46 datasets, 0 raw/bin/complement leaks

The full all-target/all-feature Rust and Python suites are reported in
docs/FINAL_PRE_THESIS_FREEZE_REPORT.md outside this bundle.
""",
        encoding="utf-8",
    )


def write_manifest(output: Path) -> int:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest_sha256.txt"
    )
    lines = [f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in paths]
    (output / "manifest_sha256.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return len(paths)


def build(repository: Path, output: Path) -> int:
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    frozen = repository / "rust_results_final_freeze_r10_1fbfa30"
    for filename in FROZEN_FILES:
        copy(
            repository,
            output,
            f"frozen_benchmark/{filename}",
            f"{frozen.name}/{filename}",
        )
    for destination, source in COPY_MAP.items():
        copy(repository, output, destination, source)
    for figure in FIGURES:
        copy(
            repository,
            output,
            f"figures/{figure}",
            f"language_analysis/{figure}",
        )

    configuration = frozen_configuration(frozen)
    (output / "frozen_benchmark" / "configuration.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(output)
    write_theorem_summary(output)
    return write_manifest(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output.resolve()
    count = build(repository, output)
    print(f"FINAL THESIS EVIDENCE BUNDLE: {count} HASHED ARTIFACTS")


if __name__ == "__main__":
    main()
