#!/usr/bin/env python3
"""Fail-closed exact-theorem reaudit of the frozen Smart-MDT benchmark.

The frozen CSV does not serialize trees.  The audit therefore combines every
row's recorded path/AXp evidence with the immutable benchmark commit's closed
predicate enum and generator invariants.  This is sufficient for this corpus:
the historical generators only emitted single Horn/Anti-Horn clauses,
Square2CNF Form I, and one canonical Boolean GF(2) equation per affine node.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path


EXPECTED = {
    "unary": ("Unary", "UnaryBaseline", "UnaryRelation", "StructuralHorn"),
    "horn": ("Horn", "Theorem3", "StarNestedHorn", "StructuralHorn"),
    "antihorn": (
        "AntiHorn",
        "Theorem4",
        "StarNestedAntiHorn",
        "StructuralAntiHorn",
    ),
    "square2cnf": ("Square2Cnf", "Theorem6", "Square2CnfFormI", "TwoSat"),
    "affine": ("Affine", "Theorem5", "SingleGf2Equation", "Gf2Gaussian"),
    "smart_certified": (
        "SmartCertified",
        "Proposition1",
        "PathCompatibleExactRelations",
        "PathCertified",
    ),
    "cals": (
        "SmartCertified",
        "Proposition1",
        "PathCompatibleExactRelations",
        "PathCertified",
    ),
    "cals_compact_explain": (
        "SmartCertified",
        "Proposition1",
        "PathCompatibleExactRelations",
        "PathCertified",
    ),
}

ALLOWED_STATES = {
    "unary": {"uncommitted"},
    "horn": {"uncommitted", "horn"},
    "antihorn": {"uncommitted", "antihorn"},
    "square2cnf": {"uncommitted", "two_sat"},
    "affine": {"uncommitted", "affine_gf2"},
}


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_file(repo: Path, sha: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def audit_historical_source(repo: Path, sha: str) -> list[Violation]:
    required = {
        "smart-mdt-rs/src/logic/predicate.rs": [
            "HornClause(Vec<Literal>)",
            "AntiHornClause(Vec<Literal>)",
            "Square2Cnf {",
            "Affine {",
            "literals.len() <= 128",
        ],
        "smart-mdt-rs/src/search/horn.rs": [
            "let ls = vec![a, b]",
            "filter(|l| l.positive).count() > 1",
        ],
        "smart-mdt-rs/src/search/antihorn.rs": [
            "let ls = vec![a, b]",
            "filter(|l| !l.positive).count() > 1",
        ],
        "smart-mdt-rs/src/search/square2cnf.rs": [
            "Predicate::Square2Cnf { a, b, c, d }",
        ],
        "smart-mdt-rs/src/search/affine.rs": [
            "ranked_boolean_features",
            "Predicate::Affine",
            "literals.sort_by_key",
        ],
        "smart-mdt-rs/src/explain/weak_axp.rs": [
            "is_binary_domain(domain)",
            "certified_opposite_completion_exists",
        ],
        "smart-mdt-rs/src/logic/path_theory.rs": [
            "predicate.certificate_shape_is_valid()",
            "PathTheoryState::AffineGf2",
        ],
    }
    violations: list[Violation] = []
    for path, needles in required.items():
        try:
            source = git_file(repo, sha, path)
        except subprocess.CalledProcessError as error:
            violations.append(Violation("historical_source_missing", f"{path}: {error}"))
            continue
        for needle in needles:
            if needle not in source:
                violations.append(
                    Violation(
                        "historical_invariant_missing",
                        f"{sha}:{path} does not contain required invariant {needle!r}",
                    )
                )
    return violations


def audit_row(row: dict[str, str], all_binary: bool, source_ok: bool) -> list[Violation]:
    problems: list[Violation] = []
    method = row.get("method", "")
    expected = EXPECTED.get(method)
    if expected is None:
        return [Violation("unknown_method", method)]
    family, _theorem, _structural, backend = expected
    checks = [
        (row.get("theorem_certified") == "true", "old theorem_certified is not true"),
        (row.get("theorem_mode_used") == "true", "theorem mode was not used"),
        (row.get("path_certified") == "true", "path was not certified"),
        (row.get("path_violation_count") == "0", "path violation count is nonzero"),
        (
            row.get("all_predicates_backend_allowed") == "true",
            "historical per-predicate/backend gate failed",
        ),
        (row.get("empirical_fallback_used") == "false", "empirical fallback was used"),
        (
            row.get("incompatible_cached_subtree_reused") == "false",
            "incompatible cached subtree was reused",
        ),
        (row.get("language_family") == family, f"language family is not {family}"),
        (row.get("backend") == backend, f"backend is not {backend}"),
        (row.get("n_fail") == "0", "held-out AXp failures are nonzero"),
        (row.get("axp_valid_rate") == "1", "AXp validity rate is not one"),
        (row.get("axp_minimal_rate") == "1", "AXp minimality rate is not one"),
        (all_binary, "dataset has a non-Boolean retained feature"),
        (source_ok, "historical generator/path source invariant failed"),
    ]
    for condition, detail in checks:
        if not condition:
            problems.append(Violation("certificate_boundary", detail))

    if method in ALLOWED_STATES:
        states = set(row.get("path_theory_state", "").split("|"))
        if not states or not states <= ALLOWED_STATES[method]:
            problems.append(
                Violation(
                    "path_theory_mismatch",
                    f"states {sorted(states)} not within {sorted(ALLOWED_STATES[method])}",
                )
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("rust_results_final_freeze_r10_1fbfa30"),
    )
    parser.add_argument("--output", type=Path, default=Path("theorem_reaudit.csv"))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    benchmark = (repo / args.input).resolve() if not args.input.is_absolute() else args.input
    output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output
    full = rows(benchmark / "full_results.csv")
    metadata = rows(benchmark / "dataset_metadata.csv")
    if len(full) != 7_360 or len(metadata) != 46:
        raise SystemExit(
            f"frozen grid mismatch: rows={len(full)} datasets={len(metadata)}; expected 7360/46"
        )
    all_binary = all(
        row.get("is_binary_features") == "true" and row.get("skipped") == "false"
        for row in metadata
    )
    shas = {row.get("git_sha", "") for row in full}
    source_violations: list[Violation] = []
    if len(shas) != 1 or "" in shas:
        source_violations.append(Violation("benchmark_sha", f"non-unique SHA set {shas}"))
    else:
        source_violations.extend(audit_historical_source(repo, next(iter(shas))))
    source_ok = not source_violations

    fields = [
        "dataset",
        "run",
        "depth",
        "method",
        "old_theorem_certified",
        "new_exact_theorem_certified",
        "language_family",
        "domain_regime",
        "theorem_id",
        "structural_check",
        "backend",
        "violation_type",
        "violation_detail",
    ]
    audit_rows: list[dict[str, str]] = []
    failures = 0
    for row in full:
        violations = audit_row(row, all_binary, source_ok)
        failures += bool(violations)
        family, theorem, structural, backend = EXPECTED.get(
            row.get("method", ""), ("Unsupported", "", "Unsupported", "None")
        )
        combined = source_violations + violations
        audit_rows.append(
            {
                "dataset": row.get("dataset", ""),
                "run": row.get("run", ""),
                "depth": row.get("depth", ""),
                "method": row.get("method", ""),
                "old_theorem_certified": row.get("theorem_certified", "false"),
                "new_exact_theorem_certified": str(not combined).lower(),
                "language_family": family,
                "domain_regime": "Boolean" if all_binary else "Unsupported",
                "theorem_id": theorem,
                "structural_check": structural,
                "backend": backend,
                "violation_type": "|".join(sorted({item.kind for item in combined})),
                "violation_detail": " | ".join(item.detail for item in combined),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit_rows)

    if failures or source_violations:
        print("OFFICIAL BENCHMARK REQUIRES REGENERATION")
        return 1
    print("OFFICIAL BENCHMARK EXACT-THEOREM VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
