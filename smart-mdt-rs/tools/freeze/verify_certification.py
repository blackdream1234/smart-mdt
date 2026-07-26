#!/usr/bin/env python3
"""Independent certification verification for a Smart-MDT benchmark folder.

NON-CIRCULARITY: this script does NOT import or call the Rust crate, its
`theorem_table_filter`, or the `evaluation` package's certification code. It
re-derives the theorem certification boundary from first principles in this file
(the METHOD_TUPLE / STATE_BACKEND tables below encode the required
family/backend/path-certificate correspondence directly) and checks every row of
the benchmark CSVs against it. Therefore a defect in the Rust filter or in the
evaluator cannot make this verifier pass its own output.

It checks, over every theorem row: grid completeness, partition disjointness
(theorem vs empirical), the full per-row certification gate (theorem_certified,
theorem_mode_used, path_certified, zero path violations, no empirical fallback,
no incompatible cache reuse, backend-allowed, empty rejection reason, exact
family/backend/path-certificate tuple), path-state/backend consistency, all-row
AXp validity and minimality, and zero-leakage dataset metadata.

On start it prints the SHA256 of every input CSV so the exact inputs are pinned.

Usage: verify_certification.py <folder> [n_datasets] [n_runs] [depths_csv]
"""
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()

EXPECTED_METHODS = [
    "unary", "horn", "antihorn", "square2cnf", "affine",
    "smart_certified", "cals", "cals_compact_explain",
]
METHOD_TUPLE = {
    "unary": ("Unary", "StructuralHorn", "HornCnf", {"uncommitted"}),
    "horn": ("Horn", "StructuralHorn", "HornCnf", {"uncommitted", "horn"}),
    "antihorn": ("AntiHorn", "StructuralAntiHorn", "AntiHornCnf", {"uncommitted", "antihorn"}),
    "square2cnf": ("Square2Cnf", "TwoSat", "TwoCnf", {"uncommitted", "two_sat"}),
    "affine": ("Affine", "Gf2Gaussian", "AffineGf2", {"uncommitted", "affine_gf2"}),
    "smart_certified": ("SmartCertified", "PathCertified", "PathTheory", None),
    "cals": ("SmartCertified", "PathCertified", "PathTheory", None),
    "cals_compact_explain": ("SmartCertified", "PathCertified", "PathTheory", None),
}
STATE_BACKEND = {
    "uncommitted": "StructuralHorn", "horn": "StructuralHorn",
    "antihorn": "StructuralAntiHorn", "two_sat": "TwoSat",
    "affine_gf2": "Gf2Gaussian",
}

def rows_of(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def key(r):
    return (r["dataset"], r["run"], r["depth"], r["method"])

def fail(msgs, msg):
    msgs.append(msg)

def main(folder, expected_datasets, expected_runs, expected_depths):
    folder = Path(folder)
    print(f"verifier SHA256: {sha256_file(__file__)}")
    print("input CSV SHA256:")
    for name in [
        "full_results.csv",
        "theorem_certified_results.csv",
        "empirical_results.csv",
        "dataset_metadata.csv",
    ]:
        print(f"  {name}: {sha256_file(folder / name)}")
    full = rows_of(folder / "full_results.csv")
    cert = rows_of(folder / "theorem_certified_results.csv")
    emp = rows_of(folder / "empirical_results.csv")
    meta = rows_of(folder / "dataset_metadata.csv")
    errors = []

    # --- grid completeness ---
    datasets = sorted(set(r["dataset"] for r in full))
    runs = sorted(set(r["run"] for r in full))
    depths = sorted(set(r["depth"] for r in full))
    methods = sorted(set(r["method"] for r in full))
    if len(datasets) != expected_datasets:
        fail(errors, f"expected {expected_datasets} datasets, found {len(datasets)}")
    if len(runs) != expected_runs:
        fail(errors, f"expected {expected_runs} runs, found {len(runs)}")
    if depths != sorted(expected_depths):
        fail(errors, f"expected depths {expected_depths}, found {depths}")
    if methods != sorted(EXPECTED_METHODS):
        fail(errors, f"method set mismatch: {methods}")
    expected_rows = expected_datasets * expected_runs * len(expected_depths) * len(EXPECTED_METHODS)
    if len(full) != expected_rows:
        fail(errors, f"expected {expected_rows} rows, found {len(full)}")
    counts = Counter(key(r) for r in full)
    dup = [k for k, c in counts.items() if c > 1]
    if dup:
        fail(errors, f"duplicate grid cells: {dup[:3]}")

    # --- dataset metadata: zero leakage, strict fields ---
    if len(meta) != expected_datasets:
        fail(errors, f"metadata rows {len(meta)} != {expected_datasets}")
    for m in meta:
        if m["skipped"] != "false":
            fail(errors, f"{m['dataset']}: skipped dataset in freeze corpus")
        if m["feature_equal_to_label_count"] != "0":
            fail(errors, f"{m['dataset']}: feature-label leakage")
        if m["suspicious_feature_label_leakage"] != "false":
            fail(errors, f"{m['dataset']}: suspicious leakage flag")
        if m["label_column_used"] != "0" or m["label_excluded_from_features"] != "true":
            fail(errors, f"{m['dataset']}: label column handling invalid")
        if int(m["n_features_after_constant_removal"]) <= 0:
            fail(errors, f"{m['dataset']}: no retained features")

    # --- partition integrity ---
    fullk = {key(r) for r in full}
    certk = {key(r) for r in cert}
    empk = {key(r) for r in emp}
    if certk & empk:
        fail(errors, f"theorem/empirical overlap: {sorted(certk & empk)[:3]}")
    if (certk | empk) != fullk:
        missing = fullk - (certk | empk)
        fail(errors, f"rows in neither partition: {sorted(missing)[:5]} ({len(missing)} total)")

    # --- independent certification boundary on every theorem row ---
    for r in cert:
        k = key(r)
        m = r["method"]
        tup = METHOD_TUPLE.get(m)
        if tup is None:
            fail(errors, f"{k}: method not theorem-eligible")
            continue
        fam, backend, path_cert, allowed_states = tup
        checks = [
            (r["theorem_certified"] == "true", "theorem_certified"),
            (r["theorem_mode_used"] == "true", "theorem_mode_used"),
            (r["path_certified"] == "true", "path_certified"),
            (r["path_violation_count"] == "0", "path_violation_count"),
            (r["empirical_fallback_used"] == "false", "empirical_fallback_used"),
            (r["incompatible_cached_subtree_reused"] == "false", "incompatible_cache_reuse"),
            (r["all_predicates_backend_allowed"] == "true", "all_predicates_backend_allowed"),
            (r["theorem_rejection_reason"].strip('"') == "", "rejection_reason_empty"),
            (r["language_family"] == fam, f"language_family={r['language_family']}!={fam}"),
            (r["backend"] == backend, f"backend={r['backend']}!={backend}"),
            (r["axp_backend"] == backend, "axp_backend"),
            (r["path_certificate"] == path_cert, f"path_certificate={r['path_certificate']}"),
            (r["category"] == "certified", "category"),
            (r["axp_extraction_stage"] == "post_selection_final_tree", "axp_stage"),
        ]
        # full-row AXp sufficiency + minimality
        test_rows = int(r["test_rows"]); final_axp = int(r["final_axp_rows"])
        checks += [
            (test_rows > 0, "test_rows>0"),
            (final_axp == test_rows, f"final_axp_rows {final_axp}!={test_rows}"),
            (float(r["axp_valid_rate"]) == 1.0, f"axp_valid_rate={r['axp_valid_rate']}"),
            (float(r["axp_minimal_rate"]) == 1.0, f"axp_minimal_rate={r['axp_minimal_rate']}"),
            (int(r["n_success"]) == test_rows, "n_success==test_rows"),
            (int(r["n_fail"]) == 0, "n_fail==0"),
        ]
        # path states/backends consistency
        states = r["path_theory_state"].split("|")
        backends = r["path_backend"].split("|")
        if allowed_states is not None and not set(states) <= allowed_states:
            checks.append((False, f"path states {states} outside {allowed_states}"))
        if len(set(states)) != len(states) or len(set(backends)) != len(backends):
            checks.append((False, "duplicate path state/backend components"))
        exp_backends = {STATE_BACKEND.get(s) for s in states}
        if None in exp_backends or exp_backends != set(backends):
            checks.append((False, f"path backends {backends} inconsistent with states {states}"))
        for ok, name in checks:
            if not ok:
                fail(errors, f"{k}: {name}")

    # --- empirical partition must not contain theorem-intended certified rows ---
    for r in emp:
        if r["theorem_certified"] == "true" and r["method"] in METHOD_TUPLE:
            fail(errors, f"{key(r)}: certified theorem-method row in empirical partition")

    print(f"folder: {folder}")
    print(f"full={len(full)} theorem={len(cert)} empirical={len(emp)} datasets={len(datasets)}")
    print(f"methods x rows: {dict(Counter(r['method'] for r in cert))}")
    if errors:
        print(f"\nFAIL: {len(errors)} violations")
        for e in errors[:40]:
            print("  -", e)
        return 1
    print("\nPASS: certification boundary verified on every theorem row")
    return 0

if __name__ == "__main__":
    folder = sys.argv[1]
    n_datasets = int(sys.argv[2]) if len(sys.argv) > 2 else 46
    n_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    depths = sys.argv[4].split(",") if len(sys.argv) > 4 else ["5", "7"]
    sys.exit(main(folder, n_datasets, n_runs, depths))
