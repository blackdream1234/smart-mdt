#!/usr/bin/env python3
"""Compare before/after benchmark CSVs, excluding timing-only columns."""
import csv
import sys
from pathlib import Path

TIMING_COLUMNS = {
    "train_time", "predict_time", "axp_time", "total_fit_time", "search_time",
    "pruning_time", "axp_rerank_time", "git_sha",  # git_sha differs because commit changed
}

def rows_of(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def key(r):
    return (r["dataset"], r["run"], r["depth"], r["method"])

def compare_file(name, before_dir, after_dir):
    b = rows_of(Path(before_dir) / name)
    a = rows_of(Path(after_dir) / name)
    bk = {key(r): r for r in b}
    ak = {key(r): r for r in a}
    errors = []
    if set(bk) != set(ak):
        errors.append(f"{name}: row-set mismatch: only-before={set(bk)-set(ak)} only-after={set(ak)-set(bk)}")
    for k in sorted(set(bk) & set(ak)):
        br, ar = bk[k], ak[k]
        cols = set(br) | set(ar)
        for c in sorted(cols - TIMING_COLUMNS):
            bv, av = br.get(c), ar.get(c)
            if bv != av:
                errors.append(f"{name} {k} column={c}: before={bv!r} after={av!r}")
    print(f"{name}: {len(b)} before rows, {len(a)} after rows, {len(errors)} differences (excl. timing/git_sha)")
    for e in errors[:30]:
        print("  -", e)
    return len(errors)

def main(before_dir, after_dir):
    total = 0
    for name in ["full_results.csv", "theorem_certified_results.csv", "empirical_results.csv"]:
        total += compare_file(name, before_dir, after_dir)
    print()
    if total == 0:
        print("PASS: zero non-timing differences between before-fix and after-fix output")
        return 0
    print(f"FAIL: {total} non-timing differences found")
    return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
