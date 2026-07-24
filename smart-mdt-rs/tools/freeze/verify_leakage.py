#!/usr/bin/env python3
"""Independent feature-label leakage verifier for the Smart-MDT corpus.

This script does NOT call the Rust loader or its metadata. It re-implements the
`.dl8` preprocessing convention (column 0 = label, constant-column removal,
Python-style binarization) from scratch in Python and independently checks each
retained feature against three leakage forms the audit (finding A10) requires to
be absent:

  * raw-label leakage      : feature column exactly equals the raw label column
  * binarized-label leakage: feature column exactly equals the binarized label
  * complement leakage     : feature column exactly equals (1 - binarized label)

It prints a per-dataset table and totals, and exits non-zero if any leakage is
found. Reference: Carbonnel et al. tractable-explaining corpus preprocessing.

Usage: verify_leakage.py <data_dir>
"""
import sys
from collections import Counter
from pathlib import Path


def load_dl8(path):
    rows = []
    width = None
    for line in path.read_text().splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        toks = [int(x) for x in t.split()]
        if width is None:
            width = len(toks)
        elif len(toks) != width:
            raise ValueError(f"{path}: inconsistent row width")
        rows.append(toks)
    return rows


def binarize_python(y):
    labels = sorted(set(y))
    if len(labels) < 2:
        return [0] * len(y)
    if len(labels) > 2:
        counts = Counter(v for v in y if v >= 0)
        majority = min(counts, key=lambda k: (-counts[k], k))  # first max count
        return [1 if v == majority else 0 for v in y]
    positive = labels[1]
    return [1 if v == positive else 0 for v in y]


def non_constant_columns(cols):
    keep = []
    for j, col in enumerate(cols):
        if len(set(col)) > 1:
            keep.append(j)
    return keep


def analyse(path):
    rows = load_dl8(path)
    if not rows:
        return None
    raw_label = [r[0] for r in rows]
    n_features = len(rows[0]) - 1
    feature_cols = [[r[1 + j] for r in rows] for j in range(n_features)]
    keep = non_constant_columns(feature_cols)
    retained = [feature_cols[j] for j in keep]
    binlabel = binarize_python(raw_label)
    complement = [1 - b for b in binlabel]

    raw_hits = sum(1 for col in retained if col == raw_label)
    bin_hits = sum(1 for col in retained if col == binlabel)
    comp_hits = sum(1 for col in retained if col == complement)
    return {
        "dataset": path.stem,
        "n_samples": len(rows),
        "n_retained": len(retained),
        "distinct_labels": len(set(binlabel)),
        "raw_label_leakage": raw_hits,
        "binarized_label_leakage": bin_hits,
        "complement_leakage": comp_hits,
    }


def main(data_dir):
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.dl8"))
    print(f"independent leakage verifier over {len(files)} .dl8 files in {data_dir}")
    print(f"{'dataset':28s} {'n':>7} {'feat':>5} {'raw':>4} {'bin':>4} {'comp':>5}")
    total_raw = total_bin = total_comp = 0
    analysed = 0
    for f in files:
        r = analyse(f)
        if r is None:
            continue
        analysed += 1
        total_raw += r["raw_label_leakage"]
        total_bin += r["binarized_label_leakage"]
        total_comp += r["complement_leakage"]
        print(
            f"{r['dataset']:28s} {r['n_samples']:>7} {r['n_retained']:>5} "
            f"{r['raw_label_leakage']:>4} {r['binarized_label_leakage']:>4} "
            f"{r['complement_leakage']:>5}"
        )
    print("-" * 60)
    print(f"datasets analysed: {analysed}")
    print(f"TOTAL raw-label leakage:        {total_raw}")
    print(f"TOTAL binarized-label leakage:  {total_bin}")
    print(f"TOTAL complement leakage:       {total_comp}")
    ok = total_raw == 0 and total_bin == 0 and total_comp == 0
    print()
    print("PASS: zero raw / binarized / complement leakage" if ok else "FAIL: leakage found")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "../data"))
