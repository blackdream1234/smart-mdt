#!/usr/bin/env python3
"""Inspect a .dl8 file using the Python GSNH-MDT parsing convention, without dependencies."""
import argparse
from collections import Counter
from pathlib import Path


def load_raw(path: Path):
    rows = []
    width = None
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        vals = [int(x) for x in line.split()]
        if len(vals) < 2:
            raise ValueError(f"{path}:{line_no}: fewer than 2 columns")
        if width is None:
            width = len(vals)
        elif len(vals) != width:
            raise ValueError(f"{path}:{line_no}: inconsistent width")
        rows.append(vals)
    if not rows:
        raise ValueError(f"{path}: no rows")
    y = [r[0] for r in rows]
    x = [r[1:] for r in rows]
    return x, y


def binarize_labels(y):
    labels = sorted(set(y))
    if len(labels) < 2:
        return [0 for _ in y]
    if len(labels) > 2:
        max_label = max([v for v in y if v >= 0], default=0)
        counts = [0] * (max_label + 1)
        for v in y:
            if v >= 0:
                counts[v] += 1
        majority = max(range(len(counts)), key=lambda i: counts[i])
        return [1 if v == majority else 0 for v in y]
    return [1 if v == labels[1] else 0 for v in y]


def remove_constant_columns(x):
    if not x:
        return [], []
    keep = []
    for j in range(len(x[0])):
        vals = [row[j] for row in x]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        if var > 1e-12:
            keep.append(j)
    return [[row[j] for j in keep] for row in x], keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', type=Path)
    args = ap.parse_args()
    x, y_raw = load_raw(args.path)
    y = binarize_labels(y_raw)
    xp, _keep = remove_constant_columns(x)
    leakage = [j for j in range(len(xp[0]) if xp else 0) if [row[j] for row in xp] == y]
    pos = sum(y) / len(y) if y else 0.0
    print(f"n_samples={len(y)}")
    print(f"n_features_original={len(x[0]) if x else 0}")
    print(f"n_features_after_constant_removal={len(xp[0]) if xp else 0}")
    print(f"raw_label_counts={dict(sorted(Counter(y_raw).items()))}")
    print(f"binarized_label_counts={dict(sorted(Counter(y).items()))}")
    print(f"positive_rate={pos}")
    print(f"majority_rate={max(pos, 1.0 - pos) if y else 0.0}")
    print(f"feature_equal_to_label_count={len(leakage)}")
    print(f"feature_equal_to_label_indices={leakage}")


if __name__ == '__main__':
    main()
