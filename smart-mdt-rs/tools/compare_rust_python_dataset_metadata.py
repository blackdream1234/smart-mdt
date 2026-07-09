#!/usr/bin/env python3
"""Compare Rust dataset_metadata.csv against Python-equivalent .dl8 parsing."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_inspector_output(text):
    out = {}
    for line in text.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--rust-metadata', required=True, type=Path)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.rust_metadata.open()))
    by_name = {r['dataset']: r for r in rows}
    ok = True
    inspector = Path(__file__).with_name('inspect_python_dl8_reference.py')
    for path in sorted(args.data.rglob('*.dl8')):
        name = path.stem
        if name not in by_name:
            print(f"missing rust metadata for {name}")
            ok = False
            continue
        proc = subprocess.run([sys.executable, str(inspector), str(path)], text=True, check=True, capture_output=True)
        py = parse_inspector_output(proc.stdout)
        rust = by_name[name]
        checks = [
            ('n_samples', int(rust['n_samples']), int(py['n_samples'])),
            ('n_features_after_constant_removal', int(rust['n_features_after_constant_removal']), int(py['n_features_after_constant_removal'])),
        ]
        for key, rv, pv in checks:
            if rv != pv:
                print(f"{name}: {key} differs rust={rv} python={pv}")
                ok = False
        rp = float(rust['positive_rate'])
        pp = float(py['positive_rate'])
        if abs(rp - pp) > 1e-12:
            print(f"{name}: positive_rate differs rust={rp} python={pp}")
            ok = False
    if not ok:
        raise SystemExit(1)
    print("Rust metadata matches Python-equivalent parser for checked fields.")


if __name__ == '__main__':
    main()
