import pandas as pd
from pathlib import Path

python_file = Path("pythonfull_results.csv")
rust_candidates = [
    Path("rust_results/summary_by_method.csv"),
    Path("rust_results/full_results.csv"),
    Path("smart-mdt-rs/experiment_artifacts/smart_mdt_rs/summary_by_method.csv"),
    Path("smart-mdt-rs/experiment_artifacts/smart_mdt_rs/full_results.csv"),
]

rust_file = next((p for p in rust_candidates if p.exists()), None)

if not python_file.exists():
    raise SystemExit(f"Missing Python file: {python_file}")

if rust_file is None:
    raise SystemExit("No Rust result file found. Run the Rust benchmark first.")

py = pd.read_csv(python_file)
rs = pd.read_csv(rust_file)

print("Python file:", python_file)
print("Rust file:", rust_file)
print("\nPython columns:", list(py.columns))
print("Rust columns:", list(rs.columns))

def method_col(df):
    for c in ["method_label", "method", "Method"]:
        if c in df.columns:
            return c
    raise SystemExit(f"No method column found in: {list(df.columns)}")

def accuracy_col(df):
    for c in ["accuracy_mean", "mean_accuracy", "accuracy", "Accuracy"]:
        if c in df.columns:
            return c
    raise SystemExit(f"No accuracy column found in: {list(df.columns)}")

pm = method_col(py)
rm = method_col(rs)
pa = accuracy_col(py)
ra = accuracy_col(rs)

py_small = py[[pm, pa]].copy()
rs_small = rs[[rm, ra]].copy()

py_small.columns = ["method", "python_accuracy"]
rs_small.columns = ["method", "rust_accuracy"]

# Normalize common names
replace = {
    "GSNH-1D": "unary",
    "1D": "unary",
    "GSNH-Horn": "horn",
    "Horn": "horn",
    "GSNH-AntiHorn": "antihorn",
    "AntiHorn": "antihorn",
    "GSNH-Square2CNF": "square2cnf",
    "Square2CNF": "square2cnf",
}
py_small["method_norm"] = py_small["method"].replace(replace).str.lower()
rs_small["method_norm"] = rs_small["method"].replace(replace).str.lower()

merged = py_small.merge(rs_small, on="method_norm", how="outer", suffixes=("_py", "_rs"))
merged["accuracy_delta_rust_minus_python"] = merged["rust_accuracy"] - merged["python_accuracy"]

out = Path("rust_python_accuracy_comparison.csv")
merged.to_csv(out, index=False)

print("\nComparison:")
print(merged.to_string(index=False))
print(f"\nSaved: {out}")
