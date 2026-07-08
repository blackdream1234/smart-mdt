# Experiment protocol

Use explicit seeds for all randomized extensions. Tuning must use training/validation data only; test labels must not drive candidate selection or tuning.

Run:

```bash
cargo run --release -- benchmark --quick
```

Outputs include `full_results.csv`, `summary_by_method.csv`, `theorem_certified_results.csv`, `empirical_results.csv`, `axp_metadata.csv`, `tuning_diagnostics.csv`, and `README_RESULTS.md`. The theorem table is filtered to Unary, Horn, AntiHorn and Square2CNF with StructuralHorn, StructuralAntiHorn or TwoSat backends.
