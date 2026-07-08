# Experiment protocol

Use explicit seeds for all randomized extensions. Tuning must use training/validation data only; test labels must not drive candidate selection or tuning.

Run:

```bash
cargo run --release -- benchmark --quick
```

Outputs include `full_results.csv`, `summary_by_method.csv`, `theorem_certified_results.csv`, `empirical_results.csv`, `axp_metadata.csv`, `tuning_diagnostics.csv`, and `README_RESULTS.md`. Every row records config and git metadata. The theorem table is filtered to Unary, Horn, AntiHorn and Square2CNF with StructuralHorn, StructuralAntiHorn or TwoSat backends.

Report certified and empirical results separately. Do not claim speedup over the Python implementation until identical train/test splits, depth limits, min-leaf settings, split families, accuracy, tree size, AXp length, training time and explanation time have been measured.

The learner is greedy/heuristic and the Rust code is tested, not formally verified.

## Full dataset benchmark

The Rust CLI supports recursive `.dl8` discovery and dataset/run/depth/method-level rows:

```bash
cargo run --release -- benchmark \
  --data ../data \
  --depths 5,7 \
  --runs 10 \
  --methods unary,horn,antihorn,square2cnf \
  --output ../rust_results
```

`full_results.csv` contains `dataset`, `run`, `depth`, `method`, accuracy, timing, tree-size, AXp, theorem metadata, git SHA and config columns. `theorem_certified_results.csv` is filtered to only `unary`, `horn`, `antihorn`, and `square2cnf` rows with certified backends.
