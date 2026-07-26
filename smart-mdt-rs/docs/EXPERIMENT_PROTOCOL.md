# Experiment protocol

Use explicit seeds for all randomized extensions. Tuning must use training/validation data only; test labels must not drive candidate selection or tuning.

Run:

```bash
cargo run --release -- benchmark --quick
```

Outputs include `full_results.csv`, `summary_by_method.csv`, `theorem_certified_results.csv`, `empirical_results.csv`, `axp_metadata.csv`, `tuning_diagnostics.csv`, and `README_RESULTS.md`. Every row records config, git metadata, `path_theory_state`, `path_backend`, and `path_certified`. The theorem table admits the five certified single-family methods and path-compatible `smart_certified` rows only.

Report certified and empirical results separately. Do not claim speedup over the Python implementation until identical train/test splits, depth limits, min-leaf settings, split families, accuracy, tree size, AXp length, training time and explanation time have been measured.

The learner is greedy/heuristic and the Rust code is tested, not formally verified.

## Full dataset benchmark

The Rust CLI supports recursive `.dl8` discovery and dataset/run/depth/method-level rows:

```bash
cargo run --release -- benchmark \
  --data ../data \
  --depths 5,7 \
  --runs 10 \
  --methods unary,horn,antihorn,square2cnf,affine,smart_certified \
  --output ../rust_results
```

`full_results.csv` contains `dataset`, `run`, `depth`, `method`, accuracy, timing, tree-size, AXp, path-level theorem metadata, git SHA and config columns. `theorem_certified_results.csv` excludes every empirical backend and admits `smart_certified` only when all root-to-leaf paths pass validation.

The `.dl8` loader treats the first column as the class label and all remaining columns as features. This avoids using a discretized feature as the target column and keeps Rust benchmark accuracy from being inflated by label-column leakage.

For comparison with older Python result files, Rust `full_results.csv` includes both native per-run columns (`method`, `accuracy`, `tree_nodes`, `mean_axp_length`, etc.) and compatibility aliases (`method_key`, `method_label`, `acc`, `size`, `expl`, `axp_backend`, `path_certificate`, `random_state`, `n_runs`, and `train_test_split_protocol`).

## Candidate diagnostics

Use `debug-candidates` to inspect root candidate masks and scores before changing the learner:

```bash
cargo run --release -- debug-candidates \
  --data ../data \
  --dataset tic-tac-toe \
  --method horn \
  --top-k 20 \
  --output ../debug_horn_tictactoe
```

The command writes `debug_candidates.csv` and `debug_candidate_masks.csv`. These files are intended for Python-parity debugging of candidate generation, predicate semantics, masks, and scores; they do not change the learner.
