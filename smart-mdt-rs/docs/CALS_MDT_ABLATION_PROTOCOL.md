# CALS-MDT ablation protocol

## Partition rule

Hyperparameters, search decisions, pruning, family pilots, and AXp reranking may
use only the benchmark training partition or its deterministic internal
validation subset. The final test partition is used only for reported accuracy
and final AXp evaluation.

## Smoke and one-run commands

Run the smoke test first:

```text
cargo run --release -- benchmark --quick
```

Then run the paired one-run comparison:

```text
cargo run --release -- benchmark \
  --data ../data \
  --depths 5,7 \
  --runs 1 \
  --methods smart_certified,cals \
  --output ../rust_results_cals_optimization_r1 \
  --strict-data-checks
```

## Ablations

Use distinct output directories and one configuration change at a time:

- A: `smart_certified` greedy baseline.
- B: CALS scoring only: greedy, `--no-branch-and-bound --no-cache
  --no-parallel --no-adaptive-language --no-prune`.
- C: B plus `--branch-and-bound`.
- D: C plus `--cache`.
- E: D plus `--tree-search lookahead --lookahead-depth 2`.
- F: E plus `--prune`.
- G: full CALS with `--cals-profile thesis --axp-rerank`.
- H: full CALS without AXp reranking: the default thesis profile.
- I: full CALS with `--no-parallel`.
- J: full CALS with `--parallel`.

The CLI also supports `--no-*` forms for reproducible ablations even though the
positive flags are the documented minimum interface.

## Required audit

Before any 10-run experiment, require 46 loaded datasets, the expected row
count, zero skipped datasets, zero feature-label leakage, zero path violations,
zero forbidden backends, no empirical fallback, and 100% theorem-table CALS
rows. Report accuracy, nodes before/after pruning, literals, mean/max AXp,
runtime, cache hit rates, bounded-selection pruning rate, and selected-family
distribution. Failed targets must be reported rather than tuned on test labels.
