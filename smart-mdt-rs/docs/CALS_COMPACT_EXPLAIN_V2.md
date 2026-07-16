# CALS-MDT CompactExplain v2

## Scope

CompactExplain v2 adds class-aware pruning, selective lookahead, conditional
candidate search and caching, post-selection AXp extraction, and verified
audience-specific explanations to the path-certified CALS-MDT learner. The
existing `smart_certified` and CALS thesis profiles remain available unchanged.

The CompactExplain profile is serial. It keeps subtree caching and provisional
AXp reranking disabled, activates selective lookahead only at eligible nodes,
and extracts exact AXps after the final certified tree has been selected and
pruned.

## Verification gates

The following commands pass from `smart-mdt-rs`:

```text
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

The implementation has regression coverage for warning attribution,
class-aware pruning guards, selective search gating, conditional search and
cache activation, final-tree AXp extraction, verified explanation compilation,
deterministic audience rendering, CLI output, and theorem-table admission.

## One-run comparison

Command:

```text
cargo run --release -- benchmark \
  --data ../data \
  --depths 5,7 \
  --runs 1 \
  --methods smart_certified,cals,cals_compact_explain \
  --output ../rust_results_cals_compact_r1 \
  --strict-data-checks
```

The run produced 276 rows: 46 datasets, two depths, one run, and three methods.
Timing is a local-machine measurement and should not be treated as portable.

| Method | Rows | Mean accuracy | Mean nodes | Mean AXp | Mean fit time (s) |
|---|---:|---:|---:|---:|---:|
| SmartCertified | 92 | 0.865359838102 | 50.239130434783 | 3.002717391304 | 0.759499147054 |
| CALS-MDT | 92 | 0.866430768251 | 12.521739130435 | 2.711956521739 | 8.599762105359 |
| CALS-MDT CompactExplain v2 | 92 | 0.867853287397 | 9.173913043478 | 2.694293478261 | 2.809002084815 |

Against the unchanged CALS profile, CompactExplain increased mean accuracy by
0.001423, reduced mean nodes by 26.74%, reduced mean AXp length by 0.65%, and
reduced mean fit time by 67.34%. It satisfies the requested thresholds of CALS
accuracy minus 0.002, at most 15 nodes, at most 2.70 mean AXp length, and at
most six seconds mean fit time.

## Certification and data audit

- Datasets loaded: 46
- Skipped datasets: 0
- Feature-label leakage findings: 0
- Path compatibility violations: 0
- Rows with a forbidden predicate backend: 0
- Empirical fallbacks: 0
- Theorem-certified rows: 276 / 276
- CompactExplain theorem-certified rows: 92 / 92
- Rows in the empirical table: 0
- Incompatible cached-subtree reuse findings: 0
- CompactExplain provisional AXp evaluations: 0
- CompactExplain rows using parallel execution: 0

Every CompactExplain row records `post_selection_final_tree` as its AXp
extraction stage. Balanced-accuracy, sensitivity, specificity, macro-F1,
minority-recall, class-support, and pruning-reason fields are present in the
benchmark output.

Across CompactExplain rows, mean validation balanced accuracy was
0.792858417966 before pruning and 0.801166966376 after pruning. Mean validation
minority recall was 0.675828558775 before pruning and 0.674221512079 after
pruning.

## Search diagnostics

- Nodes using greedy selection: 2,454
- Nodes using selective lookahead: 1,139
- Conditional branch-and-bound activations: 0
- Direct-search decisions avoiding branch-and-bound: 2,820
- Cache activations: 5,698
- Estimated candidate-generation work saved: 45,402
- Subtree-cache hits: 0, as the CompactExplain profile disables that cache

The absence of branch-and-bound activations in this run is expected: candidate
pools stayed below the configured activation threshold, so the cheaper direct
bounded path was used.

## Constant-root and warning audit

CompactExplain selected constant roots for these dataset/depth pairs:

- `balance-scale-bin`: depths 5 and 7
- `letter`: depths 5 and 7
- `seismic_bumps-bin`: depths 5 and 7
- `winequality-red-bin`: depths 5 and 7

The root-collapse guard rejected constant replacements for `bank_conv-bin`
(depths 5 and 7), `forest-fires-un` (depths 5 and 7), `taiwan_binarised`
(depths 5 and 7), `wine1-un` (depth 5), and `wine3-un` (depths 5 and 7).

The benchmark emitted 25 warnings, and every warning contains a non-empty
dataset key. Grouped all-constant and all-zero-AXp warnings also record method,
affected-row count, runs, depths, warning type, and reason.

## Verified explanations

The `explain` command writes:

- `verified_explanation.json`, the deterministic, audited explanation schema;
- `human_explanation.txt`, rendered exclusively from that JSON.

Supported audiences are `general`, `clinical`, `engineering`, `management`,
`audit`, and `technical`. Clinical rendering describes a model prediction,
does not assert diagnosis or causation, reports support and uncertainty, and
recommends professional review when confidence or support is low.

## Limitations

- The benchmark is a one-run validation, not an estimate of cross-run timing
  variance or generalization to other datasets.
- Profile thresholds were validated on the repository benchmark corpus and
  remain configurable for different class distributions or risk tolerances.
- Counterfactuals are restricted to certified reference rows; an explanation
  reports the limitation when no such reference is available.
- AXp reranking remains an optional experimental flag and is disabled in the
  default CompactExplain profile.
- Generated benchmark directories are local evidence and are not tracked by
  Git.
