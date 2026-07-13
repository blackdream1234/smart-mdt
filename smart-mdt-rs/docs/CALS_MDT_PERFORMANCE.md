# CALS-MDT performance report

## Baseline

The immutable pre-optimization baseline is recorded in
`CALS_OPTIMIZATION_BASELINE.md`. Across 92 SmartCertified rows it measured mean
accuracy 0.865011914977, 50.239130434783 nodes, mean AXp length
3.025815217391, and mean fit time 2.182994191283 seconds on the development
machine.

## Implementation-level evidence

- Random small-data tests prove bitset counts, masks, impurity, gain, balance,
  and majority decisions equal the former naive computation.
- Exact bounded top-k tests compare branch-and-bound with bounded exhaustive
  selection for every certified family.
- Cache on/off and bounded eviction return identical trees.
- Serial, two-thread, four-thread, and default-thread runs return identical
  candidate lists, trees, and theorem/path metadata.
- Greedy, sparse-lookahead, and global-beam tests enforce depth/node budgets,
  deterministic completion, and path certification.
- Pruning tests prove non-increasing nodes/literals and certification.

## Benchmark status

The post-implementation quick test and 46-dataset one-run ablation are recorded
in this document during Phase 14. Performance targets are not claimed before
those measurements. In particular, global beam is not described as exact and
the completed-frontier branch-and-bound is not described as construction-time
candidate-space pruning.

## Measurement caveats

Fit-time comparisons are machine-specific. The default thesis profile performs
an internal grow/prune split, and reported accuracy remains final-test accuracy.
Cache and candidate-pruning rates must be interpreted alongside fallback counts.
Time-limited search is valid anytime search but is excluded from deterministic
comparisons.
