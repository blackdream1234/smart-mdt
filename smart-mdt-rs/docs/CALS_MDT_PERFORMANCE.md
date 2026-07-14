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

## Quick smoke test

The required release-mode quick benchmark completed all five legacy rows. The
toy accuracies were unary 1.0, Horn 0.0, AntiHorn 1.0, Square2CNF 0.0, and
best-certified 1.0. Their trees had three nodes except AntiHorn, which had five;
all mean AXp lengths were 1.0. This smoke workload is an execution check, not a
performance comparison.

## Paired 46-dataset result

The release-mode one-run benchmark covered 46 datasets, depths 5 and 7, and
both `smart_certified` and `cals` (184 rows). Aggregate results were:

| Method | Accuracy | Grown nodes | Final nodes | Final literals | Mean AXp | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| SmartCertified | 0.865360 | 50.239 | 50.239 | 30.076 | 3.003 | 1.072 |
| CALS-MDT | 0.866431 | 45.913 | 12.522 | 14.826 | 2.712 | 11.769 |

CALS-MDT improved mean accuracy by 0.001071 while reducing final nodes by
75.08%, literals by 50.70%, and mean AXp length by 9.68%. Pruning reduced grown
nodes by 72.73% and changed 89 of 92 CALS trees. Runtime was 10.98 times the
paired SmartCertified runtime, so the runtime acceptance target was not met.

Depth-specific CALS accuracy was 0.863599 at depth 5 and 0.869262 at depth 7.
Final node means were 9.957 and 15.087 respectively. The corresponding
SmartCertified accuracies were 0.864897 and 0.865823, with 34.130 and 66.348
nodes.

## One-run ablation

Every row below represents all 46 datasets at both requested depths (92 rows):

| ID | Configuration | Accuracy | Final nodes | Grown nodes | Literals | Mean AXp | Fit seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| A | SmartCertified greedy | 0.865360 | 50.239 | 50.239 | 30.076 | 3.003 | 1.072 |
| B | CALS scoring only | 0.861507 | 51.043 | 51.043 | 54.065 | 3.079 | 3.463 |
| C | B + bounded branch-and-bound | 0.861507 | 51.043 | 51.043 | 54.065 | 3.079 | 4.688 |
| D | C + cache | 0.861507 | 51.043 | 51.043 | 54.065 | 3.079 | 9.667 |
| E | D + sparse lookahead | 0.861507 | 51.043 | 51.043 | 54.065 | 3.079 | 13.422 |
| F | E + pruning | 0.865426 | 12.696 | 46.565 | 14.554 | 2.764 | 10.298 |
| G | Full CALS + AXp reranking | 0.866884 | 12.913 | 46.500 | 15.707 | 2.721 | 34.193 |
| H | Full CALS without AXp reranking | 0.866431 | 12.522 | 45.913 | 14.826 | 2.712 | 11.769 |
| I | Full CALS serial | 0.866431 | 12.522 | 45.913 | 14.826 | 2.712 | 8.905 |
| J | Full CALS parallel | 0.866431 | 12.522 | 45.913 | 14.826 | 2.712 | 11.769 |

H and J intentionally reuse the same default parallel, no-AXp result. Serial
and parallel output metrics are identical, but eight-thread execution was 32.2%
slower on this machine. AXp reranking improved accuracy by 0.000453 relative to
H, but increased nodes, literals, mean AXp, and runtime; it therefore remains
optional rather than the thesis default.

## Search, cache, and family diagnostics

For the paired CALS result, predicate-mask cache hit rate was 34.61%, candidate
cache hit rate was 46.12%, and subtree-cache hit rate was 0%. The completed
frontier bound rejected 117,996 of 2,498,060 evaluated candidates (4.72%) with
zero exhaustive fallbacks. Selected-family counts were Square2CNF 184, Horn
182, Unary 89, AntiHorn 45, and Affine 30. These counts can include more than
one family in a tree because distinct unary branches may commit independently.

The ablation shows that cache and parallel overhead do not pay for themselves
on this workload. These are implemented, deterministic capabilities rather
than demonstrated speedups.

## Theory and data audit

The paired run and every ablation loaded 46 datasets with zero skipped datasets
and zero feature-label leakage findings. All CALS rows were theorem-certified
and path-certified. Path violations, forbidden predicates/backends, empirical
fallbacks, incompatible cached-subtree reuse, and nonempty theorem rejection
reasons were all zero. The theorem table contained all 184 paired rows and the
empirical table contained none.

Global beam remains heuristic rather than exact. The implemented bounded
branch-and-bound proves equality with bounded exhaustive selection over its
completed candidate frontier; it is not construction-time candidate-space
pruning and does not establish globally optimal trees.

## Measurement caveats

Fit-time comparisons are machine-specific. The default thesis profile performs
an internal grow/prune split, and reported accuracy remains final-test accuracy.
Cache and candidate-pruning rates must be interpreted alongside fallback counts.
Time-limited search is valid anytime search but is excluded from deterministic
comparisons.
