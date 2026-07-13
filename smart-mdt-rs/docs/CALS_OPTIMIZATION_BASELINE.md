# CALS-MDT optimization baseline

## Scope

This baseline was captured from base commit
`71d09617f0a5e88aeedee7659cb541a64bc4f29c` before optimization work on
`feature/cals-optimization-engine`. Timings are measurements from the local
development machine and should be compared only with runs made under the same
conditions.

The base branch contained trailing whitespace in `src/data/binning.rs`, so the
first `cargo fmt --check` failed. That formatting-only defect was removed before
the baseline gates and benchmarks below were accepted.

## Search terminology and current behavior

- The current learner is greedy: it selects one split at the current node and
  recursively repeats that decision.
- Its `beam_width` is a **candidate beam width** used inside family candidate
  generation at one node. The default is 8.
- `max_candidates_per_node` caps the combined ranked pool at 64 candidates.
- There is no global partial-tree beam in this baseline. A future **tree beam
  width** will count competing partial trees and must remain distinct from the
  candidate beam width.
- The theorem-certified single-family methods are `unary`, `horn`, `antihorn`,
  `square2cnf`, and certified Boolean `affine` using `Gf2Gaussian`.
- `smart_certified` is path-certified rather than globally single-family. It
  greedily searches all compatible certified families while carrying
  `PathTheoryState`; different branches may commit independently, but one path
  may not mix incompatible committed families.

## Verification gates

Commands:

```text
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

After the formatting-only repair, all commands passed. The test suite contained
59 integration tests, with no unit-test or doc-test failures.

## Quick benchmark

Command:

```text
cargo run --release -- benchmark --quick \
  --output ../rust_results_cals_optimization_baseline_quick
```

The run wrote five rows. Unary, AntiHorn, and BestCertified achieved accuracy
1.0 on the synthetic split; Horn and Square2CNF achieved 0.0. Every tree had 3
nodes and mean AXp length 1. Four rows entered the theorem table; the legacy
`best-certified` method remained excluded by its reporting policy.

## SmartCertified one-run benchmark

Command:

```text
cargo run --release -- benchmark \
  --data ../data \
  --depths 5,7 \
  --runs 1 \
  --methods smart_certified \
  --output ../rust_results_cals_optimization_baseline_r1 \
  --strict-data-checks
```

| Depth | Rows | Mean accuracy | Mean nodes | Mean AXp length | Mean fit time (s) |
|---:|---:|---:|---:|---:|---:|
| 5 | 46 | 0.865078876218 | 34.130434782609 | 2.945652173913 | 2.010812333565 |
| 7 | 46 | 0.864944953736 | 66.347826086957 | 3.105978260870 | 2.355176049000 |
| All | 92 | 0.865011914977 | 50.239130434783 | 3.025815217391 | 2.182994191283 |

Total measured fit time was 200.835465598 seconds. All 92 result rows entered
`theorem_certified_results.csv`; none entered `empirical_results.csv`.

## Invariant audit

- Datasets loaded: 46
- Skipped datasets: 0
- Feature-label leakage findings: 0
- Path compatibility violations: 0
- SmartCertified theorem-certified rows: 92 / 92
- Empirical SmartCertified rows: 0
- Data warnings: one `suspicious_majority_rate` warning for
  `winequality-red-bin` (majority rate 0.9937460913070669)

Generated benchmark directories are intentionally local and are not tracked by
Git.
