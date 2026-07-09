# Performance baseline

This document records the current Rust-only baseline. It is not a Python speedup claim.

## Environment

- Command: `cargo bench`
- Build mode: Cargo bench profile / optimized release-like build
- Date: 2026-07-08 UTC
- CPU: Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz
- CPU count visible to container: 3
- Architecture: x86_64

## What was benchmarked

The checked-in benchmark binaries measure:

1. safe predicate mask evaluation versus direct naive row evaluation on a 512-row, 8-feature synthetic Boolean dataset;
2. candidate search time for Unary, Horn, AntiHorn and Square2CNF on the same synthetic dataset;
3. Horn, AntiHorn and 2-SAT backend throughput over repeated 64-variable formulas;
4. deletion-based AXp extraction on a small four-feature binary-domain tree.

## Current output

```text
AXp extraction: 413.329µs last=[1]
SAT backend horn: 4.972787ms last=true
SAT backend antihorn: 9.330522ms last=true
SAT backend two_sat: 14.968801ms last=true
safe bitset-like mask evaluation: 11.358µs count=256
naive row evaluation: 2.531µs count=256
candidate search unary: 296.082µs candidates=8
candidate search horn: 71ns candidates=0
candidate search antihorn: 111ns candidates=28
candidate search square2cnf: 80ns candidates=5
```

## Interpretation boundary

These numbers are smoke-test baselines for the new crate. They are not statistically robust Criterion benchmarks, and they do not compare against the older Python implementation. No speedup over Python, no asymptotic performance claim, and no production scalability claim should be made from this table alone.

Future work should replace these binaries with Criterion benchmarks, pin CPU frequency where possible, run multiple input sizes, and report confidence intervals.
