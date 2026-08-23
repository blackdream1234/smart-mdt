# Pre-Merge Research Review

## Review scope

- Branch: `audit/exact-theorem-language-characterization`
- Reviewed commit: `5665dcdd90576c15783242d254342789a15399ab`
- Parent baseline: `b8be361`
- Frozen benchmark commit recorded in every official row: `1fbfa30`
- Branch delta: 89 files, 31,808 insertions, 164 deletions

This review classifies the committed branch delta only. Unrelated modified and
untracked experiment outputs in the local worktree are not part of the branch
commit or this review.

## Files changed and implementation impact

| Group | Count | Impact classification |
|---|---:|---|
| `docs/` | 2 | Theorem contract and language-study report; documentation only. |
| `evaluation/tests/` | 1 | Determinism/statistical-analysis regression tests only. |
| `language_analysis/` | 66 | Controlled fixtures/results, post-hoc analysis code, CSV/TeX/PDF outputs; no learner invocation except the already-completed controlled run. |
| `smart-mdt-rs/src/` | 11 | Exact certification, explanation dispatch, benchmark metadata, and post-selection diagnostics. |
| `smart-mdt-rs/tests/` | 7 | Exact theorem boundary and diagnostic regression tests. |
| `smart-mdt-rs/tools/` | 1 | Read-only frozen-benchmark theorem reaudit. |
| `theorem_reaudit.csv` | 1 | Generated 7,360-row audit evidence. |

The 11 Rust source files have the following bounded effects:

- `src/logic/{theorem,certificate,predicate,language,mod}.rs`: canonical exact
  relation recognizers and fail-closed structured theorem certificates.
- `src/explain/{weak_axp,axp_deletion}.rs`: exact domain/path certificate checks
  and solver dispatch for explanation verification.
- `src/eval/{benchmark,report}.rs`: exact certificate columns, theorem-table
  filtering, and diagnostic CSV output.
- `src/tree/{adaptive,training}.rs`: post-selection traversal of the already
  selected/pruned final tree to record node-family usage. The traversal is
  explicitly diagnostic-only and executes after model selection.

## Selection-critical audit

The branch changes no files in any of these selection-critical locations:

- `smart-mdt-rs/src/search/` — candidate generation and split scoring;
- `smart-mdt-rs/src/tree/learner.rs` — tree construction and split choice;
- `smart-mdt-rs/src/tree/prune.rs` — pruning decisions;
- `smart-mdt-rs/src/tree/tree_search.rs` — global/greedy search objectives.

It therefore does not change:

- learned split selection;
- candidate scores or tie-breaking;
- tree prediction semantics;
- pruning behaviour;
- the CALS objective;
- the CompactExplain objective;
- seeds, folds, depth limits, methods, or benchmark protocol.

The exact certificate gate can reject a relation or row that cannot prove the
paper's assumptions. That is an intentional change to theorem eligibility and
reporting, not to the learned model. Current generators already emit the
theorem-safe sublanguages verified by the reaudit: two-literal Horn/Anti-Horn,
Square2CNF Form I, and one canonical Boolean GF(2) equation.

## Verification evidence

- Exact reaudit: 7,360/7,360 rows valid, zero violations.
- Independent certification verifier: 7,360 theorem rows, zero empirical rows,
  full held-out AXp validity/minimality, and zero fallback/path/cache violations.
- Independent leakage verifier: 46/46 datasets, zero raw-label, binarized-label,
  or complement leakage.
- Exact theorem regression tests cover star-nested Horn/Anti-Horn, all three
  Square2CNF forms and complements, one-equation GF(2), Boolean-domain guards,
  and path-theory compatibility.

## Decision

**TRAINING SEMANTICS UNCHANGED**

**BENCHMARK REGENERATION NOT REQUIRED**

The frozen 7,360-row result grid remains authoritative for thesis evidence.
