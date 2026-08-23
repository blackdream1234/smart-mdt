# Final Pre-Thesis Freeze Report

# Executive Summary

| Item | Final freeze value |
|---|---|
| Current branch | `audit/exact-theorem-language-characterization` |
| Reviewed theorem/language commit | `5665dcdd90576c15783242d254342789a15399ab` |
| Frozen benchmark commit | `1fbfa30` |
| Frozen datasets | 46 |
| Frozen rows | 7,360 |
| Frozen methods | 8: Unary, Horn, Anti-Horn, Square2CNF, Affine, SmartCertified, CALS, CALS CompactExplain |
| Frozen runs/depths | 10 runs; depths 5 and 7 |
| Controlled experiment | 600 rows; 8 targets × 3 noise levels × 5 methods × 5 runs |
| Exact theorem reaudit | 7,360/7,360 valid; zero violations |
| Rust validation | 172 tests passed; all targets/features; Clippy clean with warnings denied; format clean |
| Python validation | 76 tests passed; `compileall` clean |
| Language artifact regeneration | 29/29 byte-identical; 0 non-identical |
| Evidence bundle | 49 hashed artifacts; 50 files including the manifest |
| SHA256 manifest | `thesis_evidence_final/manifest_sha256.txt` |
| Manifest SHA256 | `ec997590e6f54eebd6c57a5eea842d5d16ce673ccb53f2824cab4dd160032b8f` |

The reviewed scientific baseline is the branch commit above. The Git commit
that adds this final freeze report and evidence bundle is assigned only after
the report is written; its exact published hash is reported in the PR/final
handoff.

# 1. Exact theorem status

The Boolean theorem boundary is implemented fail-closed for Unary,
star-nested Horn, star-nested Anti-Horn, one Boolean GF(2) equation, and exact
Square2CNF Forms I–III/constants. Complement construction, Boolean-domain
guards, Proposition-1 path compatibility, and backend matching are explicit
certificate fields rather than method-name inference.

The exact reaudit was rerun from the tool, not trusted as a stale copy. All
7,360 rows passed. The targeted theorem/path suites contributed 31 passing
tests; the complete Rust suite contributed 172 passing tests.

# 2. Frozen benchmark status

The authoritative artifact remains
`rust_results_final_freeze_r10_1fbfa30/`: 46 Boolean datasets, 10 runs, two
depths, eight methods, and 7,360 rows from commit `1fbfa30`.

Independent verification found:

- 7,360 theorem rows and zero empirical rows;
- complete held-out AXp extraction for every test row;
- AXp validity and minimality equal to 1.0 on every theorem row;
- zero AXp failures, empirical fallbacks, path violations, or incompatible
  cached-subtree reuse;
- exact family/backend/path-certificate correspondence;
- zero raw-label, binarized-label, or complement leakage across all 46
  datasets.

The branch does not modify split scoring, candidate generation, learner tree
construction, pruning, tree-search objectives, CALS objectives,
CompactExplain objectives, seeds, splits, or benchmark protocol. Exact
certification and post-selection diagnostics do not change learned models.

**TRAINING SEMANTICS UNCHANGED.**

**BENCHMARK REGENERATION NOT REQUIRED.**

# 3. Language characterization status

The fixed-language analysis uses dataset as the statistical unit after
averaging runs/depths. Per-dataset results, ranks, winners, corrected tests,
training-fold-only structural signatures, and exploratory specialization
results are complete. All study inputs are theorem-certified Boolean rows.

Independent regeneration produced 29 CSV, Markdown, LaTeX, and PDF artifacts:
29 were byte-identical and zero differed. An independently rebuilt evidence
bundle was also byte-identical to `thesis_evidence_final/`.

# 4. Controlled expressive-power status

The existing 600-row controlled experiment covers Unary, simple/chained
star-nested Horn, the chained Anti-Horn dual, Square2CNF Forms I–III, and XOR3
at 0%, 5%, and 10% deterministic label noise. Every native family achieved
perfect accuracy on its clean target.

No controlled training was rerun. The authoritative controlled rows were
reused exactly.

# 5. Controlled compactness status

Accuracy alone did not reveal specialization because several non-native
families could represent a small Boolean target with a larger tree. The new
0%-noise node, predicate-literal, and AXp heatmaps explicitly mark smaller
values as better and explain this distinction in their captions.

The source-derived XOR3 comparison is:

| Family | Accuracy | Mean nodes | Predicate literals | Mean AXp |
|---|---:|---:|---:|---:|
| Boolean Affine | 1.000 | 3 | 3 | 3.000 |
| Square2CNF | 1.000 | 15 | 28 | 3.000 |
| Unary | 1.000 | 23 | 11 | 3.000 |

The 40-row compactness table records accuracy gap and node/literal ratios
relative to the native family for every clean target/family pair. The XOR3
values are asserted against source rows by both the generator and regression
test; they are not inserted as replacement benchmark results.

# 6. Fixed-language statistical conclusions

With deterministic tie-breaking, fixed-language wins were Unary 19,
Square2CNF 10, Horn 10, Anti-Horn 4, and Affine 3. The mean winner/runner-up
margin was only 0.00239, so the win count is descriptive rather than evidence
of a large practical separation.

After Holm correction, no pairwise accuracy difference among Unary, Horn,
Anti-Horn, and Square2CNF was significant. Affine was lower on this corpus
than Unary (adjusted p=0.026986, paired rank-biserial 0.489), Horn
(0.000213, 0.684), Anti-Horn (0.000273, 0.673), and Square2CNF
(0.000342, 0.661).

All structural-signal bootstrap intervals crossed zero. Root logical signals
therefore did not establish a predictor of relative family advantage in this
46-dataset corpus.

# 7. Oracle conclusions

The oracle selects the best one fixed family globally for each dataset; it is
not an optimum over all mixed-language trees.

| Adaptive method | Mean regret | Beats fixed oracle | Matches | Underperforms |
|---|---:|---:|---:|---:|
| SmartCertified | 0.004718 | 7 | 0 | 39 |
| CALS | -0.000524 | 21 | 2 | 23 |
| CALS CompactExplain | 0.004138 | 14 | 0 | 32 |

CALS can exceed a globally fixed family by choosing different theorem
families locally, but it does not do so uniformly.

# 8. CALS language-use conclusions

The frozen final-tree counts establish that CALS used all five certified
families: Unary 18.5%, Horn 29.4%, Anti-Horn 10.9%, Square2CNF 34.7%, and
Affine 6.6% of recorded final internal nodes.

Historical per-node depth, root identity, arity, and gain were not present in
the frozen CSV and were not invented. The report uses exact all-node counts;
new instrumentation records richer fields only for future runs and has no
selection effect.

# 9. Claims that are NOT supported

- There is no universal best fixed family.
- Structural root signal does not predict the best family in this corpus.
- CALS does not always beat the fixed oracle.
- The current Horn learner does not search the full maximal star-nested class;
  it emits a theorem-safe two-literal sublanguage.
- The current Square2CNF learner emits Form I only; it does not search the
  complete maximal Square2CNF class.
- Ordered finite-domain theorem classes are not implemented.

The normative allowed/forbidden wording for 14 major claims is in
`THESIS_CLAIMS_EVIDENCE.md`.

# 10. Freeze decision

All theorem, frozen-evidence, compactness, claims, manifest, reproducibility,
formatting, lint, Rust, Python, leakage, and theorem-table gates passed.

**READY FOR THESIS REPORT**
