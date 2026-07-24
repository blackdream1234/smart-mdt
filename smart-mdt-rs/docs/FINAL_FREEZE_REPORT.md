# Executive Summary

This report records the final verification and regeneration of the Smart-MDT
thesis artifact. Historical pre-audit result folders are explicitly excluded
from the evidence boundary.

| Item | Value |
|---|---|
| Repository implementation commit | `ac4f181e2219d820688ddb79d21d18923c636061` |
| Audit commit | `b00eb8188ea262998e2b8c5df445d3d5e3757dd8` |
| Benchmark commit | `1fbfa303de68c3af1613e8531218f246c0e90693` |
| Evaluation correction commit | `c5b113983bb17eba7318a1ed42868b9f031d2c9d` |
| Datasets | 46 |
| Benchmark rows | 7,360 |
| Figures | 34 files: 17 PNG and 17 PDF |
| Tables | 12 files |
| Reports | 7 publication files: 5 generated source/manifest files and 2 compiled PDFs |
| Rust tests | 161 passed; all configured bench and example targets passed |
| Python tests | 72 passed |

The benchmark artifact is
`rust_results_final_freeze_r10_1fbfa30/`. The canonical evaluation is
`evaluation_final_freeze_r10_c5b1139/`.

## Section 1: Verification of Audit Fixes

Every A01–A20 finding in `CORRECTNESS_AUDIT.md` was inspected against its
implementation and regression coverage. The following final results were
obtained:

| Audit issue | Result | Verification summary |
|---|---|---|
| A01 | PASS | Exact Horn, AntiHorn, 2-SAT, and GF(2) opposite-path AXp solvers pass the 21-feature and 4,000 randomized completion regressions. |
| A02 | PASS | Rust certificates and Python admission require theorem mode. |
| A03 | PASS | Effective Boolean polarity, canonical generated Affine shape, and per-predicate/cumulative 128-variable GF(2) limits are enforced. |
| A04 | PASS | Exact method/language/backend/path-certificate tuples and state/backend sets are enforced in Rust and Python. |
| A05 | PASS | Every theorem tree receives a final path audit; `best-certified` remains empirical. |
| A06 | PASS | SAT clauses are normalized and malformed signed identifiers fail closed. |
| A07 | PASS | AXp dimensions, row indices, selected features, and tree scope are validated before indexing. |
| A08 | PASS | Every predicate family round-trips through deterministic JSON and malformed JSON is recoverable. |
| A09 | PASS | Sparse label counting and lowest-label majority ties match the Python protocol. |
| A10 | PASS | Raw-label, binarized-label, and complement leakage are independently checked. |
| A11 | PASS | Missing expected validation classes count as zero pruning support. |
| A12 | PASS | Greedy node budgets, zero budgets, and invalid pruning fractions are enforced explicitly. |
| A13 | PASS | AXps cover every held-out row and carry exact validity, minimality, success, and failure counts. |
| A14 | PASS | Unknown, missing, conflicting, wrong-command, malformed numeric, and invalid enum arguments fail explicitly. |
| A15 | PASS | Debug candidate output rejects unsupported non-root locations. |
| A16 | PASS | Benchmark and debug CSV writers preserve embedded quotes by RFC-style doubling. |
| A17 | PASS | Statistics use 46 independent dataset blocks after averaging runs and depths. |
| A18 | PASS | Holm correction covers all 15 configured hypotheses and conclusions use adjusted p-values. |
| A19 | PASS | Certification evidence is complete, exact, bounded, cross-partition checked, and includes full-row AXp evidence. |
| A20 | PASS | Method grids, diagnostic keys/domains, empirical methods, and empty comparisons fail closed or render with a complete schema. |

Freeze verification reproduced and repaired five additional boundary defects:

1. CLI values could consume a following option, command-specific flags could
   be silently ignored, and unknown commands could succeed.
2. Python theorem admission did not bind a method to its exact certificate
   tuple.
3. Python theorem admission ignored `theorem_mode_used`.
4. Rust and Python theorem admission did not require a self-contained,
   full-held-out-row AXp evidence tuple.
5. Both Affine generators retained gain-ranked literal order even though the
   GF(2) certificate requires strict feature order. The fix canonicalizes the
   chosen feature set; XOR semantics, masks, and numerical scores are
   unchanged. The regression
   `affine_generators_canonicalize_gain_ranked_feature_order` covers both
   generators.

A later AXp caching experiment was reverted by `ac4f181` because this freeze
task explicitly prohibited performance optimization. It was not used to
generate or admit the benchmark artifact.

### Final verification gates

The following commands pass on the final non-optimized implementation:

```text
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
python -m pytest -q evaluation/tests
python -m compileall -q evaluation
git diff --check
```

The Rust suite executes 161 tests plus all configured bench and example
targets. The Python suite executes 72 tests.

The committed corpus regression and a separate Python implementation both
inspect exactly 46 deterministically ordered datasets. They confirm valid
binarization, non-empty retained features, label-column exclusion, and zero
raw-label, binarized-label, or complement leakage.

## Section 2: Benchmark Verification

The initial one-run validation found and reproduced a genuine Affine
certificate-order defect. After its correction, the all-row one-run process
passed the former HTRU-2 failure point but was interrupted by the execution
environment. The completed ten-run artifact contains a complete run-0 slice
of 736 cells and therefore supplies the intended one-pass grid in addition to
the other nine runs.

The full benchmark was executed from `1fbfa30` with strict data checks:

| Item | Result |
|---|---|
| Output | `rust_results_final_freeze_r10_1fbfa30/` |
| Datasets processed | 46 / 46 |
| Methods | 8 |
| Runs | 10, numbered 0–9 |
| Depths | 5 and 7 |
| Expected/actual rows | 7,360 / 7,360 |
| Approximate wall time | 71 h 28 min, from output-directory creation to final CSV write |
| Failed rows | 0 |
| Skipped datasets | 0 |
| Benchmark warnings | 49; all attributed and non-certification-related |

The methods are `unary`, `horn`, `antihorn`, `square2cnf`, `affine`,
`smart_certified`, `cals`, and `cals_compact_explain`, with 920 rows per
method.

Two independent verifiers and the evaluation admission boundary agree:

| Certification gate | Result |
|---|---:|
| Theorem-certified rows | 7,360 / 7,360 |
| Empirical rows | 0 |
| Forbidden predicates/backends | 0 |
| Feature-label leakage | 0 / 46 datasets |
| Path violations | 0 |
| Incompatible cached-subtree reuse | 0 |
| Empirical fallbacks | 0 |
| AXp evidence violations | 0 |
| Full-row AXp violations | 0 |
| Held-out rows certified for AXp sufficiency/minimality | 9,833,600 |
| AXp failures | 0 |

Every theorem row records theorem mode, an exact method/family/backend/path
certificate tuple, compatible path states/backends, an empty rejection reason,
and `post_selection_final_tree` extraction. For every row,
`final_axp_rows = test_rows`, `n_success = test_rows`,
`axp_valid_rate = axp_minimal_rate = 1`, and `n_fail = 0`.

## Section 3: Evaluation Verification

The canonical evaluation was regenerated from the immutable benchmark with
10,000 paired dataset-block bootstrap resamples, seed `20260718`, and 95%
confidence intervals. Repeated runs and depths are averaged within each of 46
dataset blocks before descriptive or inferential analysis.

The evaluator generated:

- 34 figure files;
- 12 table files;
- `evaluation_report.md` and `evaluation_report.tex`;
- `executive_summary.md` and `executive_summary.tex`;
- `reproducibility_manifest.json`;
- compiled 16-page `evaluation_report.pdf`; and
- compiled one-page `executive_summary.pdf`.

The evaluation report's certification section,
`tables/certification_summary.tex`, and this report jointly provide the
certification report requested by the freeze protocol.

Holm's step-down correction covers all 15 configured method-by-metric
hypotheses. The significance table contains raw and adjusted p-values, paired
dataset-block bootstrap intervals, paired Cohen's d, and standard Cliff's
delta as a distributional reference.

The complete evaluation was generated twice into independent directories.
With identical dependency versions and deterministic LaTeX metadata, all 53
publication artifacts—including both compiled PDFs—were byte-identical.

## Section 4: Scientific Consistency

The canonical sources for every current publication claim are
`evaluation_final_freeze_r10_c5b1139/tables/` and
`evaluation_final_freeze_r10_c5b1139/report/`.

Key dataset-block means are:

| Method | Accuracy | Tree nodes | Mean AXp | Fit time (s) |
|---|---:|---:|---:|---:|
| Unary | 0.858646 | 53.235 | 3.927 | 0.047 |
| Horn | 0.859690 | 51.217 | 5.137 | 0.065 |
| AntiHorn | 0.859848 | 51.487 | 5.553 | 0.066 |
| Square2CNF | 0.859064 | 50.807 | 6.357 | 0.444 |
| Boolean Affine/GF(2) | 0.853536 | 53.126 | 7.897 | 0.162 |
| SmartCertified | 0.859578 | 50.689 | 4.184 | 0.731 |
| CALS-MDT | 0.864820 | 12.730 | 2.976 | 8.895 |
| CompactExplain | 0.860157 | 11.124 | 2.721 | 2.861 |

CALS-MDT has the highest mean accuracy. CompactExplain has the smallest mean
tree and shortest mean AXp. Unary has the lowest measured fit time.

For CALS-MDT versus CompactExplain accuracy, CompactExplain is lower by
0.004662, with 95% bootstrap CI `[-0.007302, -0.002171]`, raw
`p = 0.001873`, Holm-adjusted `p = 0.011236`, and paired Cohen's
`d = -0.523` (medium). For SmartCertified versus CALS-MDT tree nodes,
CALS-MDT is lower by 37.959 nodes, with 95% bootstrap CI
`[-45.818, -30.680]`, Holm-adjusted `p = 4.576e-08`, and paired Cohen's
`d = -1.411` (large).

Current generated reports use the regenerated values. Pre-audit numerical
documents are labeled historical and excluded from freeze evidence. Runtime
claims remain explicitly local-machine measurements, not portable guarantees.
No empirical backend or empirical row appears in the theorem results.

## Section 5: Reproducibility

### Fixed configuration

| Setting | Value |
|---|---|
| Benchmark seed | 42 |
| Evaluation seed | 20260718 |
| Bootstrap resamples | 10,000 |
| Confidence level | 0.95 |
| Depths | 5 and 7 |
| Runs | 10 |
| Statistical unit | Dataset block |
| Multiple testing | Holm step-down |

### Software and dependency identity

| Component | Version/evidence |
|---|---|
| Rust compiler | `rustc 1.96.1 (31fca3adb 2026-06-26)` |
| Cargo | `cargo 1.96.1 (356927216 2026-06-26)` |
| Python | `3.13.14` |
| pytest | `9.0.3` |
| pdfTeX | `3.141592653-2.6-1.40.29`, TeX Live 2026/Debian |
| `Cargo.lock` SHA256 | `0a7dfc492523bea3fdf4f54850e6fd16a137b5fac1c2493ede4d66105e1167be` |
| `pyproject.toml` SHA256 | `7528fe401b0eb77aa53c976571a1dec0b87fe4d2e8c912642bd7dd5f59142492` |

### Benchmark command

The completed source run used the following configuration. Its CSVs were
copied byte-for-byte from `rust_results_manual_check/` into the canonical path;
the accompanying README was updated only to standardize the Smart-MDT title
and record the benchmark commit:

```text
cargo run --release -- benchmark \
  --data ../data \
  --depths 5,7 \
  --runs 10 \
  --methods unary,horn,antihorn,square2cnf,affine,smart_certified,cals,cals_compact_explain \
  --output ../rust_results_manual_check \
  --strict-data-checks
```

### Evaluation command

```text
python -m evaluation.report \
  --input rust_results_final_freeze_r10_1fbfa30 \
  --output evaluation_final_freeze_r10_c5b1139 \
  --bootstrap-resamples 10000 \
  --seed 20260718 \
  --confidence-level 0.95
```

The two reproducibility runs additionally set `LC_ALL=C`, `TZ=UTC`,
`PYTHONHASHSEED=0`, single-threaded BLAS/OpenMP variables, and a fixed
`SOURCE_DATE_EPOCH`. The same `SOURCE_DATE_EPOCH` was supplied to both
two-pass LaTeX builds.

`FINAL_FREEZE_SHA256SUMS` records every canonical benchmark CSV, figure, table,
source report, manifest, compiled PDF, and this final report. The benchmark
CSV hashes were unchanged before and after evaluation.

Raw benchmark CSVs include local wall-clock timing measurements, so independent
benchmark executions are not expected to be byte-identical. Reproducibility is
provided by the archived immutable CSV hashes and by byte-identical evaluation
outputs generated from those inputs; changing the timing methodology to force
identical raw CSV bytes was prohibited.

## Section 6: Freeze Decision

READY FOR FREEZE
