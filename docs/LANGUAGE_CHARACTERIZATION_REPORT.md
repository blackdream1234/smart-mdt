# Exact-Theorem Language Characterization Report

## 1. Exact theoretical boundary

The normative specification is Carbonnel, Cooper, Hebrard, Morales, and
Marques-Silva, *Explaining Multivariate Decision Trees: Characterising
Tractable Languages* (supplied journal version, 2025). The implementation
contract is recorded in [THEOREM_CONFORMANCE_SPEC.md](THEOREM_CONFORMANCE_SPEC.md).

For a finite Boolean language closed under complement, Theorem 7 permits weak
AXp tractability only when every node relation is equivalent to at least one
of: star-nested Horn, star-nested Anti-Horn, one GF(2) equation, or square
2CNF. Ordinary Horn, ordinary Anti-Horn, arbitrary affine CNF, and arbitrary
2CNF are not substitutes for these complement-closed node classes.

Proposition 1 additionally requires exact complement closure, supported
assignment constraints, and a polynomial-time CSP super-language for every
opposite-class path. A node relation and a path CSP are different objects: a
Square2CNF node must satisfy Theorem 6, while a path may be a general 2CNF; an
affine node is one equation, while an affine path is a system of equations.

The repository and frozen corpus are Boolean only. It does not claim the
complete ordered finite-domain result. The `OrderedFiniteUI` metadata value is
reserved, but current ordered-domain requests fail closed as `Unsupported`.
In particular, no arity-three-or-higher UI-affine relation is certified.

## 2. Theorem-to-code mapping

The exact class representations and recognizers are in
`smart-mdt-rs/src/logic/theorem.rs`:

- `StarNestedHorn` checks normalized Horn polarity and a total inclusion chain
  of negative sets. Its complement follows Proposition 2 recursively and is
  revalidated.
- `StarNestedAntiHorn` is the exact polarity dual and uses the dual complement
  construction.
- `SingleGf2Equation` cancels repeated variables modulo two, sorts variables,
  represents `0=0`/`0=1`, and complements by flipping the RHS.
- `Square2CnfRelation` recognizes constants and Forms I–III and implements the
  exact I↔III and II↔II complement map.

`TheoremCertificate` records domain regime, family, theorem, structural check,
complement check, backend, assignment support, and path check. The legacy
family/backend constructor cannot certify a result. Predicate construction,
tree/path compatibility, actual Boolean-domain checks, path normalization,
solver dispatch, and final held-out AXp sufficiency/minimality must all pass.

Current certified learners intentionally use safe sublanguages:

| Learner | Generated node relation | Exact certificate |
|---|---|---|
| Unary | one Boolean literal | Unary baseline |
| Horn | one two-literal Horn clause | Theorem 3, star-nested special case |
| Anti-Horn | one two-literal Anti-Horn clause | Theorem 4, polarity-dual special case |
| Square2CNF | Form I `(a∨b)∧(c∨d)` | Theorem 6 |
| Affine | one canonical Boolean XOR equation | Theorem 5 |
| SmartCertified/CALS | exact nodes with one theory per path | direct Proposition-1 path certificate |

## 3. Frozen benchmark reaudit

The tracked frozen artifact `rust_results_final_freeze_r10_1fbfa30/` contains
46 datasets, 10 runs, two depths, eight methods, and 7,360 rows. Every retained
feature in every dataset is Boolean.

`smart-mdt-rs/tools/theorem_reaudit.py` combines each row's recorded domain,
path, backend, cache, fallback, and full-held-out AXp evidence with the
immutable benchmark commit's closed predicate/generator invariants. The
resulting [theorem_reaudit.csv](../theorem_reaudit.csv) contains 7,360 exact
certificates and zero violations.

**Reaudit result: OFFICIAL BENCHMARK EXACT-THEOREM VALID.** A full 71-hour
benchmark regeneration was not required: historical Horn and Anti-Horn nodes
are single clauses, Square2CNF nodes are Form I, affine nodes are one canonical
Boolean equation, and every actual tree/path passed the historical structural,
domain, solver, and AXp gates.

## 4. Controlled expressive-power experiment

The controlled study uses the complete six-variable Boolean domain repeated
eight times, deterministic noise of 0%, 5%, and 10%, five deterministic
train/test repetitions, depth 4, and all five fixed learners. Targets include
unary, simple and chained star-nested Horn, the exact chained Anti-Horn dual,
Square2CNF Forms I–III, and three-variable parity. All 600 result rows are
theorem-certified.

At 0% noise, the native family was always perfect and was usually the most
compact perfect model:

| Target | Native family | Accuracy | Mean nodes | Predicate literals | Mean AXp |
|---|---:|---:|---:|---:|---:|
| `x0` | Unary | 1.000 | 3 | 1 | 1.000 |
| `¬x0 ∨ x1` | Horn | 1.000 | 3 | 2 | 1.216 |
| Star-nested Horn chain | Horn | 1.000 | 7 | 6 | 1.710 |
| Exact Anti-Horn dual chain | Anti-Horn | 1.000 | 7 | 6 | 1.743 |
| Square2CNF Form I | Square2CNF | 1.000 | 3 | 4 | 2.000 |
| Square2CNF Form II | Square2CNF | 1.000 | 5 | 8 | 2.000 |
| Square2CNF Form III | Square2CNF | 1.000 | 3 | 4 | 2.000 |
| `x0 XOR x1 XOR x2` | Affine | 1.000 | 3 | 3 | 3.000 |

Perfect accuracy is not unique because a deeper tree from another family can
represent many of these functions. The important controlled distinction is
compactness. On parity, Square2CNF and Unary also reached 1.000 accuracy but
used 15 and 23 nodes, respectively, versus 3 affine nodes. On Square2CNF Forms
I–III, Square2CNF used 3, 5, and 3 nodes; other perfect learners often required
substantially more. Horn and Square2CNF both represented the Horn chain in 7
nodes, and Anti-Horn and Square2CNF both represented the dual chain in 7 nodes.
The simple implication belongs to several class intersections and therefore
does not distinguish Horn from Anti-Horn or Square2CNF.

Noise reduces accuracy and grows all trees; it does not invalidate the target
language. Native-family accuracies at 10% deterministic noise were 0.888
(Unary), 0.875/0.896 (simple/chained Horn), 0.929 (Anti-Horn chain),
0.883/0.881/0.888 (Square Forms I–III), and 0.906 (Affine).

The complete table is
`language_analysis/controlled_expressive_power.csv`; the expanded heatmap is
`controlled_expressive_power_heatmap.pdf`. Compactness evidence is exported in
`controlled_compactness_summary.csv` and `.tex`, with separate 0%-noise node,
predicate-literal, and AXp heatmaps. Each compactness figure marks smaller
values as better and states why accuracy alone cannot identify direct native
representation.

## 5. Fixed-family real-data results

Runs and depths were first averaged within each dataset. Dataset is the unit
of analysis.

| Family | Accuracy mean | Accuracy median | Mean nodes | Predicate literals | Mean AXp | Fit time (s) | Accuracy rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unary | 0.858646 | 0.901415 | 53.235 | 26.117 | 3.927 | 0.047 | 2.641 |
| Star-nested Horn | 0.859690 | 0.902891 | 51.217 | 50.217 | 5.137 | 0.065 | 2.663 |
| Star-nested Anti-Horn | 0.859848 | 0.895946 | 51.487 | 50.487 | 5.553 | 0.066 | 2.891 |
| Square2CNF | 0.859064 | 0.890299 | 50.807 | 99.613 | 6.357 | 0.444 | 2.761 |
| Boolean Affine | 0.853536 | 0.881165 | 53.126 | 60.042 | 7.897 | 0.162 | 4.043 |

With deterministic tie-breaking, Unary won 19 datasets, Horn 10,
Square2CNF 10, Anti-Horn 4, and Affine 3. Five datasets had an accuracy tie at
the best value. The mean winner-to-runner-up margin was only 0.00239, so winner
counts should not be read as large practical separations.

Per-dataset/depth results, winners, and ranks are in
`dataset_language_results.csv`, `language_winners.csv`, and
`language_ranks.csv`.

## 6. Dataset structural characterization

`dataset_structural_signatures.csv` records sample size, feature count,
sample-to-feature ratio, class prevalence/imbalance, and feature prevalence and
sparsity. Logical signals are recomputed independently on each of the ten
deterministic 70% training folds and then averaged. No held-out label is used.

Each logical signal is the best root information gain under the same
theorem-valid Boolean candidate semantics and construction beam used by the
learner: unary literal, two-literal star-nested Horn, its exact Anti-Horn dual,
Form-I Square2CNF, or a two/three-variable GF(2) equation. Interaction signals
are reported relative to unary gain.

These signals characterize the learner's available local split evidence. They
are not claims that one root statistic fully describes a dataset's deeper
logical structure.

## 7. Family specialization

**EXPLORATORY STRUCTURAL CHARACTERIZATION.** With 46 datasets, Spearman tests
use one value per dataset and bootstrap confidence intervals resample datasets.

| Family | Spearman ρ: signal vs family advantage | 95% bootstrap CI | Uncorrected p |
|---|---:|---:|---:|
| Unary | 0.001 | [-0.270, 0.282] | 0.994 |
| Horn | -0.085 | [-0.392, 0.211] | 0.576 |
| Anti-Horn | -0.137 | [-0.408, 0.154] | 0.363 |
| Square2CNF | 0.139 | [-0.169, 0.412] | 0.358 |
| Affine | 0.113 | [-0.143, 0.351] | 0.455 |

Every interval crosses zero. Therefore the real-data corpus does **not**
establish that these root signals predict relative family accuracy. The
controlled experiment establishes representational specialization; the
corpus-level structural-performance association remains unestablished.

## 8. Fixed-language oracle

For each dataset, the oracle selects the fixed family with the highest mean
accuracy. Regret is oracle accuracy minus adaptive-method accuracy.

| Adaptive method | Mean regret | Median regret | Beats all fixed | Exact matches | Underperforms |
|---|---:|---:|---:|---:|---:|
| SmartCertified | 0.004718 | 0.004423 | 7 | 0 | 39 |
| CALS | -0.000524 | 0.000257 | 21 | 2 | 23 |
| CALS CompactExplain | 0.004138 | 0.002967 | 14 | 0 | 32 |

CALS has slightly negative mean regret, so on average it marginally exceeded
the globally fixed-language oracle. This is possible because the oracle fixes
one language for an entire dataset, whereas CALS can select different local
families. The dataset counts also show that CALS is not uniformly superior:
it underperformed the oracle on 23 of 46 datasets.

## 9. CALS family usage

The frozen diagnostic contains exact family counts from the selected final
tree after pruning. Both adaptive policies used all five families:

| Method | Unary | Horn | Anti-Horn | Square2CNF | Affine |
|---|---:|---:|---:|---:|---:|
| CALS (nodes) | 999 | 1,586 | 586 | 1,871 | 354 |
| CALS (share) | 18.5% | 29.4% | 10.9% | 34.7% | 6.6% |
| CompactExplain (nodes) | 1,179 | 1,063 | 414 | 1,720 | 281 |
| CompactExplain (share) | 25.3% | 22.8% | 8.9% | 36.9% | 6.0% |

This directly establishes that CALS uses multiple certified languages rather
than behaving as one fixed-family learner.

The frozen CSV did not preserve per-node depth, root identity, arity, or gain.
`cals_language_usage.csv` therefore retains exact all-node counts and marks
those historical fields unavailable. The depth figure is explicitly grouped
by configured maximum tree depth, not reconstructed node depth. The Rust
benchmark now adds diagnostic-only per-final-node output with node depth,
family, arity/literals, training-node information gain recomputed after final
selection/pruning, pruning survival, and root flag for future runs; it does
not affect selection. `structural_dataset_group` and
`cals_language_usage_by_structural_group.csv` aggregate the exact frozen node
counts by the strongest positive training-fold-only interaction signal (or
`unary_dominant` when none is positive). Historical details are not invented
and would require retraining to recover.

## 10. Statistical tests

Descriptive results are separated from inference. Friedman tests across the
five fixed families use 46 dataset blocks:

| Metric | Friedman statistic | p |
|---|---:|---:|
| Accuracy | 26.659 | 2.33e-5 |
| Nodes | 15.471 | 0.00382 |
| Predicate literals | 162.338 | 4.61e-34 |
| Mean AXp length | 156.017 | 1.04e-32 |
| Fit time | 126.800 | 1.88e-26 |

Pairwise Wilcoxon signed-rank tests use paired dataset values and Holm
correction within each metric. For accuracy, Unary, Horn, Anti-Horn, and
Square2CNF did not differ significantly from one another after correction.
Affine was lower than Unary (Holm p=0.0270, paired rank-biserial 0.489), Horn
(p=0.000213, 0.684), Anti-Horn (p=0.000273, 0.673), and Square2CNF
(p=0.000342, 0.661), where positive effect means the first named method was
higher. Complete descriptive, Friedman, and corrected pairwise results are in
`language_statistics.csv`.

## 11. Limitations

- The real-data corpus has only 46 dataset units. Structural correlations are
  exploratory, not causal laws.
- The fixed learners search theorem-safe sublanguages rather than every
  relation in each maximal theorem class. In particular, learned Horn and
  Anti-Horn nodes are two-literal clauses and learned Square2CNF nodes are Form
  I, although controlled labels include richer exact targets.
- Multiple fixed languages can represent the same function with deeper trees;
  controlled accuracy alone cannot identify a unique native family.
- Hyperparameters were not manipulated to force native-family wins.
- Fit times are local-machine measurements.
- The oracle knows the best globally fixed family only after evaluation; it is
  a descriptive comparator, not a deployable selector.
- Historical CALS per-node depth/root data were not present in the frozen CSV.
  Only exact all-node family counts are analyzed for that artifact.
- Ordered finite-domain UI backends are not implemented or evaluated.

## 12. Thesis-ready conclusions and language-selection guide

| Data/structural condition | Observed best/competitive language | Empirical evidence | Theoretical reason | Limitation |
|---|---|---|---|---|
| Strong single-feature rule | Unary | `x0` reached 1.000 with 3 nodes and one predicate literal | A unary relation expresses the rule directly | Real-corpus signal-to-advantage association not established |
| Directional implication / nested negative chain | Star-nested Horn, with Square2CNF competitive | Horn represented the exact chain perfectly in 7 nodes; Square2CNF tied, while Anti-Horn/Unary were below 1.0 | Theorem 3 admits nested negative sets and one positive literal per clause | Simple binary implication lies in several class intersections |
| Dual-polarity nested chain | Star-nested Anti-Horn, with Square2CNF competitive | Anti-Horn represented the exact dual perfectly in 7 nodes; Square2CNF tied | Theorem 4 is the exact polarity dual | Corpus-level predictor not established |
| Coupled pairwise-clause structure | Square2CNF | Forms I–III were perfect with 3, 5, and 3 nodes; other perfect learners were generally larger | Theorem 6 directly represents the three square forms | Current learner emits Form-I nodes and may compose a deeper tree for Forms II/III |
| Boolean parity/XOR | Boolean Affine | XOR3 was perfect with 3 affine nodes versus 15 Square2CNF and 23 Unary nodes | One GF(2) equation represents parity directly | Applies only to Boolean domains; affine had the lowest real-data mean accuracy |
| Heterogeneous local structure | CALS | CALS used every family and had mean fixed-oracle regret -0.000524; it beat all fixed families on 21 datasets | Per-path exact CSP certificates permit local family selection | CALS underperformed the fixed oracle on 23 datasets and costs more time |

The defensible conclusion is not that one theorem family dominates. The
controlled experiment demonstrates compact representational specialization,
the real benchmark shows small accuracy differences among four non-affine
fixed families, and CALS demonstrably combines all families. Dataset-wide
selection rules based only on the tested root signals are **not established by
this corpus**.
