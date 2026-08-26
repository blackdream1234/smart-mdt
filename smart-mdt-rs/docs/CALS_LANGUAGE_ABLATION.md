# Controlled CALS language ablation

## Scientific intervention

The experiment varies only `AllowedLanguages`. `CalsConfig::thesis()` and
`CalsConfig::compact_explain()` retain their respective scoring, search,
branch-and-bound, caching, pruning, candidate budgets, node budgets, and AXp
settings. The unrestricted `cals` and `cals_compact_explain` methods use
`AllowedLanguages::all()` and retain their previous behavior.

The five certified members are Unary, Horn, Anti-Horn, Square2CNF, and Boolean
Affine/GF(2). A single-language variant passes a singleton mask into the same
optimizer profile. The mask filters family generators before candidates are
ranked or scored.

## Cache and theorem safety

The full allowed-language mask is included in `SearchStateKey`'s candidate
configuration key. Candidate-pool, best-subtree, node-statistics, and lookahead
caches use that key. Predicate-mask caching remains keyed only by predicate,
which is safe because a predicate's row mask does not depend on which other
families were admissible. Caches are per fit, so there is no cross-dataset or
cross-run state.

Every restricted method remains in theorem mode. Candidate construction uses
the existing exact predicate recognizers and Boolean guard for Affine. Search
still updates the path-theory state, the final tree is audited, and every held-
out row receives the existing sufficiency and subset-minimality AXp checks.

## Methods

Single-language CALS:

- `cals_unary`
- `cals_horn`
- `cals_antihorn`
- `cals_square2cnf`
- `cals_affine`

Single-language CompactExplain:

- `cals_compact_explain_unary`
- `cals_compact_explain_horn`
- `cals_compact_explain_antihorn`
- `cals_compact_explain_square2cnf`
- `cals_compact_explain_affine`

Mixed-language controls remain `cals` and `cals_compact_explain`. The result
CSV records `optimizer_profile`, `allowed_language_set`, `ablation_variant`,
`predicate_literals`, `root_language`, and final-tree family counts.

## Pilot

The pilot uses the exact repository dataset stems:
`tic-tac-toe`, `car_evaluation-bin`, `balance-scale-bin`, `anneal`, and
`HTRU_2-bin`; three deterministic runs; depths 5 and 7; the ten restricted
methods; both mixed-language controls; and the five language-specific learners
needed for contemporaneous comparisons.

Run from `smart-mdt-rs/`:

```bash
bash tools/run_cals_language_ablation_pilot.sh
```

The script refuses to overwrite an existing output directory. It writes only
to `experiment_artifacts/cals_language_ablation_pilot/`; no frozen thesis
artifact is modified.

## Statistical plan

`tools/cals_language_ablation_analysis.py` first averages runs and depths within
each dataset × method block. Datasets are the independent paired units. For
accuracy, nodes, predicate literals, AXp length, and fit time it computes paired
Wilcoxon signed-rank tests, 10,000 deterministic paired-bootstrap resamples,
95% CIs for the mean paired difference, Holm correction within each metric,
and paired Cohen's d. The pilot output is explicitly exploratory.

## Full experiment (prepared, not launched)

The following command produces exactly 46 × 10 × 2 × 10 = 9,200 new rows. Do
not run it until the pilot evidence has been reviewed:

```bash
cargo run --release -- benchmark \
  --data ../data \
  --runs 10 \
  --depths 5,7 \
  --methods cals_unary,cals_horn,cals_antihorn,cals_square2cnf,cals_affine,cals_compact_explain_unary,cals_compact_explain_horn,cals_compact_explain_antihorn,cals_compact_explain_square2cnf,cals_compact_explain_affine \
  --seed 42 \
  --strict-data-checks \
  --output experiment_artifacts/cals_language_ablation_full_r10_d5_d7
```

The new output directory must not already exist. Existing language-specific and
mixed-language rows may be combined later only after their non-language
configuration, split protocol, seed schedule, depths, theorem mode, and AXp
protocol are verified identical.
