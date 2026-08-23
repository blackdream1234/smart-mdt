# Exact-language characterization artifacts

Generate the complete study from the repository root with:

```bash
python3 smart-mdt-rs/tools/theorem_reaudit.py \
  --input rust_results_final_freeze_r10_1fbfa30 \
  --output theorem_reaudit.csv

python3 language_analysis/generate_controlled_fixtures.py \
  --output language_analysis/controlled_fixtures

cd smart-mdt-rs
cargo run --release -- benchmark \
  --data ../language_analysis/controlled_fixtures \
  --depths 4 --runs 5 \
  --methods unary,horn,antihorn,square2cnf,affine \
  --output ../language_analysis/controlled_raw --seed 314159
cd ..

python3 language_analysis/study.py \
  --benchmark rust_results_final_freeze_r10_1fbfa30 \
  --data data \
  --controlled language_analysis/controlled_raw/full_results.csv \
  --output language_analysis
```

The theorem reaudit is a mandatory gate. The study refuses non-Boolean,
non-certified, incomplete, or theorem-boundary-violating frozen inputs.

Statistical unit: dataset after averaging runs and depths. Structural signals
are derived independently on each deterministic 70% training fold and then
averaged; held-out labels are never used. Structural correlations are labeled
exploratory. Raw fit times are local-machine measurements.

The frozen CSV records exact final-tree family counts but not root identity,
per-node depth, per-node gain, or per-node arity. Accordingly,
`cals_language_usage.csv` preserves exact all-node counts and explicitly marks
the unavailable fields instead of inventing them. It also assigns the
training-fold-only structural dataset group, with the corresponding aggregate
in `cals_language_usage_by_structural_group.csv`. The Rust benchmark now
writes diagnostic-only `cals_language_usage.csv` records for each selected
final node (depth, family, arity/literals, recomputed training-node gain,
pruning survival, and root flag).
Reproducing historical per-node details would require retraining rather than
reinterpretation of the frozen artifact.
