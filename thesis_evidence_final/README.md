# Final Thesis Research Evidence

This bundle is a content-addressed research freeze. Paths in
`manifest_sha256.txt` are relative to this directory and sorted
deterministically. The manifest does not hash itself because a file cannot
contain its own stable digest.

## Evidence classes

### OFFICIAL FROZEN BENCHMARK EVIDENCE

`frozen_benchmark/` is the authoritative 7,360-row, 46-dataset result grid
from commit `1fbfa30`, including result partitions, metadata, warnings,
diagnostics, and configuration. It was not regenerated for this freeze.

### CONTROLLED ADDITIONAL EXPERIMENT

`controlled/` and the controlled figures summarize the already-completed
600-row exact-target experiment. The compactness tables and figures are
post-processing of those existing rows; no controlled training was rerun.

### POST-HOC LANGUAGE ANALYSIS

`language_characterization/`, `oracle/`, and the corresponding figures use
dataset as the statistical unit after aggregating the frozen runs/depths.
Structural signals use training-fold labels only and remain exploratory.

### DIAGNOSTIC-ONLY CALS ANALYSIS

`cals/` contains exact final-tree family counts retained by the frozen run.
Unavailable historical root/per-node fields remain explicitly unavailable;
no diagnostics were invented. Diagnostic instrumentation does not affect
model selection.

`claims/THESIS_CLAIMS_EVIDENCE.md` is the normative wording register for
thesis claims.
