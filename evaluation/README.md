# Smart-MDT research evaluation framework

This package turns a Smart-MDT benchmark output folder into a deterministic,
publication-ready evaluation. It validates the input, computes descriptive and
paired inferential statistics, generates PNG and vector PDF figures, writes
booktabs LaTeX tables, and renders Markdown, LaTeX, and executive reports
without manual editing.

The framework is evaluation-only. It does not import, configure, or modify the
Rust learning algorithms.

## Requirements

- Python 3.11 or newer
- pandas
- NumPy
- SciPy
- Matplotlib
- Jinja2
- pytest for the test suite

Install the package and runtime dependencies in an isolated environment:

```text
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

For development and tests, use `python -m pip install -e '.[test]'`. The
standalone `evaluation/requirements.txt` is also provided for environments that
manage the repository on `PYTHONPATH`. No notebook or seaborn dependency is
used.

## Usage

Run from the repository root:

```text
python -m evaluation.report \
  --input rust_results_all_methods_final
```

After editable installation, the command can also be run from `smart-mdt-rs`
with the benchmark folder addressed exactly as produced by the Rust CLI:

```text
python -m evaluation.report \
  --input ../rust_results_all_methods_final
```

The default output root is `evaluation/`. To keep generated material elsewhere:

```text
python -m evaluation.report \
  --input ../rust_results_all_methods_final \
  --output ../paper_evaluation \
  --bootstrap-resamples 10000 \
  --seed 20260718 \
  --confidence-level 0.95
```

The base seed is deterministically split by comparison and metric, so adding a
new comparison does not change existing bootstrap streams.

## Input validation

`full_results.csv` is required. The loader validates:

- required columns and canonical metric aliases;
- finite numeric values and valid ranges;
- missing values;
- duplicate `(dataset, run, depth, method)` rows;
- recognized method names.

The other benchmark CSVs and `README_RESULTS.md` are optional. Missing optional
files are recorded in the report. Figures and tables whose diagnostic input is
absent are still generated with an explicit “No data available” result.

Canonical evaluation metrics use these benchmark columns:

| Evaluation metric | Preferred benchmark column | Fallback |
| --- | --- | --- |
| Accuracy | `accuracy` | none |
| Tree nodes | `tree_nodes` | none |
| Predicate literals | `predicate_literals` | `literals_after_prune`, then `literals_before_prune` |
| Mean AXp length | `mean_axp_length` | none |
| Fit time | `fit_time_seconds` | `total_fit_time`, then `train_time` |

## Output structure

```text
evaluation/
├── figures/
│   ├── accuracy_boxplot.{png,pdf}
│   ├── accuracy_violin.{png,pdf}
│   ├── ...
│   └── pruning_diagnostics.{png,pdf}
├── tables/
│   ├── statistics.{csv,tex,md}
│   ├── significance.{csv,tex}
│   ├── dataset_summary.{csv,tex}
│   ├── certification_summary.tex
│   ├── warnings_summary.csv
│   ├── search_summary.tex
│   ├── pruning_summary.tex
│   └── axp_summary.tex
└── report/
    ├── evaluation_report.md
    ├── evaluation_report.tex
    ├── executive_summary.md
    ├── executive_summary.tex
    └── reproducibility_manifest.json
```

All LaTeX tables use booktabs, captions, labels, aligned numeric columns, and
three-decimal presentation. `evaluation_report.tex` is a standalone document;
compile it from `evaluation/report/` so its relative figure and table paths
resolve:

```text
pdflatex evaluation_report.tex
pdflatex evaluation_report.tex
```

## Statistical methods

- Descriptive statistics: mean, median, sample variance, sample standard
  deviation, range, and Student-t 95% confidence interval.
- Pairwise tests: two-sided paired Wilcoxon signed-rank tests aligned by
  dataset, run, and depth.
- Bootstrap: 10,000 paired resamples by default, reporting the observed and
  bootstrap mean difference plus percentile confidence interval.
- Effect sizes: standard Cliff’s delta and paired Cohen’s d with negligible,
  small, medium, and large magnitude labels.
- Dataset comparison: deterministic accuracy winner, wins, average ranks, and
  the Nemenyi critical difference at alpha 0.05.

Differences and effect sizes in `significance.csv` are oriented as right-hand
method minus left-hand method. Accuracy is maximized; all complexity,
explanation-length, and runtime metrics are minimized.

## Reproducibility

Matplotlib uses a fixed style and deterministic PDF metadata. Tables are
stably sorted. Bootstrap random generators use stable SHA-256-derived seeds.
The manifest records:

- the evaluation configuration;
- Python and dependency versions;
- SHA-256 hashes of every available benchmark input;
- SHA-256 hashes of every generated table, figure, and report.

Re-running with identical inputs, dependencies, and configuration produces
byte-identical tables, reports, PNGs, and PDFs.

## Tests

Run:

```text
python -m pytest evaluation/tests
```

The suite covers CSV validation, descriptive statistics, deterministic
bootstrap intervals, paired Wilcoxon tests, both effect sizes, dataset ranks,
certification summaries, warning/search/pruning/AXp analysis, booktabs LaTeX,
all figure formats, complete report generation, and byte-level reproducibility.
