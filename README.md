# Smart-MDT

This repository contains the certified Smart-MDT Rust implementation and its
reproducible research tooling.

- Rust learner and benchmark documentation: [`smart-mdt-rs/`](smart-mdt-rs/)
- Automated thesis and publication evaluation:
  [`evaluation/README.md`](evaluation/README.md)

The current freeze decision and authoritative artifact paths are recorded in
[`smart-mdt-rs/docs/FINAL_FREEZE_REPORT.md`](smart-mdt-rs/docs/FINAL_FREEZE_REPORT.md).
Historical folders such as `rust_results_all_methods_final` are not current
publication evidence.

Generate the complete evaluation from a freshly audited benchmark folder:

```text
python -m evaluation.report --input <benchmark-output-directory>
```
