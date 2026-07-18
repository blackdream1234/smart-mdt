# Smart-MDT

This repository contains the certified Smart-MDT Rust implementation and its
reproducible research tooling.

- Rust learner and benchmark documentation: [`smart-mdt-rs/`](smart-mdt-rs/)
- Automated thesis and publication evaluation:
  [`evaluation/README.md`](evaluation/README.md)

Generate the complete evaluation from an existing benchmark folder:

```text
python -m evaluation.report --input rust_results_all_methods_final
```
