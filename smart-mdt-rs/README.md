# smart-mdt-rs

`smart-mdt-rs` implements **Certificate-Guided Smart MDT (CGS-MDT)**, a Rust research prototype for certificate-first multivariate decision trees.

It separates theorem-certified families (Unary, Horn, AntiHorn, Square2CNF) from empirical extensions. It does not claim global optimality, full formal verification, or theorem certification for affine/tuned/mixed modes.

## Commands

```bash
cargo test
cargo run --release -- train --data ../data/car-un.dl8 --method horn --max-depth 5
cargo run --release -- benchmark --quick
```

## Full benchmark

```bash
cargo run --release -- benchmark --data ../data --depths 5,7 --runs 10 --methods unary,horn,antihorn,square2cnf --output ../rust_results
```
