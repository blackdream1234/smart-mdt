//! Timing harness for full-tree AXp extraction, used as freeze evidence for the
//! `AxpCheckContext` caching fix (commit d75391d). It trains a unary theorem-mode
//! tree on a `.dl8` dataset and times `extract_final_tree_axps` over the first N
//! rows, reporting wall time and the certification result. Running it before and
//! after the fix (e.g. by stashing `src/explain/*.rs`) reproduces the reported
//! speedup with an unchanged certification outcome.
//!
//! Usage: cargo run --release --example time_axp_extraction -- <dataset.dl8> [rows] [depth]
use smart_mdt_rs::{
    data::Dataset,
    explain::extract_final_tree_axps,
    tree::{learn, LanguagePolicy, LearnerConfig},
};
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "../data/letter.dl8".into());
    let rows: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let depth: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let ds = Dataset::from_dl8_like(&path).expect("load dataset");
    let cfg = LearnerConfig {
        max_depth: depth,
        language_policy: LanguagePolicy::UnaryOnly,
        ..LearnerConfig::default()
    };
    let tree = learn(&ds, &cfg).expect("train tree");

    let start = Instant::now();
    let summary = extract_final_tree_axps(&tree, &ds.features, rows, true);
    let elapsed = start.elapsed().as_secs_f64();

    println!(
        "dataset={} rows={} cols={} depth={} tree_nodes={}",
        path,
        ds.features.rows(),
        ds.features.cols(),
        depth,
        tree.nodes()
    );
    println!(
        "axp_rows={} wall={:.4}s per_row={:.5}s certified={} valid={} minimal={}",
        rows,
        elapsed,
        elapsed / rows as f64,
        summary.theorem_certified,
        summary.valid_count,
        summary.minimal_count,
    );
}
