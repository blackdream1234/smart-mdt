use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    eval::{run_full_benchmark, theorem_table_filter, BenchmarkConfig},
    search::SplitScoreProfile,
    tree::{learn_with_diagnostics, tree_is_certified, CalsConfig, TreeSearchStrategy},
};
use std::{fs, path::PathBuf, process::Command};

fn dataset() -> Dataset {
    let rows: Vec<Vec<f64>> = (0..16)
        .map(|mask| (0..4).map(|bit| ((mask >> bit) & 1) as f64).collect())
        .collect();
    let labels = rows
        .iter()
        .map(|row| u32::from((row[0] == 1.0) ^ (row[1] == 1.0)))
        .collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

#[test]
fn thesis_profile_enables_the_complete_certified_pipeline() {
    let cals = CalsConfig::thesis();
    assert_eq!(cals.scoring.profile, SplitScoreProfile::SparseCertified);
    assert!(cals.branch_and_bound.enabled);
    assert!(cals.branch_and_bound.exhaustive_fallback);
    assert!(cals.cache.enabled);
    assert_eq!(
        cals.tree_search.strategy,
        TreeSearchStrategy::SparseLookahead
    );
    assert_eq!(cals.tree_search.lookahead_depth, 2);
    assert_eq!(cals.tree_search.candidate_beam_width, 16);
    assert_eq!(cals.tree_search.tree_beam_width, 8);
    assert!(cals.parallel.enabled);
    assert!(cals.pruning.enabled);
    assert!(cals.adaptive_language.enabled);
    assert!(!cals.axp_rerank.enabled);
}

#[test]
fn unified_cals_config_trains_a_deterministic_certified_tree() {
    let data = dataset();
    let config = CalsConfig::thesis().learner_config(3, 42);
    let (first, first_diagnostics) = learn_with_diagnostics(&data, &config).unwrap();
    let (second, _) = learn_with_diagnostics(&data, &config).unwrap();
    assert_eq!(first, second);
    assert!(tree_is_certified(&first));
    assert!(first.nodes() <= config.tree_search.node_budget);
    assert!(first_diagnostics.pruning.path_certified_after);
}

#[test]
fn cals_benchmark_method_is_theorem_admissible() {
    let base = std::env::temp_dir().join(format!("smart-mdt-cals-method-{}", std::process::id()));
    let data_dir = base.join("data");
    let output = base.join("out");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&data_dir).unwrap();
    fs::copy(
        PathBuf::from("tests/fixtures/affine_xor8.dl8"),
        data_dir.join("affine_xor8.dl8"),
    )
    .unwrap();
    let rows = run_full_benchmark(&BenchmarkConfig {
        data_dir,
        depths: vec![2],
        runs: 1,
        methods: vec!["cals".into()],
        output,
        seed: 42,
        strict_data_checks: false,
        cals: CalsConfig::thesis(),
    })
    .unwrap();
    assert_eq!(rows.len(), 1);
    assert!(rows[0].path_certified);
    assert!(theorem_table_filter(&rows[0]));
    let _ = fs::remove_dir_all(base);
}

#[test]
fn cli_accepts_cals_method_and_required_flags() {
    let binary = env!("CARGO_BIN_EXE_smart-mdt-rs");
    let output = Command::new(binary)
        .args([
            "train",
            "--data",
            "tests/fixtures/affine_xor8.dl8",
            "--method",
            "cals",
            "--max-depth",
            "2",
            "--cals-profile",
            "thesis",
            "--tree-search",
            "lookahead",
            "--tree-beam-width",
            "2",
            "--candidate-beam-width",
            "4",
            "--lookahead-depth",
            "1",
            "--node-budget",
            "7",
            "--score-profile",
            "sparse-certified",
            "--branch-and-bound",
            "--cache",
            "--cache-max-entries",
            "100",
            "--parallel",
            "--threads",
            "2",
            "--adaptive-language",
            "--prune",
            "--prune-validation-fraction",
            "0.25",
            "--prune-alpha-nodes",
            "0.0",
            "--prune-alpha-leaves",
            "0.0",
            "--prune-alpha-literals",
            "0.0",
            "--accuracy-epsilon",
            "0.01",
            "--axp-rerank",
            "--axp-shortlist",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Internal"));
}
