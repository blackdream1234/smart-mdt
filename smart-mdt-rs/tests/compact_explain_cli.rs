use smart_mdt_rs::{
    eval::{run_full_benchmark, theorem_table_filter, BenchmarkConfig},
    explain::render_human_explanation,
    tree::{CalsConfig, TreeSearchStrategy},
};
use std::{fs, path::PathBuf, process::Command};

#[test]
fn compact_profile_is_serial_selective_and_class_aware() {
    let compact = CalsConfig::compact_explain();
    assert_eq!(
        compact.tree_search.strategy,
        TreeSearchStrategy::SelectiveLookahead
    );
    assert!(compact.tree_search.selective.enabled);
    assert_eq!(compact.tree_search.selective.maximum_depth, 2);
    assert_eq!(compact.tree_search.selective.candidate_beam_width, 8);
    assert_eq!(compact.tree_search.selective.tree_beam_width, 4);
    assert!(!compact.parallel.enabled);
    assert!(compact.pruning.class_aware.enabled);
    assert!(compact.cache.predicate_masks);
    assert!(compact.cache.candidate_pools);
    assert!(!compact.cache.best_subtrees);
    assert!(compact.conditional_search.enabled);
    assert!(!compact.axp_rerank.enabled);
}

#[test]
fn compact_benchmark_row_is_theorem_admissible_and_reports_class_metrics() {
    let base =
        std::env::temp_dir().join(format!("smart-mdt-compact-method-{}", std::process::id()));
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
        methods: vec!["cals_compact_explain".into()],
        output: output.clone(),
        seed: 42,
        strict_data_checks: false,
        cals: CalsConfig::thesis(),
        compact_explain: CalsConfig::compact_explain(),
    })
    .unwrap();
    assert_eq!(rows.len(), 1);
    assert!(theorem_table_filter(&rows[0]));
    assert!(rows[0].path_certified);
    assert_eq!(rows[0].path_violation_count, 0);
    assert_eq!(rows[0].parallel_threads, 0);
    assert_eq!(rows[0].provisional_axp_evaluations, 0);
    assert_eq!(rows[0].axp_extraction_stage, "post_selection_final_tree");
    let header = fs::read_to_string(output.join("full_results.csv")).unwrap();
    let header = header.lines().next().unwrap();
    for column in [
        "validation_balanced_accuracy_before_prune",
        "validation_balanced_accuracy_after_prune",
        "validation_minority_recall_before_prune",
        "validation_minority_recall_after_prune",
        "validation_class_support",
        "pruning_root_reason",
        "nodes_using_selective_lookahead",
        "branch_and_bound_activation_count",
        "cache_activation_count",
    ] {
        assert!(header.split(',').any(|value| value == column));
    }
    let _ = fs::remove_dir_all(base);
}

#[test]
fn explain_cli_writes_verified_json_and_text_derived_from_it() {
    let output =
        std::env::temp_dir().join(format!("smart-mdt-compact-explain-{}", std::process::id()));
    let _ = fs::remove_dir_all(&output);
    let status = Command::new(env!("CARGO_BIN_EXE_smart-mdt-rs"))
        .args([
            "explain",
            "--data",
            "tests/fixtures/affine_xor8.dl8",
            "--method",
            "cals_compact_explain",
            "--max-depth",
            "2",
            "--row",
            "0",
            "--audience",
            "clinical",
            "--output",
        ])
        .arg(&output)
        .status()
        .unwrap();
    assert!(status.success());
    let json = fs::read_to_string(output.join("verified_explanation.json")).unwrap();
    let text = fs::read_to_string(output.join("human_explanation.txt")).unwrap();
    assert_eq!(text, render_human_explanation(&json).unwrap());
    assert!(text.contains("model prediction"));
    assert!(text.contains("not a diagnosis"));
    let _ = fs::remove_dir_all(output);
}
