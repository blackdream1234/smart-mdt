use smart_mdt_rs::{
    data::Dataset,
    eval::{accuracy, run_debug_candidates, DebugCandidateConfig},
    search::antihorn::generate_antihorn_with_diagnostics,
    search::horn::generate_horn_with_diagnostics,
    search::square2cnf::generate_square2cnf_with_diagnostics,
    tree::{learn, predict_all, tree_is_certified, LanguagePolicy, LearnerConfig},
};
use std::{fs, path::PathBuf, process::Command};

fn cfg(dataset: &str, method: &str) -> DebugCandidateConfig {
    DebugCandidateConfig {
        data_dir: PathBuf::from("tests/fixtures"),
        dataset: dataset.into(),
        method: method.into(),
        depth: 0,
        node_path: "root".into(),
        top_k: 20,
        output: std::env::temp_dir().join(format!(
            "smart-mdt-debug-{dataset}-{method}-{}",
            std::process::id()
        )),
        seed: 42,
        max_candidates_per_node: 128,
        beam_width: 32,
    }
}

#[test]
fn debug_candidates_command_writes_csvs() {
    let out = std::env::temp_dir().join(format!("smart-mdt-debug-cli-{}", std::process::id()));
    let _ = fs::remove_dir_all(&out);
    let exe = env!("CARGO_BIN_EXE_smart-mdt-rs");
    let status = Command::new(exe)
        .args([
            "debug-candidates",
            "--data",
            "tests/fixtures",
            "--dataset",
            "horn_separable",
            "--method",
            "horn",
            "--top-k",
            "5",
            "--output",
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    assert!(out.join("debug_candidates.csv").exists());
    assert!(out.join("debug_candidate_masks.csv").exists());
    let text = fs::read_to_string(out.join("debug_candidates.csv")).unwrap();
    assert!(text.contains("true_count"));
    let _ = fs::remove_dir_all(out);
}

#[test]
fn unary_root_candidates_include_valid_non_empty_splits() {
    let c = cfg("horn_separable", "unary");
    let _ = fs::remove_dir_all(&c.output);
    run_debug_candidates(&c).unwrap();
    let text = fs::read_to_string(c.output.join("debug_candidates.csv")).unwrap();
    assert!(text.lines().skip(1).any(|line| line.contains(",false,")));
}

#[test]
fn certified_toy_candidate_pools_are_not_all_rejected() {
    for (dataset, method) in [
        ("horn_separable", "horn"),
        ("antihorn_separable", "antihorn"),
        ("square2cnf_separable", "square2cnf"),
    ] {
        let c = cfg(dataset, method);
        let _ = fs::remove_dir_all(&c.output);
        let rows = run_debug_candidates(&c).unwrap();
        assert!(
            !rows.is_empty(),
            "{dataset}/{method} produced empty diagnostics"
        );
        let text = fs::read_to_string(c.output.join("debug_candidates.csv")).unwrap();
        assert!(
            text.lines().skip(1).any(|line| line.contains(",false,")),
            "all rejected for {dataset}/{method}"
        );
    }
}

#[test]
fn counts_sum_to_node_samples_and_children_not_both_empty() {
    let c = cfg("square2cnf_separable", "square2cnf");
    let _ = fs::remove_dir_all(&c.output);
    run_debug_candidates(&c).unwrap();
    let text = fs::read_to_string(c.output.join("debug_candidates.csv")).unwrap();
    let header: Vec<_> = text.lines().next().unwrap().split(',').collect();
    let idx = |name: &str| header.iter().position(|h| *h == name).unwrap();
    for line in text.lines().skip(1) {
        let cols: Vec<_> = line.split(',').collect();
        let n: usize = cols[idx("n_node_samples")].parse().unwrap();
        let t: usize = cols[idx("true_count")].parse().unwrap();
        let f: usize = cols[idx("false_count")].parse().unwrap();
        assert_eq!(t + f, n);
        assert!(t > 0 || f > 0);
    }
}

#[test]
fn horn_learner_uses_non_constant_positive_gain_split_on_toy_fixture() {
    let ds = Dataset::from_dl8_like("tests/fixtures/horn_separable.dl8").unwrap();
    let (_candidates, diag) = generate_horn_with_diagnostics(&ds, 1, 32);
    assert!(diag.candidate_count_after_filtering > 0);
    assert!(diag.best_gain > 0.0);

    let cfg = LearnerConfig {
        max_depth: 2,
        min_samples_split: 2,
        min_samples_leaf: 1,
        max_candidates_per_node: 64,
        beam_width: 32,
        split_score: Default::default(),
        branch_and_bound: Default::default(),
        cache: Default::default(),
        tree_search: Default::default(),
        conditional_search: Default::default(),
        parallel: Default::default(),
        pruning: Default::default(),
        adaptive_language: Default::default(),
        axp_rerank: Default::default(),
        language_policy: LanguagePolicy::HornOnly,
        theorem_mode: true,
        random_seed: 42,
    };
    let tree = learn(&ds, &cfg).unwrap();
    let preds = predict_all(&tree, &ds.features);
    let majority = ds
        .labels
        .iter()
        .filter(|&&y| y == 1)
        .count()
        .max(ds.labels.iter().filter(|&&y| y == 0).count()) as f64
        / ds.labels.len() as f64;
    assert!(tree.nodes() > 1, "Horn learner produced a constant tree");
    assert!(accuracy(&ds.labels, &preds) > majority);
    assert!(tree_is_certified(&tree));
}

#[test]
fn horn_debug_and_learner_candidate_generation_agree_on_tic_tac_toe_root() {
    let ds = Dataset::from_dl8_like("../data/tic-tac-toe.dl8").unwrap();
    let (candidates, diag) = generate_horn_with_diagnostics(&ds, 1, 32);
    assert!(diag.candidate_count_after_filtering > 0);
    assert!(candidates.iter().any(|c| c.score.predictive_gain > 0.0));

    let c = DebugCandidateConfig {
        data_dir: PathBuf::from("../data"),
        dataset: "tic-tac-toe".into(),
        method: "horn".into(),
        depth: 0,
        node_path: "root".into(),
        top_k: 20,
        output: std::env::temp_dir().join(format!(
            "smart-mdt-debug-tictactoe-horn-{}",
            std::process::id()
        )),
        seed: 42,
        max_candidates_per_node: 128,
        beam_width: 32,
    };
    let _ = fs::remove_dir_all(&c.output);
    let rows = run_debug_candidates(&c).unwrap();
    assert!(!rows.is_empty());
    let text = fs::read_to_string(c.output.join("debug_candidates.csv")).unwrap();
    assert!(text.lines().skip(1).any(|line| line.contains(",false,")));
    assert!(text.lines().skip(1).any(|line| {
        let cols: Vec<_> = line.split(',').collect();
        // impurity_gain is column 20 in the diagnostics schema.
        cols.get(19)
            .and_then(|s| s.parse::<f64>().ok())
            .is_some_and(|gain| gain > 0.0)
    }));
    let _ = fs::remove_dir_all(&c.output);
}

#[test]
fn antihorn_learner_uses_non_constant_positive_gain_split_on_toy_fixture() {
    let ds = Dataset::from_dl8_like("tests/fixtures/antihorn_separable.dl8").unwrap();
    let (candidates, diag) = generate_antihorn_with_diagnostics(&ds, 1, 32);
    assert!(diag.candidate_count_after_filtering > 0);
    assert!(diag.best_gain > 0.0);
    assert!(candidates
        .iter()
        .all(|c| c.left_count + c.right_count == ds.labels.len()));
    assert!(candidates
        .iter()
        .all(|c| c.left_count > 0 || c.right_count > 0));

    let cfg = LearnerConfig {
        max_depth: 2,
        min_samples_split: 2,
        min_samples_leaf: 1,
        max_candidates_per_node: 64,
        beam_width: 32,
        split_score: Default::default(),
        branch_and_bound: Default::default(),
        cache: Default::default(),
        tree_search: Default::default(),
        conditional_search: Default::default(),
        parallel: Default::default(),
        pruning: Default::default(),
        adaptive_language: Default::default(),
        axp_rerank: Default::default(),
        language_policy: LanguagePolicy::AntiHornOnly,
        theorem_mode: true,
        random_seed: 42,
    };
    let tree = learn(&ds, &cfg).unwrap();
    let preds = predict_all(&tree, &ds.features);
    let majority = ds
        .labels
        .iter()
        .filter(|&&y| y == 1)
        .count()
        .max(ds.labels.iter().filter(|&&y| y == 0).count()) as f64
        / ds.labels.len() as f64;
    assert!(
        tree.nodes() > 1,
        "AntiHorn learner produced a constant tree"
    );
    assert!(accuracy(&ds.labels, &preds) > majority);
    assert!(tree_is_certified(&tree));
}

#[test]
fn antihorn_debug_candidates_on_tic_tac_toe_are_positive_gain() {
    let ds = Dataset::from_dl8_like("../data/tic-tac-toe.dl8").unwrap();
    let (candidates, diag) = generate_antihorn_with_diagnostics(&ds, 1, 32);
    assert!(diag.candidate_count_after_filtering > 0);
    assert!(candidates.iter().any(|c| c.score.predictive_gain > 0.0));

    let c = DebugCandidateConfig {
        data_dir: PathBuf::from("../data"),
        dataset: "tic-tac-toe".into(),
        method: "antihorn".into(),
        depth: 0,
        node_path: "root".into(),
        top_k: 20,
        output: std::env::temp_dir().join(format!(
            "smart-mdt-debug-tictactoe-antihorn-{}",
            std::process::id()
        )),
        seed: 42,
        max_candidates_per_node: 128,
        beam_width: 32,
    };
    let _ = fs::remove_dir_all(&c.output);
    let rows = run_debug_candidates(&c).unwrap();
    assert!(!rows.is_empty());
    let text = fs::read_to_string(c.output.join("debug_candidates.csv")).unwrap();
    assert!(text.lines().skip(1).any(|line| line.contains(",false,")));
    assert!(text.lines().skip(1).any(|line| {
        let cols: Vec<_> = line.split(',').collect();
        cols.get(19)
            .and_then(|s| s.parse::<f64>().ok())
            .is_some_and(|gain| gain > 0.0)
    }));
    let _ = fs::remove_dir_all(&c.output);
}

#[test]
fn square2cnf_learner_uses_non_constant_positive_gain_split_on_toy_fixture() {
    let ds = Dataset::from_dl8_like("tests/fixtures/square2cnf_separable.dl8").unwrap();
    let (candidates, diag) = generate_square2cnf_with_diagnostics(&ds, 1, 32);
    assert!(diag.candidate_count_after_filtering > 0);
    assert!(diag.best_gain > 0.0);
    assert!(candidates
        .iter()
        .all(|c| c.left_count + c.right_count == ds.labels.len()));
    assert!(candidates
        .iter()
        .all(|c| c.left_count > 0 || c.right_count > 0));

    let cfg = LearnerConfig {
        max_depth: 2,
        min_samples_split: 2,
        min_samples_leaf: 1,
        max_candidates_per_node: 64,
        beam_width: 32,
        split_score: Default::default(),
        branch_and_bound: Default::default(),
        cache: Default::default(),
        tree_search: Default::default(),
        conditional_search: Default::default(),
        parallel: Default::default(),
        pruning: Default::default(),
        adaptive_language: Default::default(),
        axp_rerank: Default::default(),
        language_policy: LanguagePolicy::Square2CnfOnly,
        theorem_mode: true,
        random_seed: 42,
    };
    let tree = learn(&ds, &cfg).unwrap();
    let preds = predict_all(&tree, &ds.features);
    let majority = ds
        .labels
        .iter()
        .filter(|&&y| y == 1)
        .count()
        .max(ds.labels.iter().filter(|&&y| y == 0).count()) as f64
        / ds.labels.len() as f64;
    assert!(
        tree.nodes() > 1,
        "Square2CNF learner produced a constant tree"
    );
    assert!(accuracy(&ds.labels, &preds) > majority);
    assert!(tree_is_certified(&tree));
}

#[test]
fn square2cnf_debug_candidates_on_tic_tac_toe_are_positive_gain() {
    let ds = Dataset::from_dl8_like("../data/tic-tac-toe.dl8").unwrap();
    let (candidates, diag) = generate_square2cnf_with_diagnostics(&ds, 1, 32);
    assert!(diag.candidate_count_after_filtering > 0);
    assert!(candidates.iter().any(|c| c.score.predictive_gain > 0.0));

    let c = DebugCandidateConfig {
        data_dir: PathBuf::from("../data"),
        dataset: "tic-tac-toe".into(),
        method: "square2cnf".into(),
        depth: 0,
        node_path: "root".into(),
        top_k: 20,
        output: std::env::temp_dir().join(format!(
            "smart-mdt-debug-tictactoe-square2cnf-{}",
            std::process::id()
        )),
        seed: 42,
        max_candidates_per_node: 128,
        beam_width: 32,
    };
    let _ = fs::remove_dir_all(&c.output);
    let rows = run_debug_candidates(&c).unwrap();
    assert!(!rows.is_empty());
    let text = fs::read_to_string(c.output.join("debug_candidates.csv")).unwrap();
    assert!(text.lines().skip(1).any(|line| line.contains(",false,")));
    assert!(text.lines().skip(1).any(|line| {
        let cols: Vec<_> = line.split(',').collect();
        cols.get(19)
            .and_then(|s| s.parse::<f64>().ok())
            .is_some_and(|gain| gain > 0.0)
    }));
    let _ = fs::remove_dir_all(&c.output);
}
