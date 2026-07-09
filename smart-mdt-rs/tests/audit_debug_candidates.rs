use smart_mdt_rs::eval::{run_debug_candidates, DebugCandidateConfig};
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
