use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    eval::{run_quick, theorem_table_filter, ResultRow},
    logic::{Backend, LanguageFamily},
    tree::{learn, predict_all, LanguagePolicy, LearnerConfig},
};
use std::{fs, path::PathBuf};

fn dataset(rows: &[Vec<f64>], labels: Vec<u32>) -> Dataset {
    Dataset::new(ColumnMajorMatrix::from_rows(rows).unwrap(), labels).unwrap()
}

#[test]
fn learner_handles_or_and_and_seed_determinism() {
    let rows = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let or_ds = dataset(&rows, vec![0, 1, 1, 1]);
    let and_ds = dataset(&rows, vec![0, 0, 0, 1]);
    let cfg = LearnerConfig {
        max_depth: 2,
        language_policy: LanguagePolicy::BestCertifiedPerNode,
        random_seed: 7,
        ..LearnerConfig::default()
    };
    let t1 = learn(&or_ds, &cfg).unwrap();
    let t2 = learn(&or_ds, &cfg).unwrap();
    assert_eq!(t1, t2);
    assert_eq!(predict_all(&t1, &or_ds.features), or_ds.labels);
    let tand = learn(&and_ds, &cfg).unwrap();
    assert_eq!(predict_all(&tand, &and_ds.features), and_ds.labels);
}

#[test]
fn theorem_mode_rejects_empirical_and_tuned_policies() {
    let ds = dataset(&[vec![0.0], vec![1.0]], vec![0, 1]);
    for policy in [
        LanguagePolicy::EmpiricalMixed,
        LanguagePolicy::TunedExperimental,
    ] {
        let cfg = LearnerConfig {
            language_policy: policy,
            theorem_mode: true,
            ..LearnerConfig::default()
        };
        assert!(learn(&ds, &cfg).is_err());
    }
}

#[test]
fn theorem_filter_requires_allowed_backend_and_method() {
    let base = ResultRow {
        dataset: "d".into(),
        run: 0,
        depth: 1,
        method: "unary".into(),
        accuracy: 1.0,
        train_time: 0.0,
        predict_time: 0.0,
        tree_nodes: 1,
        leaves: 1,
        max_depth_reached: 0,
        mean_axp_length: 0.0,
        axp_time: 0.0,
        theorem_certified: true,
        language_family: LanguageFamily::Unary,
        backend: Backend::StructuralHorn,
        git_sha: "abc".into(),
        config: "seed=1".into(),
        random_state: 1,
        n_runs: 1,
        train_test_split_protocol: "deterministic_hash_70_30_first_label".into(),
    };
    assert!(theorem_table_filter(&base));
    assert!(!theorem_table_filter(&ResultRow {
        backend: Backend::IntervalDfsFallback,
        ..base.clone()
    }));
    assert!(!theorem_table_filter(&ResultRow {
        language_family: LanguageFamily::EmpiricalMixed,
        backend: Backend::EmpiricalMixed,
        ..base.clone()
    }));
    assert!(!theorem_table_filter(&ResultRow {
        method: "tuned-experimental".into(),
        ..base.clone()
    }));
    assert!(!theorem_table_filter(&ResultRow {
        method: "best-certified".into(),
        ..base
    }));
}

#[test]
fn quick_benchmark_writes_required_deterministic_outputs() {
    let out: PathBuf =
        std::env::temp_dir().join(format!("smart-mdt-rs-audit-{}", std::process::id()));
    let _ = fs::remove_dir_all(&out);
    let first = run_quick(&out).unwrap();
    let full_first = fs::read_to_string(out.join("full_results.csv")).unwrap();
    let second = run_quick(&out).unwrap();
    let full_second = fs::read_to_string(out.join("full_results.csv")).unwrap();
    assert_eq!(first.len(), second.len());
    assert_eq!(full_first, full_second);
    for file in [
        "full_results.csv",
        "summary_by_method.csv",
        "theorem_certified_results.csv",
        "empirical_results.csv",
        "axp_metadata.csv",
        "README_RESULTS.md",
    ] {
        assert!(out.join(file).exists(), "missing {file}");
    }
    let theorem = fs::read_to_string(out.join("theorem_certified_results.csv")).unwrap();
    assert!(!theorem.contains("affine"));
    assert!(!theorem.contains("bestpn"));
    assert!(!theorem.contains("NaN"));
    assert!(theorem.contains("config"));
    assert!(theorem.contains("git_sha"));
    let _ = fs::remove_dir_all(&out);
}
