use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    eval::{run_full_benchmark, theorem_table_filter, BenchmarkConfig, ResultRow},
    logic::{
        candidate_is_compatible, next_theory_state, Backend, LanguageFamily, Literal,
        PathTheoryState, Predicate, ThresholdAtom, ThresholdOp,
    },
    tree::{
        learn, tree_is_certified, tree_path_theory_states, LanguagePolicy, LearnerConfig, TreeNode,
    },
};
use std::{fs, path::PathBuf};

fn lit(feature: u32) -> Literal {
    Literal {
        atom: ThresholdAtom {
            feature,
            threshold_id: 0,
            threshold: 0.5,
            op: ThresholdOp::GreaterEqual,
        },
        positive: true,
    }
}

fn leaf(class: u32) -> TreeNode {
    TreeNode::Leaf { class, samples: 1 }
}

fn node(predicate: Predicate, left: TreeNode, right: TreeNode) -> TreeNode {
    TreeNode::Internal {
        predicate,
        left: Box::new(left),
        right: Box::new(right),
        majority_class: 0,
    }
}

fn unary(feature: u32) -> Predicate {
    Predicate::Unary(lit(feature))
}

fn horn(feature: u32) -> Predicate {
    Predicate::HornClause(vec![lit(feature), lit(feature + 1).negated()])
}

fn antihorn(feature: u32) -> Predicate {
    Predicate::AntiHornClause(vec![lit(feature), lit(feature + 1)])
}

fn square2cnf(feature: u32) -> Predicate {
    Predicate::Square2Cnf {
        a: lit(feature),
        b: lit(feature + 1),
        c: lit(feature + 2),
        d: lit(feature + 3),
    }
}

fn affine(feature: u32) -> Predicate {
    Predicate::Affine {
        literals: vec![lit(feature), lit(feature + 1)],
        rhs: true,
    }
}

#[test]
fn unary_then_horn_is_valid() {
    let tree = node(unary(0), node(horn(1), leaf(0), leaf(1)), leaf(0));
    assert!(tree_is_certified(&tree));
    assert_eq!(
        next_theory_state(PathTheoryState::Uncommitted, &unary(0)).unwrap(),
        PathTheoryState::Uncommitted
    );
}

#[test]
fn unary_then_affine_is_valid() {
    let tree = node(unary(0), node(affine(1), leaf(0), leaf(1)), leaf(0));
    assert!(tree_is_certified(&tree));
}

#[test]
fn horn_then_horn_is_valid() {
    let tree = node(horn(0), node(horn(2), leaf(0), leaf(1)), leaf(0));
    assert!(tree_is_certified(&tree));
    assert!(candidate_is_compatible(PathTheoryState::Horn, &horn(2)));
}

#[test]
fn horn_then_antihorn_is_rejected() {
    let tree = node(horn(0), node(antihorn(2), leaf(0), leaf(1)), leaf(0));
    assert!(!tree_is_certified(&tree));
    assert!(!candidate_is_compatible(
        PathTheoryState::Horn,
        &antihorn(2)
    ));
}

#[test]
fn horn_then_affine_is_rejected() {
    let tree = node(horn(0), node(affine(2), leaf(0), leaf(1)), leaf(0));
    assert!(!tree_is_certified(&tree));
    assert!(next_theory_state(PathTheoryState::Horn, &affine(2)).is_err());
}

#[test]
fn two_sat_then_square2cnf_is_valid() {
    let tree = node(
        square2cnf(0),
        node(square2cnf(4), leaf(0), leaf(1)),
        leaf(0),
    );
    assert!(tree_is_certified(&tree));
    assert_eq!(
        tree_path_theory_states(&tree).unwrap(),
        vec![PathTheoryState::TwoSat]
    );
}

#[test]
fn affine_then_boolean_affine_is_valid() {
    let tree = node(affine(0), node(affine(2), leaf(0), leaf(1)), leaf(0));
    assert!(tree_is_certified(&tree));
    assert_eq!(
        tree_path_theory_states(&tree).unwrap(),
        vec![PathTheoryState::AffineGf2]
    );
}

#[test]
fn unary_branches_may_commit_independently() {
    let tree = node(
        unary(0),
        node(horn(1), leaf(0), leaf(1)),
        node(affine(3), leaf(0), leaf(1)),
    );
    assert!(tree_is_certified(&tree));
    assert_eq!(
        tree_path_theory_states(&tree).unwrap(),
        vec![PathTheoryState::Horn, PathTheoryState::AffineGf2]
    );
}

#[test]
fn every_smart_certified_learned_path_is_compatible() {
    let rows: Vec<Vec<f64>> = (0..16)
        .map(|mask| (0..4).map(|bit| ((mask >> bit) & 1) as f64).collect())
        .collect();
    let labels = rows
        .iter()
        .map(|row| u32::from((row[0] == 1.0) ^ (row[1] == 1.0)))
        .collect();
    let ds = Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap();
    let cfg = LearnerConfig {
        max_depth: 5,
        beam_width: 16,
        max_candidates_per_node: 128,
        language_policy: LanguagePolicy::SmartCertified,
        theorem_mode: true,
        ..LearnerConfig::default()
    };
    let tree = learn(&ds, &cfg).unwrap();
    assert!(tree.nodes() > 1);
    assert!(tree_is_certified(&tree));
    assert!(tree_path_theory_states(&tree).is_ok());
}

fn smart_row(path_certified: bool) -> ResultRow {
    ResultRow {
        dataset: "d".into(),
        run: 0,
        depth: 2,
        method: "smart_certified".into(),
        accuracy: 1.0,
        train_time: 0.0,
        predict_time: 0.0,
        tree_nodes: 3,
        leaves: 2,
        max_depth_reached: 1,
        mean_axp_length: 1.0,
        axp_time: 0.0,
        theorem_certified: path_certified,
        language_family: LanguageFamily::SmartCertified,
        backend: Backend::PathCertified,
        path_theory_state: if path_certified {
            "horn|affine_gf2".into()
        } else {
            "incompatible".into()
        },
        path_backend: if path_certified {
            "StructuralHorn|Gf2Gaussian".into()
        } else {
            "Unsupported".into()
        },
        path_certified,
        git_sha: "abc".into(),
        config: "{}".into(),
        random_state: 42,
        n_runs: 1,
        train_test_split_protocol: "deterministic_hash_70_30_first_label".into(),
    }
}

#[test]
fn theorem_reporting_admits_smart_certified_only_after_path_validation() {
    assert!(theorem_table_filter(&smart_row(true)));
    assert!(!theorem_table_filter(&smart_row(false)));
    assert!(!theorem_table_filter(&ResultRow {
        path_backend: "Affine".into(),
        ..smart_row(true)
    }));
}

#[test]
fn smart_certified_benchmark_emits_path_metadata_and_enters_theorem_table() {
    let base = std::env::temp_dir().join(format!(
        "smart-mdt-path-theory-benchmark-{}",
        std::process::id()
    ));
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
        depths: vec![3],
        runs: 1,
        methods: vec!["smart_certified".into()],
        output: output.clone(),
        seed: 42,
        strict_data_checks: false,
    })
    .unwrap();
    assert_eq!(rows.len(), 1);
    assert!(rows[0].path_certified);
    assert!(!rows[0].path_theory_state.is_empty());
    assert!(!rows[0].path_backend.is_empty());

    let full = fs::read_to_string(output.join("full_results.csv")).unwrap();
    let header = full.lines().next().unwrap();
    for column in ["path_theory_state", "path_backend", "path_certified"] {
        assert!(header.split(',').any(|name| name == column));
    }
    let theorem = fs::read_to_string(output.join("theorem_certified_results.csv")).unwrap();
    assert!(theorem.lines().skip(1).any(|line| {
        line.split(',').nth(3) == Some("smart_certified")
            && line.contains("PathCertified")
            && !line.contains("Empirical")
    }));
    let empirical = fs::read_to_string(output.join("empirical_results.csv")).unwrap();
    assert!(!empirical
        .lines()
        .skip(1)
        .any(|line| line.split(',').nth(3) == Some("smart_certified")));

    let _ = fs::remove_dir_all(&base);
}
