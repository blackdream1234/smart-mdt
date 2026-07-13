use smart_mdt_rs::{
    data::{is_boolean_column, predicate_scope_is_boolean, ColumnMajorMatrix, Dataset},
    eval::{
        accuracy, run_debug_candidates, run_full_benchmark, BenchmarkConfig, DebugCandidateConfig,
    },
    logic::{Backend, LanguageFamily, Literal, Predicate, ThresholdAtom, ThresholdOp},
    search::affine::{generate_affine_with_diagnostics, AffineConfig},
    tree::{learn, predict_all, tree_is_certified, LanguagePolicy, LearnerConfig, TreeNode},
};
use std::{fs, path::PathBuf};

fn bool_lit(feature: u32) -> Literal {
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

fn boolean_rows(features: usize) -> ColumnMajorMatrix {
    let rows: Vec<Vec<f64>> = (0..(1usize << features))
        .map(|m| (0..features).map(|j| ((m >> j) & 1) as f64).collect())
        .collect();
    ColumnMajorMatrix::from_rows(&rows).unwrap()
}

#[test]
fn affine_complement_flips_rhs_only() {
    let literals = vec![bool_lit(0), bool_lit(1), bool_lit(2)];
    let p = Predicate::Affine {
        literals: literals.clone(),
        rhs: false,
    };
    let comp = p.affine_complement().unwrap();
    match comp {
        Predicate::Affine {
            literals: cl,
            rhs: cr,
        } => {
            assert!(cr, "complement must flip rhs from false to true");
            assert_eq!(cl, literals, "complement must keep literals/coefficients");
        }
        other => panic!("expected affine complement, got {other:?}"),
    }
}

#[test]
fn affine_predicate_and_complement_partition_boolean_domain_exactly() {
    let x = boolean_rows(3);
    for rhs in [false, true] {
        let p = Predicate::Affine {
            literals: vec![bool_lit(0), bool_lit(1), bool_lit(2)],
            rhs,
        };
        let comp = p.affine_complement().unwrap();
        for i in 0..x.rows() {
            let xor = (0..3).fold(false, |acc, j| acc ^ (x.get(i, j) == 1.0));
            // Brute-force XOR semantics.
            assert_eq!(p.eval(&x, i), xor == rhs);
            // Predicate and complement partition the domain: exactly one holds.
            assert_ne!(p.eval(&x, i), comp.eval(&x, i));
        }
    }
}

#[test]
fn non_boolean_arity3_affine_is_rejected_from_theorem_certification() {
    let ds = Dataset::from_dl8_like("tests/fixtures/affine_mixed.dl8").unwrap();
    // Feature index 2 is non-Boolean in this fixture.
    assert!(is_boolean_column(&ds.features, 0));
    assert!(is_boolean_column(&ds.features, 1));
    assert!(!is_boolean_column(&ds.features, 2));

    let arity3 = Predicate::Affine {
        literals: vec![bool_lit(0), bool_lit(1), bool_lit(2)],
        rhs: true,
    };
    assert!(!predicate_scope_is_boolean(&ds.features, &arity3));
    // The Boolean-domain guard forbids certifying this candidate.
    let base_certified = arity3.certificate(true).theorem_certified;
    let guarded_certified = base_certified && predicate_scope_is_boolean(&ds.features, &arity3);
    assert!(!guarded_certified);

    // An all-Boolean-scope affine over features {0,1} is certifiable.
    let arity2 = Predicate::Affine {
        literals: vec![bool_lit(0), bool_lit(1)],
        rhs: true,
    };
    assert!(predicate_scope_is_boolean(&ds.features, &arity2));
    assert!(arity2.certificate(true).theorem_certified);
}

#[test]
fn debug_candidates_affine_reports_guard_rejection_on_mixed_domain() {
    let out = std::env::temp_dir().join(format!("smart-mdt-affine-mixed-{}", std::process::id()));
    let _ = fs::remove_dir_all(&out);
    let cfg = DebugCandidateConfig {
        data_dir: PathBuf::from("tests/fixtures"),
        dataset: "affine_mixed".into(),
        method: "affine".into(),
        depth: 0,
        node_path: "root".into(),
        top_k: 50,
        output: out.clone(),
        seed: 42,
        max_candidates_per_node: 128,
        beam_width: 32,
    };
    run_debug_candidates(&cfg).unwrap();
    let text = fs::read_to_string(out.join("debug_candidates.csv")).unwrap();
    let header: Vec<&str> = text.lines().next().unwrap().split(',').collect();
    for col in [
        "predicate_debug",
        "arity",
        "rhs",
        "boolean_scope",
        "theorem_certified",
        "score_profile",
        "information_gain",
        "gain_ratio",
        "balance_component",
        "literal_penalty",
        "family_penalty",
        "fragmentation_penalty",
        "estimated_subtree_penalty",
        "instability_penalty",
        "canonical_tie_break_key",
        "rejected",
        "rejected_reason",
    ] {
        assert!(header.contains(&col), "missing debug column {col}");
    }
    let idx = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let mut saw_certified_boolean = false;
    let mut saw_guard_rejection = false;
    for line in text.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        let boolean_scope = cols[idx("boolean_scope")] == "true";
        let certified = cols[idx("theorem_certified")] == "true";
        let reason = cols[idx("rejected_reason")];
        if boolean_scope && certified {
            saw_certified_boolean = true;
        }
        if !boolean_scope {
            // Non-Boolean scope must never be theorem-certified.
            assert!(!certified, "non-Boolean affine was certified: {line}");
            if reason.contains("non_boolean_scope") {
                saw_guard_rejection = true;
            }
        }
    }
    assert!(
        saw_certified_boolean,
        "expected a certified Boolean-scope affine candidate"
    );
    assert!(
        saw_guard_rejection,
        "expected a non_boolean_scope rejection"
    );
    let _ = fs::remove_dir_all(&out);
}

#[test]
fn affine_learner_is_non_constant_certified_and_beats_majority_on_xor() {
    let ds = Dataset::from_dl8_like("tests/fixtures/affine_xor8.dl8").unwrap();
    let (candidates, diag) = generate_affine_with_diagnostics(&ds, 1, 32, AffineConfig::default());
    assert!(diag.candidate_count_after_filtering > 0);
    assert!(diag.best_gain > 0.0);
    assert!(candidates
        .iter()
        .all(|c| c.left_count + c.right_count == ds.labels.len()));

    let cfg = LearnerConfig {
        max_depth: 3,
        min_samples_split: 2,
        min_samples_leaf: 1,
        max_candidates_per_node: 64,
        beam_width: 32,
        split_score: Default::default(),
        language_policy: LanguagePolicy::AffineOnly,
        theorem_mode: true,
        random_seed: 42,
    };
    let tree = learn(&ds, &cfg).unwrap();
    assert!(tree.nodes() > 1, "affine learner produced a constant tree");
    assert!(tree_is_certified(&tree));

    let preds = predict_all(&tree, &ds.features);
    let ones = ds.labels.iter().filter(|&&y| y == 1).count();
    let majority = ones.max(ds.labels.len() - ones) as f64 / ds.labels.len() as f64;
    let acc = accuracy(&ds.labels, &preds);
    assert!(
        acc > majority,
        "affine accuracy {acc} did not beat majority {majority}"
    );
    assert!(
        (acc - 1.0).abs() < 1e-9,
        "XOR should be perfectly separable, got {acc}"
    );

    match &tree {
        TreeNode::Internal { predicate, .. } => {
            assert_eq!(predicate.language(), LanguageFamily::Affine);
            assert_eq!(predicate.backend(), Backend::Gf2Gaussian);
        }
        other => panic!("expected internal affine root, got {other:?}"),
    }
}

#[test]
fn affine_learner_produces_certified_gf2_tree_on_xor_separable_fixture() {
    let ds = Dataset::from_dl8_like("tests/fixtures/xor_separable.dl8").unwrap();
    let cfg = LearnerConfig {
        max_depth: 3,
        min_samples_split: 2,
        min_samples_leaf: 1,
        max_candidates_per_node: 64,
        beam_width: 32,
        split_score: Default::default(),
        language_policy: LanguagePolicy::AffineOnly,
        theorem_mode: true,
        random_seed: 42,
    };
    let tree = learn(&ds, &cfg).unwrap();
    assert!(tree.nodes() > 1, "affine tree is constant on xor_separable");
    assert!(tree_is_certified(&tree));
    let preds = predict_all(&tree, &ds.features);
    assert!((accuracy(&ds.labels, &preds) - 1.0).abs() < 1e-9);
    match &tree {
        TreeNode::Internal { predicate, .. } => {
            assert_eq!(predicate.backend(), Backend::Gf2Gaussian);
        }
        other => panic!("expected internal affine root, got {other:?}"),
    }
}

#[test]
fn affine_appears_only_in_certified_table_with_gf2_backend() {
    let base = std::env::temp_dir().join(format!("smart-mdt-affine-bench-{}", std::process::id()));
    let data_dir = base.join("data");
    let out = base.join("out");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&data_dir).unwrap();
    fs::copy(
        "tests/fixtures/affine_xor8.dl8",
        data_dir.join("affine_xor8.dl8"),
    )
    .unwrap();

    let cfg = BenchmarkConfig {
        data_dir,
        depths: vec![5],
        runs: 1,
        methods: vec!["affine".into()],
        output: out.clone(),
        seed: 42,
        strict_data_checks: false,
    };
    run_full_benchmark(&cfg).unwrap();

    let method_col = |text: &str| -> Vec<String> {
        text.lines()
            .skip(1)
            .filter_map(|l| l.split(',').nth(3).map(str::to_string))
            .collect()
    };

    let full = fs::read_to_string(out.join("full_results.csv")).unwrap();
    assert!(
        method_col(&full).iter().any(|m| m == "affine"),
        "affine missing from full_results"
    );

    let certified = fs::read_to_string(out.join("theorem_certified_results.csv")).unwrap();
    assert!(
        method_col(&certified).iter().any(|m| m == "affine"),
        "affine missing from theorem_certified_results"
    );
    // Certified rows must carry the GF(2) backend.
    for line in certified.lines().skip(1) {
        if line.split(',').nth(3) == Some("affine") {
            assert!(
                line.contains("Gf2Gaussian"),
                "certified affine row is not GF(2)-backed: {line}"
            );
        }
    }

    let empirical = fs::read_to_string(out.join("empirical_results.csv")).unwrap();
    assert!(
        !method_col(&empirical).iter().any(|m| m == "affine"),
        "certified affine leaked into empirical_results"
    );

    let _ = fs::remove_dir_all(&base);
}
