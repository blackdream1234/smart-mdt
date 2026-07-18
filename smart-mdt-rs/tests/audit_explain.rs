use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    explain::{extract_axp_deletion, weak_axp_check},
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    tree::{predict_row, TreeNode},
};

fn ge(feature: u32) -> Literal {
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

fn stump(predicate: Predicate, true_class: u32, false_class: u32) -> TreeNode {
    TreeNode::Internal {
        predicate,
        majority_class: false_class,
        left: Box::new(TreeNode::Leaf {
            class: true_class,
            samples: 1,
        }),
        right: Box::new(TreeNode::Leaf {
            class: false_class,
            samples: 1,
        }),
    }
}

fn binary_domain(n: usize) -> ColumnMajorMatrix {
    let rows: Vec<Vec<f64>> = (0..(1usize << n))
        .map(|m| (0..n).map(|j| ((m >> j) & 1) as f64).collect())
        .collect();
    ColumnMajorMatrix::from_rows(&rows).unwrap()
}

fn and_tree() -> TreeNode {
    TreeNode::Internal {
        predicate: Predicate::Unary(ge(0)),
        majority_class: 0,
        left: Box::new(TreeNode::Internal {
            predicate: Predicate::Unary(ge(1)),
            majority_class: 0,
            left: Box::new(TreeNode::Leaf {
                class: 1,
                samples: 1,
            }),
            right: Box::new(TreeNode::Leaf {
                class: 0,
                samples: 1,
            }),
        }),
        right: Box::new(TreeNode::Leaf {
            class: 0,
            samples: 2,
        }),
    }
}

fn brute_weak(tree: &TreeNode, instance: &[f64], selected: &[u32], target: u32) -> bool {
    let n = instance.len();
    for mask in 0..(1usize << n) {
        let row: Vec<f64> = (0..n).map(|j| ((mask >> j) & 1) as f64).collect();
        if selected
            .iter()
            .all(|&f| row[f as usize] == instance[f as usize])
        {
            let x = ColumnMajorMatrix::from_rows(&[row]).unwrap();
            if predict_row(tree, &x, 0) != target {
                return false;
            }
        }
    }
    true
}

#[test]
fn weak_axp_matches_bruteforce_completions_for_binary_tree() {
    let tree = and_tree();
    let domain = binary_domain(2);
    let instance = vec![1.0, 1.0];
    let target = 1;
    for selected in [vec![], vec![0], vec![1], vec![0, 1]] {
        let got = weak_axp_check(&tree, &domain, &instance, target, &selected, true);
        assert_eq!(
            got.is_weak_axp,
            brute_weak(&tree, &instance, &selected, target)
        );
        assert!(got.metadata.theorem_certified);
    }
}

#[test]
fn deletion_returns_subset_minimal_known_axp() {
    let tree = and_tree();
    let domain = binary_domain(2);
    let axp = extract_axp_deletion(&tree, &domain, 3, true);
    assert_eq!(axp.features, vec![0, 1]);
    assert!(weak_axp_check(&tree, &domain, &[1.0, 1.0], 1, &axp.features, true).is_weak_axp);
    for f in axp.features.clone() {
        let smaller: Vec<_> = axp.features.iter().copied().filter(|x| *x != f).collect();
        assert!(!weak_axp_check(&tree, &domain, &[1.0, 1.0], 1, &smaller, true).is_weak_axp);
    }
}

#[test]
fn theorem_check_rejects_a_non_boolean_domain_even_for_a_boolean_query_row() {
    let domain = ColumnMajorMatrix::from_rows(&[vec![0.0], vec![2.0]]).unwrap();
    let predicate = Predicate::Unary(Literal {
        atom: ThresholdAtom {
            feature: 0,
            threshold_id: 0,
            threshold: 1.5,
            op: ThresholdOp::GreaterEqual,
        },
        positive: true,
    });
    let tree = stump(predicate, 1, 0);
    let checked = weak_axp_check(&tree, &domain, &[0.0], 0, &[], true);
    assert!(!checked.is_weak_axp);
    assert!(!checked.metadata.theorem_certified);
    assert!(checked
        .metadata
        .rejected_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("Boolean")));

    // Dataset-row semantics outside theorem mode must still see the value 2.
    assert!(!weak_axp_check(&tree, &domain, &[0.0], 0, &[], false).is_weak_axp);
}

#[test]
fn certified_path_blocking_handles_more_than_twenty_boolean_features() {
    let domain = ColumnMajorMatrix::from_rows(&[vec![0.0; 21]]).unwrap();
    let tree = stump(Predicate::Unary(ge(0)), 1, 0);
    let empty = weak_axp_check(&tree, &domain, &[0.0; 21], 0, &[], true);
    assert!(!empty.is_weak_axp);
    assert!(empty.metadata.theorem_certified);

    let axp = extract_axp_deletion(&tree, &domain, 0, true);
    assert_eq!(axp.features, vec![0]);
    assert!(axp.metadata.theorem_certified);
}

#[test]
fn empirical_completion_fallback_never_claims_theorem_certification() {
    // With more than twenty features, non-theorem mode checks only rows present
    // in the supplied domain. The unseen completion x0=1 reaches class 1, so
    // the empty feature set is not a formal weak AXp even though this one-row
    // empirical domain cannot witness the counterexample.
    let domain = ColumnMajorMatrix::from_rows(&[vec![0.0; 21]]).unwrap();
    let tree = stump(Predicate::Unary(ge(0)), 1, 0);
    let checked = weak_axp_check(&tree, &domain, &[0.0; 21], 0, &[], false);
    assert!(checked.is_weak_axp);
    assert!(!checked.metadata.theorem_mode);
    assert!(!checked.metadata.theorem_certified);
}

#[test]
fn certified_path_solvers_match_exhaustive_boolean_completion_randomly() {
    let domain = binary_domain(4);
    let mut seed = 0x51a7_2d3c_94e8_b601u64;
    for case in 0..4_000 {
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let family = case % 4;
        let first = (next() % 4) as u32;
        let mut second = (next() % 3) as u32;
        if second >= first {
            second += 1;
        }
        let root = match family {
            0 => Predicate::HornClause(vec![ge(first).negated(), ge(second)]),
            1 => Predicate::AntiHornClause(vec![ge(first), ge(second).negated()]),
            2 => Predicate::Square2Cnf {
                a: ge(first),
                b: ge(second).negated(),
                c: ge((first + 1) % 4).negated(),
                d: ge((second + 1) % 4),
            },
            _ => {
                let mut features = vec![first, second];
                features.sort_unstable();
                Predicate::Affine {
                    literals: features.into_iter().map(ge).collect(),
                    rhs: next() & 1 == 1,
                }
            }
        };
        let child_feature = (next() % 4) as u32;
        let child = stump(
            Predicate::Unary(if next() & 1 == 1 {
                ge(child_feature)
            } else {
                ge(child_feature).negated()
            }),
            (next() & 1) as u32,
            (next() & 1) as u32,
        );
        let tree = TreeNode::Internal {
            predicate: root,
            majority_class: 0,
            left: Box::new(child),
            right: Box::new(TreeNode::Leaf {
                class: (next() & 1) as u32,
                samples: 1,
            }),
        };
        let row = (next() % 16) as usize;
        let instance = (0..4)
            .map(|feature| domain.get(row, feature))
            .collect::<Vec<_>>();
        let target = predict_row(&tree, &domain, row);
        let selected = (0..4)
            .filter(|feature| (next() >> feature) & 1 == 1)
            .map(|feature| feature as u32)
            .collect::<Vec<_>>();
        let checked = weak_axp_check(&tree, &domain, &instance, target, &selected, true);
        assert!(
            checked.metadata.theorem_certified,
            "case {case} was unexpectedly rejected: {:?}",
            checked.metadata.rejected_reason
        );
        assert_eq!(
            checked.is_weak_axp,
            brute_weak(&tree, &instance, &selected, target),
            "case {case}, family {family}, selected {selected:?}"
        );
    }
}

#[test]
fn non_boolean_affine_reference_cannot_produce_a_verified_explanation() {
    use smart_mdt_rs::explain::{compile_verified_explanation, ExplanationAudience};

    let reference = Dataset::new(
        ColumnMajorMatrix::from_rows(&[vec![0.0], vec![2.0]]).unwrap(),
        vec![0, 1],
    )
    .unwrap();
    let tree = stump(
        Predicate::Affine {
            literals: vec![ge(0)],
            rhs: true,
        },
        1,
        0,
    );
    let error = compile_verified_explanation(&tree, &reference, 0, ExplanationAudience::Technical)
        .unwrap_err();
    assert!(format!("{error}").contains("TheoremRejected"));
}

#[test]
fn out_of_bounds_tree_scope_is_rejected_without_prediction_panic() {
    use smart_mdt_rs::explain::{compile_verified_explanation, ExplanationAudience};

    let reference = Dataset::new(
        ColumnMajorMatrix::from_rows(&[vec![0.0], vec![1.0]]).unwrap(),
        vec![0, 1],
    )
    .unwrap();
    let tree = stump(Predicate::Unary(ge(7)), 1, 0);
    let error = compile_verified_explanation(&tree, &reference, 0, ExplanationAudience::Technical)
        .unwrap_err();
    assert!(format!("{error}").contains("out-of-bounds"));
}

#[test]
fn weak_axp_rejects_malformed_empirical_inputs_without_panicking() {
    let domain = ColumnMajorMatrix::from_rows(&[vec![0.0], vec![1.0]]).unwrap();
    let tree = stump(Predicate::Unary(ge(0)), 1, 0);

    for checked in [
        weak_axp_check(&tree, &domain, &[0.0], 0, &[7], false),
        weak_axp_check(&tree, &domain, &[], 0, &[], false),
        weak_axp_check(
            &stump(Predicate::Unary(ge(7)), 1, 0),
            &domain,
            &[0.0],
            0,
            &[],
            false,
        ),
    ] {
        assert!(!checked.is_weak_axp);
        assert!(!checked.metadata.theorem_certified);
        assert!(checked.metadata.rejected_reason.is_some());
    }
}

#[test]
fn axp_deletion_rejects_bad_row_and_tree_scope_without_panicking() {
    let domain = ColumnMajorMatrix::from_rows(&[vec![0.0], vec![1.0]]).unwrap();
    let tree = stump(Predicate::Unary(ge(0)), 1, 0);

    let bad_row = extract_axp_deletion(&tree, &domain, domain.rows(), false);
    assert!(bad_row.features.is_empty());
    assert!(bad_row
        .metadata
        .rejected_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("row is out of bounds")));

    let bad_scope = extract_axp_deletion(&stump(Predicate::Unary(ge(7)), 1, 0), &domain, 0, true);
    assert!(bad_scope.features.is_empty());
    assert!(bad_scope
        .metadata
        .rejected_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("tree scope is out of bounds")));
}
