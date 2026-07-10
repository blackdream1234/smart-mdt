use smart_mdt_rs::{
    data::ColumnMajorMatrix,
    explain::weak_axp_check,
    logic::{Backend, LanguageFamily, Literal, Predicate, ThresholdAtom, ThresholdOp},
    tree::{predict_row, TreeNode},
};

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

fn affine(literals: Vec<Literal>, parity: bool) -> Predicate {
    Predicate::EmpiricalAffine { literals, parity }
}

fn xor_tree() -> TreeNode {
    TreeNode::Internal {
        predicate: affine(vec![lit(0), lit(1)], true),
        left: Box::new(TreeNode::Leaf {
            class: 1,
            samples: 2,
        }),
        right: Box::new(TreeNode::Leaf {
            class: 0,
            samples: 2,
        }),
        majority_class: 0,
    }
}

fn binary_domain(cols: usize) -> ColumnMajorMatrix {
    let rows: Vec<Vec<f64>> = (0..(1usize << cols))
        .map(|m| (0..cols).map(|j| ((m >> j) & 1) as f64).collect())
        .collect();
    ColumnMajorMatrix::from_rows(&rows).unwrap()
}

fn brute_weak(tree: &TreeNode, instance: &[f64], target: u32, selected: &[u32]) -> bool {
    let n = instance.len();
    (0..(1usize << n)).all(|mask| {
        let row: Vec<f64> = (0..n).map(|j| ((mask >> j) & 1) as f64).collect();
        if selected
            .iter()
            .all(|&f| row[f as usize] == instance[f as usize])
        {
            let x = ColumnMajorMatrix::from_rows(&[row]).unwrap();
            predict_row(tree, &x, 0) == target
        } else {
            true
        }
    })
}

#[test]
fn gf2_weak_axp_matches_bruteforce_for_small_affine_tree() {
    let tree = xor_tree();
    let domain = binary_domain(2);
    let instance = vec![1.0, 0.0];
    let target = predict_row(
        &tree,
        &ColumnMajorMatrix::from_rows(std::slice::from_ref(&instance)).unwrap(),
        0,
    );
    for selected in [vec![], vec![0], vec![1], vec![0, 1]] {
        let got = weak_axp_check(&tree, &domain, &instance, target, &selected, false);
        assert_eq!(
            got.is_weak_axp,
            brute_weak(&tree, &instance, target, &selected)
        );
        assert_eq!(got.metadata.backend, Backend::Gf2Gaussian);
        assert!(!got.metadata.theorem_certified);
    }
}

#[test]
fn affine_more_than_twenty_features_is_rejected() {
    let tree = TreeNode::Internal {
        predicate: affine((0..21).map(lit).collect(), true),
        left: Box::new(TreeNode::Leaf {
            class: 1,
            samples: 1,
        }),
        right: Box::new(TreeNode::Leaf {
            class: 0,
            samples: 1,
        }),
        majority_class: 0,
    };
    let domain = ColumnMajorMatrix::from_rows(&[vec![0.0; 21], vec![1.0; 21]]).unwrap();
    let got = weak_axp_check(&tree, &domain, &[1.0; 21], 1, &[0], false);
    assert!(!got.is_weak_axp);
    assert!(got
        .metadata
        .rejected_reason
        .as_deref()
        .unwrap_or("")
        .contains("limit"));
}

#[test]
fn affine_non_boolean_scope_is_rejected() {
    let tree = xor_tree();
    let domain = ColumnMajorMatrix::from_rows(&[vec![0.0, 0.0], vec![0.25, 1.0]]).unwrap();
    let got = weak_axp_check(&tree, &domain, &[1.0, 0.0], 1, &[0, 1], false);
    assert!(!got.is_weak_axp);
    assert!(got
        .metadata
        .rejected_reason
        .as_deref()
        .unwrap_or("")
        .contains("non-Boolean"));
}

#[test]
fn mixed_family_theorem_mode_is_rejected() {
    let mixed = TreeNode::Internal {
        predicate: Predicate::HornClause(vec![lit(0), lit(1).negated()]),
        left: Box::new(TreeNode::Internal {
            predicate: Predicate::Square2Cnf {
                a: lit(0),
                b: lit(1),
                c: lit(0),
                d: lit(1),
            },
            left: Box::new(TreeNode::Leaf {
                class: 1,
                samples: 1,
            }),
            right: Box::new(TreeNode::Leaf {
                class: 0,
                samples: 1,
            }),
            majority_class: 1,
        }),
        right: Box::new(TreeNode::Leaf {
            class: 1,
            samples: 1,
        }),
        majority_class: 1,
    };
    let domain = binary_domain(2);
    let got = weak_axp_check(&mixed, &domain, &[1.0, 1.0], 1, &[0, 1], true);
    assert!(!got.metadata.theorem_certified);
    assert_eq!(got.metadata.language_family, LanguageFamily::EmpiricalMixed);
}

#[test]
fn inconsistent_gf2_path_is_detected_as_blocked() {
    let p = affine(vec![lit(0), lit(1)], true);
    let tree = TreeNode::Internal {
        predicate: p.clone(),
        left: Box::new(TreeNode::Internal {
            predicate: p,
            left: Box::new(TreeNode::Leaf {
                class: 1,
                samples: 1,
            }),
            right: Box::new(TreeNode::Leaf {
                class: 0,
                samples: 1,
            }),
            majority_class: 1,
        }),
        right: Box::new(TreeNode::Leaf {
            class: 1,
            samples: 1,
        }),
        majority_class: 1,
    };
    let domain = binary_domain(2);
    let got = weak_axp_check(&tree, &domain, &[1.0, 0.0], 1, &[], false);
    assert!(got.is_weak_axp);
    assert_eq!(got.metadata.backend, Backend::Gf2Gaussian);
}

#[test]
fn certified_backend_metadata_requires_executed_backend_not_fallback() {
    let tree = TreeNode::Internal {
        predicate: Predicate::HornClause(vec![lit(0)]),
        left: Box::new(TreeNode::Leaf {
            class: 1,
            samples: 1,
        }),
        right: Box::new(TreeNode::Leaf {
            class: 0,
            samples: 1,
        }),
        majority_class: 1,
    };
    let domain = binary_domain(1);
    let got = weak_axp_check(&tree, &domain, &[1.0], 1, &[0], true);
    assert!(got.is_weak_axp);
    assert!(got.metadata.theorem_certified);
    assert_eq!(got.metadata.backend, Backend::StructuralHorn);
    assert_ne!(got.metadata.backend, Backend::IntervalDfsFallback);
}
