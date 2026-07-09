use smart_mdt_rs::{
    data::ColumnMajorMatrix,
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
