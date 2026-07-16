use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    tree::{
        deterministic_pruning_split, learn, prune_with_validation, tree_is_certified,
        ClassAwarePruningConfig, LanguagePolicy, LearnerConfig, PruningConfig, PruningReason,
        PruningSelectionMode, TreeNode,
    },
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

fn dataset() -> Dataset {
    let rows: Vec<Vec<f64>> = (0..20)
        .map(|row| vec![(row % 2) as f64, ((row / 2) % 2) as f64])
        .collect();
    let labels = (0..20).map(|row| u32::from(row % 5 == 0)).collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

fn overgrown_tree() -> TreeNode {
    TreeNode::Internal {
        predicate: Predicate::Unary(lit(0)),
        majority_class: 0,
        left: Box::new(TreeNode::Internal {
            predicate: Predicate::HornClause(vec![lit(0), lit(1).negated()]),
            majority_class: 0,
            left: Box::new(TreeNode::Leaf {
                class: 0,
                samples: 5,
            }),
            right: Box::new(TreeNode::Leaf {
                class: 0,
                samples: 5,
            }),
        }),
        right: Box::new(TreeNode::Leaf {
            class: 0,
            samples: 10,
        }),
    }
}

fn separating_tree(samples: usize) -> TreeNode {
    TreeNode::Internal {
        predicate: Predicate::Unary(lit(0)),
        majority_class: 0,
        left: Box::new(TreeNode::Leaf {
            class: 1,
            samples: samples / 2,
        }),
        right: Box::new(TreeNode::Leaf {
            class: 0,
            samples: samples - samples / 2,
        }),
    }
}

fn binary_dataset(majority: usize, minority: usize) -> Dataset {
    let mut rows = vec![vec![0.0]; majority];
    rows.extend(vec![vec![1.0]; minority]);
    let mut labels = vec![0; majority];
    labels.extend(vec![1; minority]);
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

fn class_aware() -> ClassAwarePruningConfig {
    ClassAwarePruningConfig {
        enabled: true,
        accuracy_epsilon: 0.5,
        balanced_accuracy_epsilon: 0.01,
        minimum_minority_recall: None,
        minimum_validation_samples: 10,
        minimum_validation_samples_per_class: 2,
        root_collapse_majority_threshold: 0.9,
        preserve_subtree_when_evidence_insufficient: true,
    }
}

#[test]
fn deterministic_stratified_validation_split_has_no_overlap() {
    let data = dataset();
    let first = deterministic_pruning_split(&data, 0.25, 42).unwrap();
    let second = deterministic_pruning_split(&data, 0.25, 42).unwrap();
    assert_eq!(first.grow_indices, second.grow_indices);
    assert_eq!(first.validation_indices, second.validation_indices);
    assert!(first
        .grow_indices
        .iter()
        .all(|index| !first.validation_indices.contains(index)));
    assert_eq!(
        first.grow_indices.len() + first.validation_indices.len(),
        data.labels.len()
    );
}

#[test]
fn pruning_reduces_known_overgrown_tree_without_mutating_original() {
    let validation = dataset();
    let original = overgrown_tree();
    let snapshot = original.clone();
    let (pruned, diagnostics) = prune_with_validation(
        &original,
        &validation,
        &PruningConfig {
            enabled: true,
            accuracy_epsilon: 0.25,
            ..PruningConfig::default()
        },
    );
    assert_eq!(original, snapshot);
    assert!(pruned.nodes() < original.nodes());
    assert!(pruned.literals() <= original.literals());
    assert!(diagnostics.nodes_after <= diagnostics.nodes_before);
    assert!(diagnostics.literals_after <= diagnostics.literals_before);
    assert!(diagnostics.validation_accuracy_after + 0.25 >= diagnostics.validation_accuracy_before);
    assert!(diagnostics.path_certified_after);
    assert!(tree_is_certified(&pruned));
}

#[test]
fn pure_leaf_is_preserved_and_cart_mode_is_supported() {
    let validation = dataset();
    let leaf = TreeNode::Leaf {
        class: 0,
        samples: 20,
    };
    let (same, diagnostics) = prune_with_validation(
        &leaf,
        &validation,
        &PruningConfig {
            enabled: true,
            selection_mode: PruningSelectionMode::Cart,
            alpha_leaves: 0.01,
            ..PruningConfig::default()
        },
    );
    assert_eq!(same, leaf);
    assert_eq!(diagnostics.nodes_before, diagnostics.nodes_after);
}

#[test]
fn disabled_pruning_reproduces_training_and_enabled_fit_is_certified() {
    let data = dataset();
    let base = LearnerConfig {
        max_depth: 3,
        beam_width: 8,
        language_policy: LanguagePolicy::SmartCertified,
        ..LearnerConfig::default()
    };
    let expected = learn(&data, &base).unwrap();
    let disabled = learn(
        &data,
        &LearnerConfig {
            pruning: PruningConfig {
                enabled: false,
                ..PruningConfig::default()
            },
            ..base.clone()
        },
    )
    .unwrap();
    assert_eq!(disabled, expected);

    let pruned = learn(
        &data,
        &LearnerConfig {
            pruning: PruningConfig {
                enabled: true,
                accuracy_epsilon: 0.05,
                ..PruningConfig::default()
            },
            ..base
        },
    )
    .unwrap();
    assert!(tree_is_certified(&pruned));
}

#[test]
fn balanced_validation_rejects_constant_root() {
    let validation = binary_dataset(20, 20);
    let original = separating_tree(40);
    let (pruned, diagnostics) = prune_with_validation(
        &original,
        &validation,
        &PruningConfig {
            enabled: true,
            accuracy_epsilon: 0.5,
            class_aware: class_aware(),
            ..PruningConfig::default()
        },
    );
    assert_eq!(pruned, original);
    assert_eq!(
        diagnostics.root_decision_reason,
        PruningReason::RootCollapseGuard
    );
    assert_eq!(diagnostics.validation_metrics_before.balanced_accuracy, 1.0);
    assert_eq!(diagnostics.validation_metrics_before.class_support[&0], 20);
    assert_eq!(diagnostics.validation_metrics_before.class_support[&1], 20);
}

#[test]
fn highly_imbalanced_validation_allows_equivalent_constant_root() {
    let validation = binary_dataset(95, 5);
    let original = TreeNode::Internal {
        predicate: Predicate::Unary(lit(0)),
        majority_class: 0,
        left: Box::new(TreeNode::Leaf {
            class: 0,
            samples: 5,
        }),
        right: Box::new(TreeNode::Leaf {
            class: 0,
            samples: 95,
        }),
    };
    let (pruned, diagnostics) = prune_with_validation(
        &original,
        &validation,
        &PruningConfig {
            enabled: true,
            class_aware: class_aware(),
            ..PruningConfig::default()
        },
    );
    assert_eq!(pruned.nodes(), 1);
    assert_eq!(
        diagnostics.root_decision_reason,
        PruningReason::ObjectiveImproved
    );
    assert!(tree_is_certified(&pruned));
}

#[test]
fn minority_recall_loss_blocks_pruning() {
    let validation = binary_dataset(95, 5);
    let original = separating_tree(100);
    let mut aware = class_aware();
    aware.balanced_accuracy_epsilon = 1.0;
    aware.minimum_minority_recall = Some(0.5);
    let (pruned, diagnostics) = prune_with_validation(
        &original,
        &validation,
        &PruningConfig {
            enabled: true,
            accuracy_epsilon: 0.1,
            class_aware: aware,
            ..PruningConfig::default()
        },
    );
    assert_eq!(pruned, original);
    assert_eq!(
        diagnostics.root_decision_reason,
        PruningReason::MinorityRecallGuard
    );
    assert_eq!(diagnostics.validation_metrics_before.minority_recall, 1.0);
    assert_eq!(diagnostics.validation_metrics_after.minority_recall, 1.0);
    assert!(diagnostics.path_certified_after);
}

#[test]
fn disabled_class_aware_mode_reproduces_legacy_pruning() {
    let validation = binary_dataset(20, 20);
    let original = separating_tree(40);
    let legacy = PruningConfig {
        enabled: true,
        accuracy_epsilon: 0.5,
        ..PruningConfig::default()
    };
    let (expected, _) = prune_with_validation(&original, &validation, &legacy);
    let (actual, diagnostics) = prune_with_validation(
        &original,
        &validation,
        &PruningConfig {
            class_aware: ClassAwarePruningConfig {
                enabled: false,
                root_collapse_majority_threshold: 1.0,
                minimum_validation_samples: usize::MAX,
                ..ClassAwarePruningConfig::default()
            },
            ..legacy
        },
    );
    assert_eq!(actual, expected);
    assert_eq!(actual.nodes(), 1);
    assert_eq!(
        diagnostics.root_decision_reason,
        PruningReason::ObjectiveImproved
    );
}
