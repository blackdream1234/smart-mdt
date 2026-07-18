use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    explain::extract_final_tree_axps,
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    tree::{learn_with_diagnostics, LanguagePolicy, LearnerConfig, PruningConfig, TreeNode},
};

fn dataset() -> Dataset {
    let rows = (0..32)
        .map(|mask| {
            (0..5)
                .map(|bit| ((mask >> bit) & 1) as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let labels = rows
        .iter()
        .map(|row| u32::from(row[0] == 1.0 || row[1] == 1.0))
        .collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

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
            samples: 8,
        }),
    }
}

#[test]
fn default_workflow_extracts_axps_only_after_final_tree_selection() {
    let data = dataset();
    let config = LearnerConfig {
        max_depth: 3,
        language_policy: LanguagePolicy::SmartCertified,
        pruning: PruningConfig {
            enabled: true,
            ..PruningConfig::default()
        },
        ..LearnerConfig::default()
    };
    assert!(!config.axp_rerank.enabled);
    let (final_tree, diagnostics) = learn_with_diagnostics(&data, &config).unwrap();
    assert_eq!(diagnostics.axp_rerank.candidates_evaluated, 0);

    let first = extract_final_tree_axps(&final_tree, &data.features, 8, true);
    let second = extract_final_tree_axps(&final_tree, &data.features, 8, true);
    assert_eq!(first.results.len(), 8);
    assert_eq!(first.mean_length, second.mean_length);
    assert_eq!(first.max_length, second.max_length);
    assert!(first.theorem_certified);
    assert_eq!(
        first
            .results
            .iter()
            .map(|result| (&result.features, &result.metadata))
            .collect::<Vec<_>>(),
        second
            .results
            .iter()
            .map(|result| (&result.features, &result.metadata))
            .collect::<Vec<_>>()
    );
}

#[test]
fn dataset_axp_metric_includes_rows_after_the_eighth_prefix() {
    let mut rows = vec![vec![0.0, 0.0]; 8];
    rows.push(vec![1.0, 1.0]);
    let features = ColumnMajorMatrix::from_rows(&rows).unwrap();
    let prefix = extract_final_tree_axps(&and_tree(), &features, 8, true);
    let complete = extract_final_tree_axps(&and_tree(), &features, features.rows(), true);
    assert_eq!(prefix.results.len(), 8);
    assert_eq!(prefix.mean_length, 1.0);
    assert_eq!(prefix.max_length, 1);
    assert_eq!(complete.results.len(), 9);
    assert_eq!(complete.mean_length, 10.0 / 9.0);
    assert_eq!(complete.max_length, 2);
    assert!(complete.theorem_certified);
}
