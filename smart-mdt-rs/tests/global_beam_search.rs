use smart_mdt_rs::{
    data::{BitSet, ColumnMajorMatrix, Dataset},
    logic::{Literal, PathTheoryState, Predicate, ThresholdAtom, ThresholdOp},
    tree::{
        learn, learn_with_diagnostics, predict_all, tree_is_certified, FrontierLeaf,
        LanguagePolicy, LearnerConfig, PartialTree, PartialTreeState, TreeNode, TreeSearchStrategy,
    },
};

fn dataset() -> Dataset {
    let rows: Vec<Vec<f64>> = (0..16)
        .map(|mask| (0..4).map(|bit| ((mask >> bit) & 1) as f64).collect())
        .collect();
    let labels = rows
        .iter()
        .map(|row| u32::from((row[0] == 1.0) ^ (row[1] == 1.0) || row[2] == 1.0))
        .collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

fn config(strategy: TreeSearchStrategy) -> LearnerConfig {
    let mut cfg = LearnerConfig {
        max_depth: 3,
        beam_width: 12,
        max_candidates_per_node: 32,
        language_policy: LanguagePolicy::SmartCertified,
        theorem_mode: true,
        ..LearnerConfig::default()
    };
    cfg.tree_search.strategy = strategy;
    cfg.tree_search.candidate_beam_width = 12;
    cfg.tree_search.tree_beam_width = 4;
    cfg.tree_search.max_expansions = 64;
    cfg
}

fn mistakes(tree: &TreeNode, data: &Dataset) -> usize {
    predict_all(tree, &data.features)
        .iter()
        .zip(&data.labels)
        .filter(|(predicted, actual)| predicted != actual)
        .count()
}

#[test]
fn beam_width_one_reproduces_equivalent_greedy_tree() {
    let data = dataset();
    let greedy = learn(&data, &config(TreeSearchStrategy::Greedy)).unwrap();
    let mut beam = config(TreeSearchStrategy::GlobalBeam);
    beam.tree_search.tree_beam_width = 1;
    beam.tree_search.max_expansions = usize::MAX;
    let searched = learn(&data, &beam).unwrap();
    assert_eq!(searched, greedy);
}

#[test]
fn global_beam_is_certified_deterministic_and_not_worse_than_greedy() {
    let data = dataset();
    let greedy = learn(&data, &config(TreeSearchStrategy::Greedy)).unwrap();
    let cfg = config(TreeSearchStrategy::GlobalBeam);
    let (first, diagnostics) = learn_with_diagnostics(&data, &cfg).unwrap();
    let second = learn(&data, &cfg).unwrap();
    assert_eq!(first, second);
    assert!(tree_is_certified(&first));
    assert_eq!(
        diagnostics
            .beam_search
            .path_incompatible_candidates_rejected,
        0
    );
    assert!(diagnostics.beam_search.states_generated > 0);
    assert!(mistakes(&first, &data) <= mistakes(&greedy, &data));
}

#[test]
fn beam_respects_depth_node_and_expansion_budgets() {
    let data = dataset();
    let mut cfg = config(TreeSearchStrategy::GlobalBeam);
    cfg.max_depth = 2;
    cfg.tree_search.node_budget = 5;
    cfg.tree_search.max_expansions = 2;
    let (tree, diagnostics) = learn_with_diagnostics(&data, &cfg).unwrap();
    assert!(tree.depth() <= 2);
    assert!(tree.nodes() <= 5);
    assert!(tree_is_certified(&tree));
    assert!(diagnostics.beam_search.expansion_budget_reached);
}

#[test]
fn timeout_and_sparse_lookahead_return_complete_certified_trees() {
    let data = dataset();
    let mut timeout = config(TreeSearchStrategy::GlobalBeam);
    timeout.tree_search.time_budget_ms = Some(0);
    let (timed_tree, diagnostics) = learn_with_diagnostics(&data, &timeout).unwrap();
    assert!(diagnostics.beam_search.time_budget_reached);
    assert!(tree_is_certified(&timed_tree));
    assert_eq!(
        predict_all(&timed_tree, &data.features).len(),
        data.labels.len()
    );

    let sparse = learn(&data, &config(TreeSearchStrategy::SparseLookahead)).unwrap();
    assert!(tree_is_certified(&sparse));
    assert_eq!(
        predict_all(&sparse, &data.features).len(),
        data.labels.len()
    );
}

#[test]
fn partial_tree_completion_keeps_independent_branch_theory_states() {
    let literal = |feature| Literal {
        atom: ThresholdAtom {
            feature,
            threshold_id: 0,
            threshold: 0.5,
            op: ThresholdOp::GreaterEqual,
        },
        positive: true,
    };
    let partial = PartialTree::Internal {
        predicate: Predicate::Unary(literal(0)),
        left: Box::new(PartialTree::Leaf {
            node_id: 1,
            class: 0,
            samples: 4,
        }),
        right: Box::new(PartialTree::Leaf {
            node_id: 2,
            class: 1,
            samples: 4,
        }),
        majority_class: 0,
    };
    let state = PartialTreeState {
        tree: partial.clone(),
        frontier: vec![
            FrontierLeaf {
                node_id: 1,
                rows: BitSet::ones(8),
                depth: 1,
                theory_state: PathTheoryState::Horn,
                majority_class: 0,
            },
            FrontierLeaf {
                node_id: 2,
                rows: BitSet::ones(8),
                depth: 1,
                theory_state: PathTheoryState::AffineGf2,
                majority_class: 1,
            },
        ],
        training_error_lower_bound: 0.0,
        complexity_cost: 0.0,
        objective_lower_bound: 0.0,
        expanded_nodes: 1,
        generated_order: 0,
    };
    assert_ne!(
        state.frontier[0].theory_state,
        state.frontier[1].theory_state
    );
    let complete: TreeNode = partial.into();
    assert!(tree_is_certified(&complete));
}
