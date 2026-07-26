use smart_mdt_rs::{
    data::{BitSet, ColumnMajorMatrix, Dataset},
    logic::PathTheoryState,
    tree::{
        learn, CacheConfig, CachedSubtree, LanguagePolicy, LearnerConfig, SearchStateKey,
        TrainingContext, TreeNode,
    },
};
use std::sync::Arc;

fn dataset() -> Dataset {
    Dataset::new(
        ColumnMajorMatrix::from_rows(&[
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ])
        .unwrap(),
        vec![0, 1, 1, 0],
    )
    .unwrap()
}

fn key(
    rows: &BitSet,
    depth: usize,
    budget: usize,
    theory: PathTheoryState,
    score: &str,
    candidate: &str,
) -> SearchStateKey {
    SearchStateKey::new(rows, depth, budget, theory, score, candidate)
}

#[test]
fn state_key_includes_theory_depth_budget_and_full_config_identity() {
    let rows = BitSet::ones(8);
    let base = key(&rows, 3, 9, PathTheoryState::Horn, "score-a", "cand-a");
    assert_ne!(
        base,
        key(&rows, 2, 9, PathTheoryState::Horn, "score-a", "cand-a")
    );
    assert_ne!(
        base,
        key(&rows, 3, 8, PathTheoryState::Horn, "score-a", "cand-a")
    );
    assert_ne!(
        base,
        key(&rows, 3, 9, PathTheoryState::AntiHorn, "score-a", "cand-a")
    );
    assert_ne!(
        base,
        key(&rows, 3, 9, PathTheoryState::Horn, "score-b", "cand-a")
    );
    assert_ne!(
        base,
        key(&rows, 3, 9, PathTheoryState::Horn, "score-a", "cand-b")
    );
}

#[test]
fn all_cache_levels_hit_on_repeated_subproblems_without_cross_dataset_reuse() {
    let config = CacheConfig::all_enabled();
    let first = TrainingContext::with_cache_config(Arc::new(dataset()), config.clone());
    let node = first.root_view();
    let state = key(&node.rows, 3, 9, node.theory_state, "score", "candidates");
    first.node_statistics_cached(&state, &node).unwrap();
    first.node_statistics_cached(&state, &node).unwrap();
    first.insert_candidate_pool(state.clone(), Vec::new());
    assert!(first.candidate_pool_cached(&state).is_some());
    first.insert_lookahead(state.clone(), 0.25);
    assert_eq!(first.lookahead_cached(&state), Some(0.25));
    first.insert_best_subtree(
        state.clone(),
        CachedSubtree {
            tree: Arc::new(TreeNode::Leaf {
                class: 0,
                samples: 4,
            }),
            training_error: 0.5,
            validation_error: None,
            node_count: 1,
            leaf_count: 1,
            literal_count: 0,
            estimated_axp_length: None,
            objective: 0.5,
            path_certified: true,
        },
    );
    assert!(first.best_subtree_cached(&state).is_some());
    let diagnostics = first.cache_diagnostics();
    assert!(diagnostics.node_statistics.hits > 0);
    assert!(diagnostics.candidate_pools.hits > 0);
    assert!(diagnostics.best_subtrees.hits > 0);
    assert!(diagnostics.lookahead.hits > 0);

    let second = TrainingContext::with_cache_config(Arc::new(dataset()), config);
    assert!(second.candidate_pool_cached(&state).is_none());
    assert!(second.best_subtree_cached(&state).is_none());
}

#[test]
fn cache_on_off_and_eviction_preserve_deterministic_tree() {
    let dataset = dataset();
    let base = LearnerConfig {
        max_depth: 4,
        language_policy: LanguagePolicy::SmartCertified,
        cache: CacheConfig::disabled(),
        ..LearnerConfig::default()
    };
    let without = learn(&dataset, &base).unwrap();
    let with = learn(
        &dataset,
        &LearnerConfig {
            cache: CacheConfig::all_enabled(),
            ..base.clone()
        },
    )
    .unwrap();
    assert_eq!(with, without);

    let tiny = CacheConfig {
        max_entries: 1,
        approximate_byte_limit: 1024,
        ..CacheConfig::all_enabled()
    };
    let context = TrainingContext::with_cache_config(Arc::new(dataset.clone()), tiny.clone());
    let root = context.root_view();
    let first = key(&root.rows, 1, 1, root.theory_state, "a", "a");
    let second = key(&root.rows, 1, 1, root.theory_state, "b", "b");
    context.insert_lookahead(first.clone(), 1.0);
    context.insert_lookahead(second.clone(), 2.0);
    assert_eq!(context.lookahead_cached(&first), None);
    assert_eq!(context.lookahead_cached(&second), Some(2.0));
    assert_eq!(context.cache_diagnostics().lookahead.evictions, 1);

    let evicting_tree = learn(
        &dataset,
        &LearnerConfig {
            cache: tiny,
            ..base
        },
    )
    .unwrap();
    assert_eq!(evicting_tree, without);
}
