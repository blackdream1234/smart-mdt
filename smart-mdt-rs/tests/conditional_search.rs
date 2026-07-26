use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    tree::{
        learn_with_diagnostics, tree_is_certified, ConditionalCandidateSearchConfig,
        LanguagePolicy, LearnerConfig, TreeSearchStrategy,
    },
};

fn dataset() -> Dataset {
    let rows = (0..64)
        .map(|mask| {
            (0..6)
                .map(|bit| ((mask >> bit) & 1) as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let labels = rows
        .iter()
        .map(|row| u32::from((row[0] == 1.0) ^ (row[1] == 1.0) || row[2] == 1.0))
        .collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

fn config() -> LearnerConfig {
    let mut config = LearnerConfig {
        max_depth: 3,
        beam_width: 8,
        max_candidates_per_node: 16,
        language_policy: LanguagePolicy::SmartCertified,
        conditional_search: ConditionalCandidateSearchConfig {
            enabled: true,
            branch_and_bound_candidate_threshold: 0,
            candidate_cache_minimum_expected_reuse: 2,
        },
        ..LearnerConfig::default()
    };
    config.branch_and_bound.enabled = true;
    config.branch_and_bound.top_k = 8;
    config.cache.candidate_pools = true;
    config.tree_search.strategy = TreeSearchStrategy::SelectiveLookahead;
    config.tree_search.selective.enabled = true;
    config.tree_search.selective.large_node_threshold = 1;
    config.tree_search.selective.candidate_beam_width = 8;
    config.tree_search.selective.tree_beam_width = 3;
    config.tree_search.max_expansions = 32;
    config
}

#[test]
fn conditional_search_activates_bounded_selection_and_reuse_cache() {
    let (tree, diagnostics) = learn_with_diagnostics(&dataset(), &config()).unwrap();
    assert!(tree_is_certified(&tree));
    assert!(
        diagnostics
            .conditional_search
            .branch_and_bound_activation_count
            > 0
    );
    assert!(diagnostics.conditional_search.cache_activation_count > 0);
    assert!(diagnostics.cache.candidate_pools.hits > 0);
}

#[test]
fn high_threshold_avoids_branch_and_bound_without_changing_tree() {
    let data = dataset();
    let mut direct = config();
    direct
        .conditional_search
        .branch_and_bound_candidate_threshold = usize::MAX;
    let (direct_tree, diagnostics) = learn_with_diagnostics(&data, &direct).unwrap();
    let mut bounded = direct.clone();
    bounded.conditional_search.enabled = false;
    let (bounded_tree, _) = learn_with_diagnostics(&data, &bounded).unwrap();
    assert_eq!(direct_tree, bounded_tree);
    assert_eq!(
        diagnostics
            .conditional_search
            .branch_and_bound_activation_count,
        0
    );
    assert!(
        diagnostics
            .conditional_search
            .branch_and_bound_avoided_count
            > 0
    );
}

#[test]
fn greedy_search_does_not_activate_candidate_cache_without_expected_reuse() {
    let mut config = config();
    config.tree_search.strategy = TreeSearchStrategy::Greedy;
    let (_, diagnostics) = learn_with_diagnostics(&dataset(), &config).unwrap();
    assert_eq!(diagnostics.conditional_search.cache_activation_count, 0);
    assert_eq!(diagnostics.cache.candidate_pools.hits, 0);
    assert_eq!(diagnostics.cache.candidate_pools.insertions, 0);
}
