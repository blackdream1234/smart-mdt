use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    search::canonical_predicate_key,
    tree::{
        learn_with_diagnostics, tree_path_theory_metadata, CacheConfig, LanguagePolicy,
        LearnerConfig, ParallelConfig, TrainingContext,
    },
};
use std::sync::Arc;

fn dataset() -> Dataset {
    let rows: Vec<Vec<f64>> = (0..32)
        .map(|mask| (0..5).map(|bit| ((mask >> bit) & 1) as f64).collect())
        .collect();
    let labels = rows
        .iter()
        .map(|row| u32::from((row[0] == row[1]) ^ (row[2] == 1.0)))
        .collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

fn config(threads: Option<usize>) -> LearnerConfig {
    LearnerConfig {
        max_depth: 3,
        beam_width: 10,
        max_candidates_per_node: 32,
        cache: CacheConfig::all_enabled(),
        parallel: ParallelConfig {
            enabled: threads != Some(1),
            threads,
            parallel_candidates: true,
            parallel_beam_states: true,
            minimum_parallel_work: 2,
        },
        language_policy: LanguagePolicy::SmartCertified,
        theorem_mode: true,
        ..LearnerConfig::default()
    }
}

#[test]
fn serial_and_parallel_candidate_lists_are_identical() {
    let data = dataset();
    let context = TrainingContext::new(Arc::new(data));
    let node = context.root_view();
    let serial = context
        .generate_candidates_parallel(
            &node,
            LanguagePolicy::SmartCertified,
            1,
            10,
            &Default::default(),
            &ParallelConfig::disabled(),
        )
        .unwrap();
    let parallel = context
        .generate_candidates_parallel(
            &node,
            LanguagePolicy::SmartCertified,
            1,
            10,
            &Default::default(),
            &ParallelConfig {
                enabled: true,
                threads: Some(4),
                ..ParallelConfig::default()
            },
        )
        .unwrap();
    let summarize = |candidates: &[smart_mdt_rs::search::SplitCandidate]| {
        let mut rows = candidates
            .iter()
            .map(|candidate| {
                (
                    canonical_predicate_key(&candidate.predicate),
                    candidate.score,
                    context
                        .full_predicate_mask(&candidate.predicate)
                        .words()
                        .to_vec(),
                )
            })
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| left.0.cmp(&right.0));
        rows
    };
    assert_eq!(summarize(&serial), summarize(&parallel));
}

#[test]
fn one_two_four_and_default_threads_select_identical_trees_and_metadata() {
    let data = dataset();
    let mut results = Vec::new();
    for threads in [Some(1), Some(2), Some(4), None] {
        let (tree, diagnostics) = learn_with_diagnostics(&data, &config(threads)).unwrap();
        results.push((tree.clone(), tree_path_theory_metadata(&tree)));
        if threads != Some(1) {
            assert!(diagnostics.parallel.candidate_batches_parallelized > 0);
            assert!(diagnostics.parallel.family_tasks > 0);
        }
    }
    assert!(results.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn cache_and_parallel_toggles_preserve_the_selected_model() {
    let data = dataset();
    let (expected, _) = learn_with_diagnostics(&data, &config(Some(1))).unwrap();
    let mut parallel_no_cache = config(Some(4));
    parallel_no_cache.cache = CacheConfig::disabled();
    let (actual, _) = learn_with_diagnostics(&data, &parallel_no_cache).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(
        tree_path_theory_metadata(&actual),
        tree_path_theory_metadata(&expected)
    );
}
