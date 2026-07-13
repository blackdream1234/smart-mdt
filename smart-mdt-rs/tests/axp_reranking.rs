use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    logic::{Backend, Literal, Predicate, ThresholdAtom, ThresholdOp},
    search::{score_split, SplitCandidate, SplitScoreConfig, SplitScoreInput},
    tree::{
        learn_with_diagnostics, tree_is_certified, AxpRerankConfig, LanguagePolicy, LearnerConfig,
        TrainingContext,
    },
};
use std::sync::Arc;

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
    Dataset::new(
        ColumnMajorMatrix::from_rows(&[
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ])
        .unwrap(),
        vec![0, 0, 1, 0],
    )
    .unwrap()
}

fn candidate(predicate: Predicate, final_score: f64) -> SplitCandidate {
    let mut score = score_split(
        SplitScoreInput {
            information_gain: 0.25,
            true_count: 2,
            false_count: 2,
            literal_count: predicate.arity(),
            family: predicate.language(),
            fragmentation: 0.0,
            estimated_subtree_cost: 0.25,
            instability: 0.0,
        },
        &SplitScoreConfig::sparse_certified(),
    );
    score.final_score = final_score;
    SplitCandidate {
        predicate,
        score,
        left_count: 2,
        right_count: 2,
    }
}

fn pair(horn_score: f64) -> Vec<SplitCandidate> {
    vec![
        candidate(Predicate::Unary(lit(0)), 0.25),
        candidate(
            Predicate::HornClause(vec![lit(0).negated(), lit(1)]),
            horn_score,
        ),
    ]
}

#[test]
fn lower_axp_candidate_wins_equal_predictive_shortlist_with_certified_metadata() {
    let context = TrainingContext::new(Arc::new(dataset()));
    let node = context.root_view();
    let reranked = context
        .rerank_candidates_by_axp(
            &node,
            pair(0.25),
            &SplitScoreConfig::sparse_certified(),
            &AxpRerankConfig {
                enabled: true,
                shortlist_size: 2,
                validation_samples: 4,
                weight_mean_axp: 0.1,
                weight_max_axp: 0.05,
                timeout_ms: None,
            },
            42,
        )
        .unwrap();
    assert!(matches!(reranked[0].predicate, Predicate::Unary(_)));
    let diagnostics = context.diagnostics().axp_rerank;
    assert_eq!(diagnostics.candidates_evaluated, 2);
    assert!(diagnostics
        .candidates
        .iter()
        .all(|candidate| candidate.theorem_certified));
    assert!(diagnostics.candidates.iter().all(|candidate| matches!(
        candidate.backend,
        Backend::StructuralHorn
            | Backend::StructuralAntiHorn
            | Backend::TwoSat
            | Backend::Gf2Gaussian
    )));
}

#[test]
fn materially_better_predictive_candidate_is_not_hidden_with_small_weights() {
    let context = TrainingContext::new(Arc::new(dataset()));
    let reranked = context
        .rerank_candidates_by_axp(
            &context.root_view(),
            pair(0.75),
            &SplitScoreConfig::sparse_certified(),
            &AxpRerankConfig {
                enabled: true,
                shortlist_size: 2,
                validation_samples: 4,
                weight_mean_axp: 0.001,
                weight_max_axp: 0.001,
                timeout_ms: None,
            },
            42,
        )
        .unwrap();
    assert!(matches!(reranked[0].predicate, Predicate::HornClause(_)));
}

#[test]
fn deterministic_sampling_and_timeout_retain_original_scores() {
    let config = AxpRerankConfig {
        enabled: true,
        shortlist_size: 2,
        validation_samples: 3,
        weight_mean_axp: 1.0,
        weight_max_axp: 1.0,
        timeout_ms: Some(0),
    };
    let run = || {
        let context = TrainingContext::new(Arc::new(dataset()));
        let candidates = context
            .rerank_candidates_by_axp(
                &context.root_view(),
                pair(0.25),
                &SplitScoreConfig::sparse_certified(),
                &config,
                7,
            )
            .unwrap();
        (candidates, context.diagnostics().axp_rerank)
    };
    let (first, first_diagnostics) = run();
    let (second, second_diagnostics) = run();
    assert_eq!(
        first
            .iter()
            .map(|candidate| candidate.score.final_score)
            .collect::<Vec<_>>(),
        vec![0.25, 0.25]
    );
    assert_eq!(
        first
            .iter()
            .map(|candidate| format!("{:?}", candidate.predicate))
            .collect::<Vec<_>>(),
        second
            .iter()
            .map(|candidate| format!("{:?}", candidate.predicate))
            .collect::<Vec<_>>()
    );
    assert_eq!(first_diagnostics.timeout_count, 2);
    assert_eq!(
        first_diagnostics
            .candidates
            .iter()
            .map(|candidate| (
                candidate.canonical_predicate.clone(),
                candidate.validation_samples
            ))
            .collect::<Vec<_>>(),
        second_diagnostics
            .candidates
            .iter()
            .map(|candidate| (
                candidate.canonical_predicate.clone(),
                candidate.validation_samples
            ))
            .collect::<Vec<_>>()
    );
}

#[test]
fn enabled_learner_reranks_only_certified_candidates() {
    let data = dataset();
    let (tree, diagnostics) = learn_with_diagnostics(
        &data,
        &LearnerConfig {
            max_depth: 2,
            beam_width: 8,
            language_policy: LanguagePolicy::SmartCertified,
            axp_rerank: AxpRerankConfig {
                enabled: true,
                shortlist_size: 4,
                validation_samples: 4,
                ..AxpRerankConfig::default()
            },
            ..LearnerConfig::default()
        },
    )
    .unwrap();
    assert!(tree_is_certified(&tree));
    assert!(diagnostics
        .axp_rerank
        .candidates
        .iter()
        .all(|candidate| candidate.theorem_certified));
}
