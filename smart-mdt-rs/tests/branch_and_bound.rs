use smart_mdt_rs::{
    data::Dataset,
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    search::{
        canonical_predicate_key, exact_branch_and_bound_top_k, family_order, score_split,
        top_k_with_config, BranchAndBoundConfig, SplitCandidate, SplitScoreConfig, SplitScoreInput,
    },
    tree::{learn, LanguagePolicy, LearnerConfig},
};
use std::sync::Arc;

fn literal(feature: u32, positive: bool) -> Literal {
    Literal {
        atom: ThresholdAtom {
            feature,
            threshold_id: 0,
            threshold: 0.5,
            op: ThresholdOp::GreaterEqual,
        },
        positive,
    }
}

fn predicates() -> Vec<Predicate> {
    vec![
        Predicate::Unary(literal(0, true)),
        Predicate::HornClause(vec![literal(0, false), literal(1, true)]),
        Predicate::AntiHornClause(vec![literal(1, true), literal(2, false)]),
        Predicate::Square2Cnf {
            a: literal(0, true),
            b: literal(1, false),
            c: literal(2, true),
            d: literal(3, false),
        },
        Predicate::Affine {
            literals: vec![literal(0, true), literal(2, true)],
            rhs: true,
        },
    ]
}

fn candidates(config: &SplitScoreConfig) -> Vec<SplitCandidate> {
    predicates()
        .into_iter()
        .enumerate()
        .flat_map(|(family_index, predicate)| {
            (0..4).map(move |offset| {
                let gain = 0.1 + family_index as f64 * 0.04 + offset as f64 * 0.007;
                SplitCandidate {
                    score: score_split(
                        SplitScoreInput {
                            information_gain: gain,
                            true_count: 7 + offset,
                            false_count: 13 - offset,
                            literal_count: predicate.arity(),
                            family: predicate.language(),
                            fragmentation: 0.3,
                            estimated_subtree_cost: 0.4,
                            instability: 0.1,
                        },
                        config,
                    ),
                    predicate: predicate.clone(),
                    left_count: 7 + offset,
                    right_count: 13 - offset,
                }
            })
        })
        .collect()
}

fn keys(candidates: &[SplitCandidate]) -> Vec<String> {
    candidates
        .iter()
        .map(|candidate| {
            format!(
                "{}:{}:{}",
                canonical_predicate_key(&candidate.predicate),
                candidate.score.final_score.to_bits(),
                candidate.left_count
            )
        })
        .collect()
}

#[test]
fn exact_bounded_top_k_matches_exhaustive_for_all_families() {
    let score_config = SplitScoreConfig::sparse_certified();
    let exhaustive_pool = candidates(&score_config);
    for top_k in 1..=8 {
        let expected = top_k_with_config(exhaustive_pool.clone(), top_k, &score_config);
        let config = BranchAndBoundConfig {
            enabled: true,
            top_k,
            ..BranchAndBoundConfig::default()
        };
        let (actual, diagnostics) = exact_branch_and_bound_top_k(
            exhaustive_pool.clone(),
            &config,
            &score_config,
            |predicate| {
                Arc::<[u64]>::from(vec![
                    predicate.arity() as u64,
                    family_order(predicate.language()) as u64,
                ])
            },
        );
        assert_eq!(keys(&actual), keys(&expected));
        assert_eq!(
            actual[0].predicate.certificate(true),
            expected[0].predicate.certificate(true)
        );
        assert!(diagnostics.partial_states_created >= actual.len());
        assert_eq!(diagnostics.exhaustive_fallback_count, 0);
    }
}

#[test]
fn bounded_search_is_deterministic_and_fallback_is_exact() {
    let score_config = SplitScoreConfig::information_gain();
    let pool = candidates(&score_config);
    let config = BranchAndBoundConfig {
        enabled: true,
        top_k: 3,
        ..BranchAndBoundConfig::default()
    };
    let run = || {
        exact_branch_and_bound_top_k(pool.clone(), &config, &score_config, |_| {
            Arc::<[u64]>::from(vec![1, 2, 3])
        })
    };
    let (first, first_diag) = run();
    let (second, second_diag) = run();
    assert_eq!(keys(&first), keys(&second));
    assert_eq!(first_diag, second_diag);
    assert!(first_diag.partial_states_pruned > 0);

    let fallback = BranchAndBoundConfig {
        max_partial_states: 1,
        ..config
    };
    let (fallback_result, diagnostics) =
        exact_branch_and_bound_top_k(pool.clone(), &fallback, &score_config, |_| {
            Arc::<[u64]>::from(vec![0])
        });
    assert_eq!(
        keys(&fallback_result),
        keys(&top_k_with_config(pool, 3, &score_config))
    );
    assert_eq!(diagnostics.exhaustive_fallback_count, 1);
}

#[test]
fn learner_tree_is_identical_with_exact_bound_enabled() {
    let dataset = Dataset::from_dl8_like("tests/fixtures/horn_separable.dl8").unwrap();
    let base = LearnerConfig {
        max_depth: 4,
        language_policy: LanguagePolicy::SmartCertified,
        ..LearnerConfig::default()
    };
    let without = learn(&dataset, &base).unwrap();
    let with = learn(
        &dataset,
        &LearnerConfig {
            branch_and_bound: BranchAndBoundConfig {
                enabled: true,
                top_k: base.max_candidates_per_node,
                ..BranchAndBoundConfig::default()
            },
            ..base
        },
    )
    .unwrap();
    assert_eq!(with, without);
}
