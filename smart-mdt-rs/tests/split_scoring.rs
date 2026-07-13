use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    logic::{LanguageFamily, Literal, PathTheoryState, Predicate, ThresholdAtom, ThresholdOp},
    search::{
        canonical_predicate_key, score_split, top_k_with_config, SplitCandidate, SplitScoreConfig,
        SplitScoreInput,
    },
    tree::TrainingContext,
};
use std::sync::Arc;

fn literal(feature: u32) -> Literal {
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

fn predicate(arity: usize) -> Predicate {
    match arity {
        1 => Predicate::Unary(literal(0)),
        2 => Predicate::HornClause(vec![literal(0).negated(), literal(1)]),
        _ => Predicate::Square2Cnf {
            a: literal(0),
            b: literal(1),
            c: literal(2),
            d: literal(3),
        },
    }
}

fn candidate(gain: f64, arity: usize, config: &SplitScoreConfig) -> SplitCandidate {
    let predicate = predicate(arity);
    SplitCandidate {
        score: score_split(
            SplitScoreInput {
                information_gain: gain,
                true_count: 8,
                false_count: 8,
                literal_count: predicate.arity(),
                family: predicate.language(),
                fragmentation: 0.0,
                estimated_subtree_cost: 0.25,
                instability: 0.0,
            },
            config,
        ),
        predicate,
        left_count: 8,
        right_count: 8,
    }
}

#[test]
fn score_is_deterministic_and_components_sum_exactly() {
    let config = SplitScoreConfig::sparse_certified();
    let input = SplitScoreInput {
        information_gain: 0.31,
        true_count: 7,
        false_count: 9,
        literal_count: 2,
        family: LanguageFamily::Horn,
        fragmentation: 0.125,
        estimated_subtree_cost: 0.4,
        instability: 0.25,
    };
    let first = score_split(input, &config);
    let second = score_split(input, &config);
    assert_eq!(first, second);
    let sum = config.gain_weight * first.predictive_gain
        + config.gain_ratio_weight * first.gain_ratio
        + first.balance_component
        - first.literal_penalty
        - first.family_penalty
        - first.fragmentation_penalty
        - first.estimated_subtree_penalty
        - first.instability_penalty;
    assert!((sum - first.final_score).abs() < 1e-15);
    assert_eq!(first.certificate_bonus, 0.0);
}

#[test]
fn configured_epsilon_prefers_simpler_predicate_but_not_over_material_gain() {
    let mut epsilon_config = SplitScoreConfig::information_gain();
    epsilon_config.literal_penalty = 0.0;
    epsilon_config.tie_epsilon = 0.01;
    let close = top_k_with_config(
        vec![
            candidate(0.505, 4, &epsilon_config),
            candidate(0.5, 1, &epsilon_config),
        ],
        2,
        &epsilon_config,
    );
    assert_eq!(close[0].predicate.arity(), 1);

    let sparse = SplitScoreConfig::sparse_certified();
    let material = top_k_with_config(
        vec![candidate(0.62, 4, &sparse), candidate(0.5, 1, &sparse)],
        2,
        &sparse,
    );
    assert_eq!(material[0].predicate.arity(), 4);
}

#[test]
fn information_gain_profile_reproduces_historical_ranking() {
    let config = SplitScoreConfig::information_gain();
    let mut candidates = vec![
        candidate(0.3, 1, &config),
        candidate(0.39, 2, &config),
        candidate(0.48, 4, &config),
    ];
    let mut historical = candidates
        .iter()
        .map(|candidate| {
            (
                candidate.score.predictive_gain - 0.071 * candidate.predicate.arity() as f64 + 0.1,
                canonical_predicate_key(&candidate.predicate),
            )
        })
        .collect::<Vec<_>>();
    historical.sort_by(|left, right| right.0.total_cmp(&left.0));
    let ranked = top_k_with_config(std::mem::take(&mut candidates), 3, &config);
    assert_eq!(
        ranked
            .iter()
            .map(|candidate| canonical_predicate_key(&candidate.predicate))
            .collect::<Vec<_>>(),
        historical
            .into_iter()
            .map(|(_, key)| key)
            .collect::<Vec<_>>()
    );
}

#[test]
fn incompatible_and_zero_child_candidates_are_rejected_before_scoring() {
    let rows = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let dataset = Dataset::new(
        ColumnMajorMatrix::from_rows(&rows).unwrap(),
        vec![0, 0, 1, 1],
    )
    .unwrap();
    let context = TrainingContext::new(Arc::new(dataset));
    let mut node = context.root_view();
    node.theory_state = PathTheoryState::Horn;
    let incompatible = Predicate::AntiHornClause(vec![literal(0), literal(1)]);
    let before = context.diagnostics();
    assert!(context
        .score_if_admissible(&node, incompatible, &SplitScoreConfig::default())
        .unwrap()
        .is_none());
    assert_eq!(context.diagnostics(), before);

    let zero_child = Predicate::Unary(Literal {
        atom: ThresholdAtom {
            feature: 0,
            threshold_id: 0,
            threshold: -1.0,
            op: ThresholdOp::GreaterEqual,
        },
        positive: true,
    });
    assert!(context
        .score_if_admissible(&node, zero_child, &SplitScoreConfig::default())
        .unwrap()
        .is_none());
}
