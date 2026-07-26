use super::{canonical_predicate_key, family_order, CandidateScore, SplitScoreConfig};
use crate::logic::Predicate;
use std::cmp::Ordering;
/// A split candidate with certificate metadata and exact mask statistics.
#[derive(Clone, Debug)]
pub struct SplitCandidate {
    pub predicate: Predicate,
    pub score: CandidateScore,
    pub left_count: usize,
    pub right_count: usize,
}
/// Keeps best candidates by final score.
pub fn top_k(xs: Vec<SplitCandidate>, k: usize) -> Vec<SplitCandidate> {
    top_k_with_config(xs, k, &SplitScoreConfig::default())
}

/// Keeps best candidates using the complete deterministic CALS tie-break order.
pub fn top_k_with_config(
    mut xs: Vec<SplitCandidate>,
    k: usize,
    config: &SplitScoreConfig,
) -> Vec<SplitCandidate> {
    xs.sort_by(|a, b| compare_candidates(a, b, config));
    xs.truncate(k);
    xs
}

pub fn compare_candidates(
    left: &SplitCandidate,
    right: &SplitCandidate,
    config: &SplitScoreConfig,
) -> Ordering {
    descending_float(
        left.score.final_score,
        right.score.final_score,
        config.tie_epsilon,
    )
    .then_with(|| {
        descending_float(
            left.score.predictive_gain,
            right.score.predictive_gain,
            config.tie_epsilon,
        )
    })
    .then_with(|| left.predicate.arity().cmp(&right.predicate.arity()))
    .then_with(|| {
        family_order(left.predicate.language()).cmp(&family_order(right.predicate.language()))
    })
    .then_with(|| {
        canonical_predicate_key(&left.predicate).cmp(&canonical_predicate_key(&right.predicate))
    })
}

fn descending_float(left: f64, right: f64, epsilon: f64) -> Ordering {
    if (left - right).abs() <= epsilon {
        Ordering::Equal
    } else {
        right.total_cmp(&left)
    }
}
