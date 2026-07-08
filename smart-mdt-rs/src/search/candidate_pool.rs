use super::CandidateScore;
use crate::logic::Predicate;
/// A split candidate with certificate metadata and exact mask statistics.
#[derive(Clone, Debug)]
pub struct SplitCandidate {
    pub predicate: Predicate,
    pub score: CandidateScore,
    pub left_count: usize,
    pub right_count: usize,
}
/// Keeps best candidates by final score.
pub fn top_k(mut xs: Vec<SplitCandidate>, k: usize) -> Vec<SplitCandidate> {
    xs.sort_by(|a, b| b.score.final_score.total_cmp(&a.score.final_score));
    xs.truncate(k);
    xs
}
