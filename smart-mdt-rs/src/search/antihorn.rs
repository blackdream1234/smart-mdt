use crate::{data::Dataset, search::SplitCandidate};
/// Generates AntiHorn OR clauses of arity 2 from top unary literals.
pub fn generate_antihorn(data: &Dataset, min_leaf: usize, beam: usize) -> Vec<SplitCandidate> {
    super::horn::combine(data, min_leaf, beam, false)
}
