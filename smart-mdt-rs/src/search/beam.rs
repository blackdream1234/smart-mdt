use super::{top_k, SplitCandidate};
/// Transparent beam cap for heuristic training search; certificates are unaffected.
pub fn beam_cap(xs: Vec<SplitCandidate>, width: usize) -> Vec<SplitCandidate> {
    top_k(xs, width)
}
