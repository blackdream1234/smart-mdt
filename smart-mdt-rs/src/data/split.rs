use super::{BitSet, ColumnMajorMatrix};
use crate::{logic::Predicate, FeatureId};
/// Returns true iff every value of the feature column lies in `{0, 1}`.
pub fn is_boolean_column(x: &ColumnMajorMatrix, feature: FeatureId) -> bool {
    x.column(feature).iter().all(|&v| v == 0.0 || v == 1.0)
}
/// Returns true iff every feature in the predicate scope is Boolean over the domain.
/// This is the guard that decides whether an affine predicate may be theorem-certified.
pub fn predicate_scope_is_boolean(x: &ColumnMajorMatrix, p: &Predicate) -> bool {
    p.scope_features().iter().all(|&f| is_boolean_column(x, f))
}
/// Computes a predicate membership mask.
pub fn predicate_mask(x: &ColumnMajorMatrix, p: &Predicate) -> BitSet {
    let mut b = BitSet::zeros(x.rows());
    for i in 0..x.rows() {
        b.set(i, p.eval(x, i));
    }
    b
}
/// Computes class counts under a mask.
pub fn class_counts(labels: &[u32], mask: &BitSet, classes: usize) -> Vec<usize> {
    let mut c = vec![0; classes];
    for (i, l) in labels.iter().enumerate() {
        if mask.get(i) {
            c[*l as usize] += 1;
        }
    }
    c
}
