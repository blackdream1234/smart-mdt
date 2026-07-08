use super::{BitSet, ColumnMajorMatrix};
use crate::logic::Predicate;
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
