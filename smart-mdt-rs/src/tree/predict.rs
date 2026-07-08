use super::TreeNode;
use crate::{data::ColumnMajorMatrix, ClassId};
/// Predicts a row.
pub fn predict_row(tree: &TreeNode, x: &ColumnMajorMatrix, row: usize) -> ClassId {
    match tree {
        TreeNode::Leaf { class, .. } => *class,
        TreeNode::Internal {
            predicate,
            left,
            right,
            ..
        } => {
            if predicate.eval(x, row) {
                predict_row(left, x, row)
            } else {
                predict_row(right, x, row)
            }
        }
    }
}
/// Predicts all rows.
pub fn predict_all(tree: &TreeNode, x: &ColumnMajorMatrix) -> Vec<ClassId> {
    (0..x.rows()).map(|i| predict_row(tree, x, i)).collect()
}
