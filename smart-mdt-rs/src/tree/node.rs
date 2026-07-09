use crate::{logic::Predicate, ClassId};
/// Decision-tree node.
#[derive(Clone, Debug, PartialEq)]
pub enum TreeNode {
    Leaf {
        class: ClassId,
        samples: usize,
    },
    Internal {
        predicate: Predicate,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
        majority_class: ClassId,
    },
}
impl TreeNode {
    /// Counts nodes.
    pub fn nodes(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Internal { left, right, .. } => 1 + left.nodes() + right.nodes(),
        }
    }
    /// Counts leaves.
    pub fn leaves(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Internal { left, right, .. } => left.leaves() + right.leaves(),
        }
    }
    /// Max depth.
    pub fn depth(&self) -> usize {
        match self {
            Self::Leaf { .. } => 0,
            Self::Internal { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }
}
