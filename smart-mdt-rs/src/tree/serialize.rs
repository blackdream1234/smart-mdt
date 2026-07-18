use super::TreeNode;
use crate::{
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    Result, SmartMdtError,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(tag = "node", rename_all = "snake_case")]
enum JsonTreeNode {
    Leaf {
        class: u32,
        samples: usize,
    },
    Internal {
        predicate: JsonPredicate,
        left: Box<JsonTreeNode>,
        right: Box<JsonTreeNode>,
        majority_class: u32,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum JsonPredicate {
    Unary {
        literal: JsonLiteral,
    },
    HornClause {
        literals: Vec<JsonLiteral>,
    },
    AntiHornClause {
        literals: Vec<JsonLiteral>,
    },
    Square2Cnf {
        a: JsonLiteral,
        b: JsonLiteral,
        c: JsonLiteral,
        d: JsonLiteral,
    },
    Affine {
        literals: Vec<JsonLiteral>,
        rhs: bool,
    },
    EmpiricalAffine {
        literals: Vec<JsonLiteral>,
        parity: bool,
    },
}

#[derive(Clone, Copy, Serialize, Deserialize)]
struct JsonLiteral {
    feature: u32,
    threshold_id: u32,
    threshold: f64,
    op: JsonThresholdOp,
    positive: bool,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum JsonThresholdOp {
    LessThan,
    GreaterEqual,
}

impl From<&TreeNode> for JsonTreeNode {
    fn from(tree: &TreeNode) -> Self {
        match tree {
            TreeNode::Leaf { class, samples } => Self::Leaf {
                class: *class,
                samples: *samples,
            },
            TreeNode::Internal {
                predicate,
                left,
                right,
                majority_class,
            } => Self::Internal {
                predicate: predicate.into(),
                left: Box::new(left.as_ref().into()),
                right: Box::new(right.as_ref().into()),
                majority_class: *majority_class,
            },
        }
    }
}

impl From<JsonTreeNode> for TreeNode {
    fn from(tree: JsonTreeNode) -> Self {
        match tree {
            JsonTreeNode::Leaf { class, samples } => Self::Leaf { class, samples },
            JsonTreeNode::Internal {
                predicate,
                left,
                right,
                majority_class,
            } => Self::Internal {
                predicate: predicate.into(),
                left: Box::new((*left).into()),
                right: Box::new((*right).into()),
                majority_class,
            },
        }
    }
}

impl From<&Predicate> for JsonPredicate {
    fn from(predicate: &Predicate) -> Self {
        match predicate {
            Predicate::Unary(literal) => Self::Unary {
                literal: (*literal).into(),
            },
            Predicate::HornClause(literals) => Self::HornClause {
                literals: literals.iter().copied().map(Into::into).collect(),
            },
            Predicate::AntiHornClause(literals) => Self::AntiHornClause {
                literals: literals.iter().copied().map(Into::into).collect(),
            },
            Predicate::Square2Cnf { a, b, c, d } => Self::Square2Cnf {
                a: (*a).into(),
                b: (*b).into(),
                c: (*c).into(),
                d: (*d).into(),
            },
            Predicate::Affine { literals, rhs } => Self::Affine {
                literals: literals.iter().copied().map(Into::into).collect(),
                rhs: *rhs,
            },
            Predicate::EmpiricalAffine { literals, parity } => Self::EmpiricalAffine {
                literals: literals.iter().copied().map(Into::into).collect(),
                parity: *parity,
            },
        }
    }
}

impl From<JsonPredicate> for Predicate {
    fn from(predicate: JsonPredicate) -> Self {
        match predicate {
            JsonPredicate::Unary { literal } => Self::Unary(literal.into()),
            JsonPredicate::HornClause { literals } => {
                Self::HornClause(literals.into_iter().map(Into::into).collect())
            }
            JsonPredicate::AntiHornClause { literals } => {
                Self::AntiHornClause(literals.into_iter().map(Into::into).collect())
            }
            JsonPredicate::Square2Cnf { a, b, c, d } => Self::Square2Cnf {
                a: a.into(),
                b: b.into(),
                c: c.into(),
                d: d.into(),
            },
            JsonPredicate::Affine { literals, rhs } => Self::Affine {
                literals: literals.into_iter().map(Into::into).collect(),
                rhs,
            },
            JsonPredicate::EmpiricalAffine { literals, parity } => Self::EmpiricalAffine {
                literals: literals.into_iter().map(Into::into).collect(),
                parity,
            },
        }
    }
}

impl From<Literal> for JsonLiteral {
    fn from(literal: Literal) -> Self {
        Self {
            feature: literal.atom.feature,
            threshold_id: literal.atom.threshold_id,
            threshold: literal.atom.threshold,
            op: match literal.atom.op {
                ThresholdOp::LessThan => JsonThresholdOp::LessThan,
                ThresholdOp::GreaterEqual => JsonThresholdOp::GreaterEqual,
            },
            positive: literal.positive,
        }
    }
}

impl From<JsonLiteral> for Literal {
    fn from(literal: JsonLiteral) -> Self {
        Self {
            atom: ThresholdAtom {
                feature: literal.feature,
                threshold_id: literal.threshold_id,
                threshold: literal.threshold,
                op: match literal.op {
                    JsonThresholdOp::LessThan => ThresholdOp::LessThan,
                    JsonThresholdOp::GreaterEqual => ThresholdOp::GreaterEqual,
                },
            },
            positive: literal.positive,
        }
    }
}

/// Serializes a complete tree as deterministic JSON.
pub fn to_json(tree: &TreeNode) -> Result<String> {
    serde_json::to_string_pretty(&JsonTreeNode::from(tree))
        .map_err(|error| SmartMdtError::Json(error.to_string()))
}

/// Deserializes a tree produced by [`to_json`].
pub fn from_json(json: &str) -> Result<TreeNode> {
    serde_json::from_str::<JsonTreeNode>(json)
        .map(Into::into)
        .map_err(|error| SmartMdtError::Json(error.to_string()))
}
