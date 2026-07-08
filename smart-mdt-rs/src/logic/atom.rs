use crate::{FeatureId, ThresholdId};
/// Threshold comparison direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ThresholdOp {
    LessThan,
    GreaterEqual,
}
/// A univariate threshold atom over a numeric feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThresholdAtom {
    pub feature: FeatureId,
    pub threshold_id: ThresholdId,
    pub threshold: f64,
    pub op: ThresholdOp,
}
impl ThresholdAtom {
    /// Evaluates the atom.
    pub fn eval_value(&self, v: f64) -> bool {
        match self.op {
            ThresholdOp::LessThan => v < self.threshold,
            ThresholdOp::GreaterEqual => v >= self.threshold,
        }
    }
    /// Complement atom.
    pub fn complement(&self) -> Self {
        Self {
            op: match self.op {
                ThresholdOp::LessThan => ThresholdOp::GreaterEqual,
                ThresholdOp::GreaterEqual => ThresholdOp::LessThan,
            },
            ..*self
        }
    }
}
