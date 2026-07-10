use super::ThresholdAtom;
/// UI-literal wrapping a threshold atom; `positive=false` means logical negation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Literal {
    pub atom: ThresholdAtom,
    pub positive: bool,
}
impl Literal {
    /// Evaluates the literal.
    pub fn eval_value(&self, v: f64) -> bool {
        self.atom.eval_value(v) == self.positive
    }
    /// Logical negation.
    pub fn negated(&self) -> Self {
        Self {
            atom: self.atom,
            positive: !self.positive,
        }
    }
}
