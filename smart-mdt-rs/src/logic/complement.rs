use super::{Literal, Predicate};
/// CNF representation of a predicate complement as clauses of literals.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplementCnf {
    pub clauses: Vec<Vec<Literal>>,
}
/// Builds exact complement encodings used by the explanation engine.
pub fn complement_cnf(p: &Predicate) -> ComplementCnf {
    match p {
        Predicate::Unary(l) => ComplementCnf {
            clauses: vec![vec![l.negated()]],
        },
        Predicate::HornClause(ls) | Predicate::AntiHornClause(ls) => ComplementCnf {
            clauses: ls.iter().map(|l| vec![l.negated()]).collect(),
        },
        Predicate::Square2Cnf { a, b, c, d } => ComplementCnf {
            clauses: vec![
                vec![a.negated(), c.negated()],
                vec![a.negated(), d.negated()],
                vec![b.negated(), c.negated()],
                vec![b.negated(), d.negated()],
            ],
        },
        // The complement of an affine (GF(2)) predicate is another affine equation
        // with the flipped right-hand side, which is not a bounded CNF for arity >= 3.
        // Path blocking for the affine family uses the GF(2) backend, so no CNF is emitted.
        Predicate::Affine { .. } | Predicate::EmpiricalAffine { .. } => {
            ComplementCnf { clauses: vec![] }
        }
    }
}
