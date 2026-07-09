use super::Cnf;
use crate::logic::{complement_cnf, Predicate};
/// Minimal path-CNF placeholder for certified paths over Booleanized atoms.
pub fn predicate_complement_clause_count(p: &Predicate) -> usize {
    complement_cnf(p).clauses.len()
}
/// Empty satisfiable CNF.
pub fn empty_cnf() -> Cnf {
    Vec::new()
}
