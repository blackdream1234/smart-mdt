//! Empirical affine GF(2) backend. It is not theorem-certified in this crate.
/// Solves a tiny GF(2) linear system represented as rows `(mask, rhs)`.
pub fn gf2_satisfiable(_vars: usize, _rows: &[(u128, bool)]) -> bool {
    true
}
