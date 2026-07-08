//! Polynomial SAT/CSP fragments used by certified explanation backends.
pub mod affine_gf2_empirical;
pub mod antihorn_sat;
pub mod horn_sat;
pub mod path_cnf;
pub mod two_sat;
pub use antihorn_sat::*;
pub use horn_sat::*;
pub use path_cnf::*;
pub use two_sat::*;
/// Signed Boolean literal, variable ids are 1-based; negative means negation.
pub type SatLit = i32;
/// CNF clause.
pub type Clause = Vec<SatLit>;
/// CNF formula.
pub type Cnf = Vec<Clause>;
/// Brute-force SAT for tests and tiny fallbacks outside theorem claims.
pub fn brute_force_sat(n: usize, cnf: &Cnf) -> bool {
    (0..(1usize << n)).any(|m| {
        cnf.iter().all(|cl| {
            cl.iter().any(|&l| {
                let v = (m >> ((l.abs() as usize) - 1)) & 1 == 1;
                if l > 0 {
                    v
                } else {
                    !v
                }
            })
        })
    })
}
