use super::{horn_sat::horn_sat, Cnf};
/// Checks AntiHorn-SAT by polarity flip to Horn-SAT.
pub fn antihorn_sat(num_vars: usize, cnf: &Cnf) -> bool {
    let flipped: Cnf = cnf
        .iter()
        .map(|c| c.iter().map(|l| -*l).collect())
        .collect();
    horn_sat(num_vars, &flipped)
}
