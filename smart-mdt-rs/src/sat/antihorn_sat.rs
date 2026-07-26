use super::{horn_sat::horn_sat, literals_are_in_range, Cnf};
/// Checks AntiHorn-SAT by polarity flip to Horn-SAT.
pub fn antihorn_sat(num_vars: usize, cnf: &Cnf) -> bool {
    if !literals_are_in_range(num_vars, cnf) {
        return false;
    }
    let flipped: Cnf = cnf
        .iter()
        .map(|c| c.iter().map(|l| -*l).collect())
        .collect();
    horn_sat(num_vars, &flipped)
}
