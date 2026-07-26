use super::{literals_are_in_range, Cnf};
/// Checks Horn-SAT by least-model propagation. Clauses must have at most one positive literal.
pub fn horn_sat(num_vars: usize, cnf: &Cnf) -> bool {
    if !literals_are_in_range(num_vars, cnf) {
        return false;
    }
    let mut val = vec![false; num_vars + 1];
    loop {
        let mut changed = false;
        for cl in cnf {
            let mut normalized = cl.clone();
            normalized.sort_unstable();
            normalized.dedup();
            if normalized
                .iter()
                .any(|literal| normalized.binary_search(&-*literal).is_ok())
            {
                continue;
            }
            let pos: Vec<_> = normalized.iter().copied().filter(|l| *l > 0).collect();
            if pos.len() > 1 {
                return false;
            }
            let neg_all_true = normalized
                .iter()
                .filter(|l| **l < 0)
                .all(|l| val[l.unsigned_abs() as usize]);
            if pos.is_empty() && neg_all_true {
                return false;
            }
            if pos.len() == 1 && neg_all_true {
                let p = pos[0] as usize;
                if !val[p] {
                    val[p] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            return true;
        }
    }
}
