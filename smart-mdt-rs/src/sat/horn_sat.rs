use super::Cnf;
/// Checks Horn-SAT by least-model propagation. Clauses must have at most one positive literal.
pub fn horn_sat(num_vars: usize, cnf: &Cnf) -> bool {
    let mut val = vec![false; num_vars + 1];
    loop {
        let mut changed = false;
        for cl in cnf {
            let pos: Vec<_> = cl.iter().copied().filter(|l| *l > 0).collect();
            if pos.len() > 1 {
                return false;
            }
            let neg_all_true = cl
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
