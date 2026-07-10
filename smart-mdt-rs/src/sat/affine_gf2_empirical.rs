//! Empirical affine GF(2) backend. It is not theorem-certified in this crate.
/// Solves a tiny GF(2) linear system represented as rows `(mask, rhs)`.
pub fn gf2_satisfiable(vars: usize, rows: &[(u128, bool)]) -> bool {
    gf2_satisfiable_with_assumptions(vars, rows, &[])
}

/// Solves a GF(2) system with single-variable assumptions `(var, value)`.
///
/// Variables are zero-based and the implementation is intentionally bounded to
/// at most 20 variables so it remains an auditable empirical helper.
pub fn gf2_satisfiable_with_assumptions(
    vars: usize,
    rows: &[(u128, bool)],
    assumptions: &[(usize, bool)],
) -> bool {
    if vars > 20 || assumptions.iter().any(|(v, _)| *v >= vars) {
        return false;
    }
    let mut system: Vec<(u128, bool)> = rows.to_vec();
    for &(var, value) in assumptions {
        system.push((1u128 << var, value));
    }
    let mut rank = 0usize;
    for col in 0..vars {
        let Some(pivot) = (rank..system.len()).find(|&r| ((system[r].0 >> col) & 1) == 1) else {
            continue;
        };
        system.swap(rank, pivot);
        for r in 0..system.len() {
            if r != rank && ((system[r].0 >> col) & 1) == 1 {
                system[r].0 ^= system[rank].0;
                system[r].1 ^= system[rank].1;
            }
        }
        rank += 1;
    }
    system.iter().all(|(mask, rhs)| *mask != 0 || !*rhs)
}
