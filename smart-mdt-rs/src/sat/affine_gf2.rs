//! Certified GF(2) affine backend.
//!
//! A path in an affine (XOR) tree is a conjunction of Boolean linear equations
//! `x_i1 ⊕ x_i2 ⊕ ... ⊕ x_ik = b` over GF(2). The conjunction is satisfiable iff
//! the linear system is consistent, which Gaussian elimination decides in
//! polynomial time using only integer/bit operations (no floating point).
//!
//! Variables are compact 0-based indices; bit `i` of a coefficient mask selects
//! variable `i`. Up to 128 distinct variables per system are supported, which is
//! sufficient for path systems after remapping the path's features to compact
//! indices.

/// One GF(2) linear equation: the XOR of the variables selected by `mask`
/// equals `rhs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gf2Equation {
    /// Coefficient mask; bit `i` set means variable `i` participates.
    pub mask: u128,
    /// Right-hand side of the equation.
    pub rhs: bool,
}

impl Gf2Equation {
    /// Builds an equation from an explicit coefficient mask.
    pub fn new(mask: u128, rhs: bool) -> Self {
        Self { mask, rhs }
    }

    /// Builds an equation from a list of variable indices. Repeated variables
    /// cancel modulo 2 because each occurrence is XORed into the coefficient mask.
    pub fn from_vars(vars: &[usize], rhs: bool) -> Self {
        let mut mask = 0u128;
        for &v in vars {
            debug_assert!(v < 128, "GF(2) variable index out of range");
            mask ^= 1u128 << v;
        }
        Self { mask, rhs }
    }
}

/// Solves a GF(2) linear system by Gaussian elimination using only integer/bit
/// operations. Returns `true` iff the system is satisfiable (consistent), i.e.
/// no row reduces to the contradiction `0 = 1`.
pub fn gf2_system_satisfiable(equations: &[Gf2Equation]) -> bool {
    let mut rows: Vec<(u128, bool)> = equations.iter().map(|e| (e.mask, e.rhs)).collect();
    let mut pivot = 0usize;
    for bit in 0..u128::BITS {
        let Some(sel) = (pivot..rows.len()).find(|&r| (rows[r].0 >> bit) & 1 == 1) else {
            continue;
        };
        rows.swap(pivot, sel);
        let (pmask, prhs) = rows[pivot];
        for (r, row) in rows.iter_mut().enumerate() {
            if r != pivot && (row.0 >> bit) & 1 == 1 {
                row.0 ^= pmask;
                row.1 ^= prhs;
            }
        }
        pivot += 1;
        if pivot == rows.len() {
            break;
        }
    }
    // Inconsistent iff a row has no variables left but a non-zero right-hand side.
    !rows.iter().any(|&(mask, rhs)| mask == 0 && rhs)
}

/// Adds assumptions `x_i = v_i` (as used by AXp checking) as single-variable
/// equations and checks satisfiability of the combined system.
pub fn gf2_satisfiable_with_assumptions(
    equations: &[Gf2Equation],
    assumptions: &[(usize, bool)],
) -> bool {
    let mut system: Vec<Gf2Equation> = equations.to_vec();
    for &(var, val) in assumptions {
        debug_assert!(var < 128, "GF(2) assumption variable index out of range");
        system.push(Gf2Equation::new(1u128 << var, val));
    }
    gf2_system_satisfiable(&system)
}

/// Brute-force reference for tests: enumerates all `2^n` assignments over `n`
/// variables and returns `true` iff some assignment satisfies every equation.
pub fn gf2_brute_force_satisfiable(n: usize, equations: &[Gf2Equation]) -> bool {
    debug_assert!(n <= 20, "brute-force GF(2) only for tiny systems");
    (0..(1u128 << n)).any(|assign| {
        equations
            .iter()
            .all(|e| ((e.mask & assign).count_ones() & 1 == 1) == e.rhs)
    })
}
