use smart_mdt_rs::sat::*;
#[test]
fn sat_backends_match_bruteforce() {
    let horn = vec![vec![-1, 2], vec![-2]];
    assert_eq!(horn_sat(2, &horn), brute_force_sat(2, &horn));
    let anti = vec![vec![1, -2], vec![2]];
    assert_eq!(antihorn_sat(2, &anti), brute_force_sat(2, &anti));
    let two = vec![vec![1, 2], vec![-1, 2], vec![1, -2]];
    assert_eq!(two_sat(2, &two), brute_force_sat(2, &two));
}
#[test]
fn contradiction_cases() {
    assert!(!two_sat(1, &vec![vec![1], vec![-1]]));
    assert!(!horn_sat(1, &vec![vec![1], vec![-1]]));
}

#[test]
fn gf2_assumptions_are_handled() {
    // x0 ⊕ x1 = 1 is satisfiable on its own.
    let eqs = vec![Gf2Equation::from_vars(&[0, 1], true)];
    assert!(gf2_system_satisfiable(&eqs));
    // Assuming x0 = 0 forces x1 = 1: still consistent.
    assert!(gf2_satisfiable_with_assumptions(&eqs, &[(0, false)]));
    // Assuming x0 = 0 and x1 = 0 contradicts x0 ⊕ x1 = 1.
    assert!(!gf2_satisfiable_with_assumptions(
        &eqs,
        &[(0, false), (1, false)]
    ));
    // Assuming x0 = 1 and x1 = 0 satisfies it.
    assert!(gf2_satisfiable_with_assumptions(
        &eqs,
        &[(0, true), (1, false)]
    ));
}

#[test]
fn gf2_inconsistent_system_is_unsat() {
    // x0 = 0 and x0 = 1 together derive 0 = 1.
    let sys = vec![
        Gf2Equation::from_vars(&[0], false),
        Gf2Equation::from_vars(&[0], true),
    ];
    assert!(!gf2_system_satisfiable(&sys));
    // A directly contradictory empty equation 0 = 1.
    assert!(!gf2_system_satisfiable(&[Gf2Equation::new(0, true)]));
    // The trivially true empty equation 0 = 0 is satisfiable.
    assert!(gf2_system_satisfiable(&[Gf2Equation::new(0, false)]));
}

#[test]
fn gf2_duplicate_variables_cancel_modulo_two() {
    // x0 ⊕ x0 cancels to the empty left-hand side.
    assert_eq!(Gf2Equation::from_vars(&[0, 0], false).mask, 0);
    // x0 ⊕ x1 ⊕ x0 == x1.
    assert_eq!(
        Gf2Equation::from_vars(&[0, 1, 0], true).mask,
        Gf2Equation::from_vars(&[1], true).mask
    );
    // x0 ⊕ x0 = 0 is trivially satisfiable; x0 ⊕ x0 = 1 is unsatisfiable.
    assert!(gf2_system_satisfiable(&[Gf2Equation::from_vars(
        &[0, 0],
        false
    )]));
    assert!(!gf2_system_satisfiable(&[Gf2Equation::from_vars(
        &[0, 0],
        true
    )]));
}
