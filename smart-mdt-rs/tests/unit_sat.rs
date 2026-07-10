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
