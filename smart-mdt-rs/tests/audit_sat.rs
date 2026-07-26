use smart_mdt_rs::sat::*;

fn next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

fn var(seed: &mut u64, n: usize) -> i32 {
    1 + (next(seed) as usize % n) as i32
}

#[test]
fn random_horn_antihorn_and_two_sat_match_bruteforce() {
    for n in 1..=10 {
        let mut seed = n as u64 * 17 + 3;
        for _ in 0..100 {
            let mut horn = Vec::new();
            let mut anti = Vec::new();
            let mut two = Vec::new();
            for _ in 0..8 {
                let positives = (next(&mut seed) % 2) as usize;
                let negatives = (next(&mut seed) % 3) as usize;
                let mut h = Vec::new();
                for _ in 0..negatives {
                    h.push(-var(&mut seed, n));
                }
                for _ in 0..positives {
                    h.push(var(&mut seed, n));
                }
                horn.push(h);

                let pos = (next(&mut seed) % 3) as usize;
                let neg = (next(&mut seed) % 2) as usize;
                let mut a = Vec::new();
                for _ in 0..pos {
                    a.push(var(&mut seed, n));
                }
                for _ in 0..neg {
                    a.push(-var(&mut seed, n));
                }
                anti.push(a);

                let l1 = if next(&mut seed) & 1 == 1 {
                    var(&mut seed, n)
                } else {
                    -var(&mut seed, n)
                };
                let l2 = if next(&mut seed) & 1 == 1 {
                    var(&mut seed, n)
                } else {
                    -var(&mut seed, n)
                };
                two.push(vec![l1, l2]);
            }
            assert_eq!(
                horn_sat(n, &horn),
                brute_force_sat(n, &horn),
                "horn n={n} {horn:?}"
            );
            assert_eq!(
                antihorn_sat(n, &anti),
                brute_force_sat(n, &anti),
                "anti n={n} {anti:?}"
            );
            assert_eq!(
                two_sat(n, &two),
                brute_force_sat(n, &two),
                "2sat n={n} {two:?}"
            );
        }
    }
}

#[test]
fn sat_edge_cases() {
    assert!(horn_sat(3, &vec![]));
    assert!(antihorn_sat(3, &vec![]));
    assert!(two_sat(3, &vec![]));
    assert!(!horn_sat(3, &vec![vec![]]));
    assert!(!antihorn_sat(3, &vec![vec![]]));
    assert!(!two_sat(3, &vec![vec![]]));
    assert!(horn_sat(2, &vec![vec![-1, 2]]));
    assert!(two_sat(2, &vec![vec![1], vec![2]]));
}

#[test]
fn gf2_gaussian_matches_bruteforce_on_random_systems() {
    for n in 1..=8usize {
        let mut seed = n as u64 * 2654435761 + 11;
        for _ in 0..300 {
            let count = (next(&mut seed) % 6) as usize;
            let eqs: Vec<Gf2Equation> = (0..count)
                .map(|_| {
                    let mask = (next(&mut seed) as u128) & ((1u128 << n) - 1);
                    let rhs = next(&mut seed) & 1 == 1;
                    Gf2Equation::new(mask, rhs)
                })
                .collect();
            assert_eq!(
                gf2_system_satisfiable(&eqs),
                gf2_brute_force_satisfiable(n, &eqs),
                "gf2 n={n} {eqs:?}"
            );
        }
    }
}

#[test]
fn gf2_gaussian_matches_bruteforce_with_random_assumptions() {
    for n in 2..=7usize {
        let mut seed = n as u64 * 40503 + 7;
        for _ in 0..300 {
            let count = (next(&mut seed) % 5) as usize;
            let eqs: Vec<Gf2Equation> = (0..count)
                .map(|_| {
                    let mask = (next(&mut seed) as u128) & ((1u128 << n) - 1);
                    Gf2Equation::new(mask, next(&mut seed) & 1 == 1)
                })
                .collect();
            let n_assume = (next(&mut seed) % (n as u64 + 1)) as usize;
            let mut assumptions = Vec::new();
            for _ in 0..n_assume {
                let var = (next(&mut seed) as usize) % n;
                assumptions.push((var, next(&mut seed) & 1 == 1));
            }
            // Fold assumptions into the system as single-variable equations and
            // compare against brute force over the combined system.
            let mut combined = eqs.clone();
            for &(v, val) in &assumptions {
                combined.push(Gf2Equation::new(1u128 << v, val));
            }
            assert_eq!(
                gf2_satisfiable_with_assumptions(&eqs, &assumptions),
                gf2_brute_force_satisfiable(n, &combined),
                "gf2+assume n={n} {eqs:?} assume={assumptions:?}"
            );
        }
    }
}
