use smart_mdt_rs::sat::{antihorn_sat, horn_sat, two_sat, Cnf};
use std::time::Instant;

fn main() {
    let horn: Cnf = (1..=64)
        .map(|i| if i == 1 { vec![1] } else { vec![-(i - 1), i] })
        .collect();
    let anti: Cnf = horn
        .iter()
        .map(|c| c.iter().map(|l| -*l).collect())
        .collect();
    let two: Cnf = (1..64).map(|i| vec![i, i + 1]).collect();
    for (name, f, cnf) in [
        ("horn", horn_sat as fn(usize, &Cnf) -> bool, &horn),
        ("antihorn", antihorn_sat as fn(usize, &Cnf) -> bool, &anti),
        ("two_sat", two_sat as fn(usize, &Cnf) -> bool, &two),
    ] {
        let t = Instant::now();
        let mut ok = false;
        for _ in 0..1000 {
            ok = f(64, cnf);
        }
        println!("SAT backend {name}: {:?} last={ok}", t.elapsed());
    }
}
