use smart_mdt_rs::{
    data::{predicate_mask, ColumnMajorMatrix, Dataset},
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    search::{
        antihorn::generate_antihorn, horn::generate_horn, square2cnf::generate_square2cnf,
        unary::generate_unary,
    },
};
use std::time::Instant;

fn main() {
    let rows: Vec<Vec<f64>> = (0..512)
        .map(|i| (0..8).map(|j| ((i >> (j % 8)) & 1) as f64).collect())
        .collect();
    let labels: Vec<u32> = rows
        .iter()
        .map(|r| ((r[0] == 1.0) || (r[1] == 1.0)) as u32)
        .collect();
    let ds = Dataset::new(
        ColumnMajorMatrix::from_rows(&rows).expect("valid bench matrix"),
        labels,
    )
    .expect("valid bench dataset");
    let lit = Literal {
        atom: ThresholdAtom {
            feature: 0,
            threshold_id: 0,
            threshold: 0.5,
            op: ThresholdOp::GreaterEqual,
        },
        positive: true,
    };
    let pred = Predicate::Unary(lit);

    let t = Instant::now();
    let mask = predicate_mask(&ds.features, &pred);
    println!(
        "safe bitset-like mask evaluation: {:?} count={}",
        t.elapsed(),
        mask.count_ones()
    );

    let t = Instant::now();
    let naive = (0..ds.features.rows())
        .filter(|&i| pred.eval(&ds.features, i))
        .count();
    println!("naive row evaluation: {:?} count={naive}", t.elapsed());

    let t = Instant::now();
    let n = generate_unary(&ds, 1).len();
    println!("candidate search unary: {:?} candidates={n}", t.elapsed());
    let t = Instant::now();
    println!(
        "candidate search horn: {:?} candidates={}",
        t.elapsed(),
        generate_horn(&ds, 1, 16).len()
    );
    let t = Instant::now();
    println!(
        "candidate search antihorn: {:?} candidates={}",
        t.elapsed(),
        generate_antihorn(&ds, 1, 16).len()
    );
    let t = Instant::now();
    println!(
        "candidate search square2cnf: {:?} candidates={}",
        t.elapsed(),
        generate_square2cnf(&ds, 1, 16).len()
    );
}
