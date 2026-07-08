use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    search::{
        antihorn::generate_antihorn, horn::generate_horn, square2cnf::generate_square2cnf,
        unary::generate_unary,
    },
};
#[test]
fn candidate_generation() {
    let ds = Dataset::new(
        ColumnMajorMatrix::from_rows(&[
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 1.0],
            vec![3.0, 1.0],
        ])
        .unwrap(),
        vec![0, 0, 1, 1],
    )
    .unwrap();
    assert!(!generate_unary(&ds, 1).is_empty());
    let _ = generate_horn(&ds, 1, 8);
    let _ = generate_antihorn(&ds, 1, 8);
    let _ = generate_square2cnf(&ds, 1, 8);
}
