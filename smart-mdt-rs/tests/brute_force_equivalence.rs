use smart_mdt_rs::{
    data::{predicate_mask, ColumnMajorMatrix},
    logic::*,
};
#[test]
fn bitset_mask_matches_naive() {
    let x = ColumnMajorMatrix::from_rows(&[vec![0.0], vec![2.0], vec![4.0]]).unwrap();
    let l = Literal {
        atom: ThresholdAtom {
            feature: 0,
            threshold_id: 0,
            threshold: 3.0,
            op: ThresholdOp::LessThan,
        },
        positive: true,
    };
    let p = Predicate::Unary(l);
    let m = predicate_mask(&x, &p);
    for i in 0..x.rows() {
        assert_eq!(m.get(i), p.eval(&x, i));
    }
}
