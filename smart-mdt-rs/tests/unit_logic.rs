use smart_mdt_rs::logic::*;
#[test]
fn literal_and_square_complement() {
    let a = ThresholdAtom {
        feature: 0,
        threshold_id: 0,
        threshold: 1.0,
        op: ThresholdOp::LessThan,
    };
    let l = Literal {
        atom: a,
        positive: true,
    };
    assert!(l.eval_value(0.0));
    assert!(!l.negated().eval_value(0.0));
    let p = Predicate::Square2Cnf {
        a: l,
        b: l.negated(),
        c: l,
        d: l.negated(),
    };
    assert_eq!(complement_cnf(&p).clauses.len(), 4);
}
