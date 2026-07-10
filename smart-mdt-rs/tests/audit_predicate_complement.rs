use smart_mdt_rs::{
    data::{predicate_mask, ColumnMajorMatrix},
    logic::*,
};

fn lit(feature: u32, positive: bool) -> Literal {
    Literal {
        atom: ThresholdAtom {
            feature,
            threshold_id: 0,
            threshold: 0.5,
            op: ThresholdOp::GreaterEqual,
        },
        positive,
    }
}

fn cnf_eval(cnf: &ComplementCnf, x: &ColumnMajorMatrix, row: usize) -> bool {
    cnf.clauses
        .iter()
        .all(|cl| cl.iter().any(|l| l.eval_value(x.get(row, l.atom.feature))))
}

#[test]
fn predicate_masks_match_naive_for_polarities_and_families() {
    let rows: Vec<Vec<f64>> = (0..8)
        .map(|m| (0..3).map(|j| ((m >> j) & 1) as f64).collect())
        .collect();
    let x = ColumnMajorMatrix::from_rows(&rows).unwrap();
    let preds = vec![
        Predicate::Unary(lit(0, true)),
        Predicate::Unary(lit(0, false)),
        Predicate::HornClause(vec![lit(0, false), lit(1, false), lit(2, true)]),
        Predicate::AntiHornClause(vec![lit(0, true), lit(1, true), lit(2, false)]),
        Predicate::Square2Cnf {
            a: lit(0, true),
            b: lit(1, false),
            c: lit(1, true),
            d: lit(2, false),
        },
        Predicate::EmpiricalAffine {
            literals: vec![lit(0, true), lit(1, true)],
            parity: true,
        },
    ];
    for p in preds {
        let m = predicate_mask(&x, &p);
        for i in 0..x.rows() {
            assert_eq!(m.get(i), p.eval(&x, i));
        }
    }
}

#[test]
fn complement_cnf_is_exact_for_certified_predicates_on_boolean_domain() {
    let rows: Vec<Vec<f64>> = (0..16)
        .map(|m| (0..4).map(|j| ((m >> j) & 1) as f64).collect())
        .collect();
    let x = ColumnMajorMatrix::from_rows(&rows).unwrap();
    let preds = vec![
        Predicate::Unary(lit(0, true)),
        Predicate::HornClause(vec![lit(0, false), lit(1, true)]),
        Predicate::AntiHornClause(vec![lit(0, true), lit(1, false)]),
        Predicate::Square2Cnf {
            a: lit(0, true),
            b: lit(1, true),
            c: lit(2, false),
            d: lit(3, false),
        },
    ];
    for p in preds {
        let c = complement_cnf(&p);
        if matches!(p, Predicate::Square2Cnf { .. }) {
            assert_eq!(c.clauses.len(), 4);
            assert_eq!(
                c.clauses.iter().map(Vec::len).collect::<Vec<_>>(),
                vec![2, 2, 2, 2]
            );
        }
        for i in 0..x.rows() {
            assert_ne!(p.eval(&x, i), cnf_eval(&c, &x, i));
        }
    }
}

#[test]
fn affine_is_empirical_not_certified() {
    let p = Predicate::EmpiricalAffine {
        literals: vec![lit(0, true)],
        parity: true,
    };
    let meta = p.certificate(true);
    assert!(!meta.theorem_certified);
    assert_eq!(meta.backend, Backend::Affine);
}
