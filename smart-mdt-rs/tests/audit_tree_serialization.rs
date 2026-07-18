use smart_mdt_rs::{
    data::ColumnMajorMatrix,
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    tree::{
        predict_all,
        serialize::{from_json, to_json},
        TreeNode,
    },
};

fn literal(feature: u32, positive: bool) -> Literal {
    Literal {
        atom: ThresholdAtom {
            feature,
            threshold_id: feature + 7,
            threshold: 0.5,
            op: ThresholdOp::GreaterEqual,
        },
        positive,
    }
}

fn leaf(class: u32) -> TreeNode {
    TreeNode::Leaf { class, samples: 4 }
}

#[test]
fn every_predicate_family_round_trips_as_json_without_prediction_changes() {
    let predicates = [
        Predicate::Unary(literal(0, true)),
        Predicate::HornClause(vec![literal(0, false), literal(1, true)]),
        Predicate::AntiHornClause(vec![literal(0, true), literal(1, false)]),
        Predicate::Square2Cnf {
            a: literal(0, true),
            b: literal(1, false),
            c: literal(1, true),
            d: literal(2, false),
        },
        Predicate::Affine {
            literals: vec![literal(0, true), literal(1, true), literal(2, true)],
            rhs: true,
        },
        Predicate::EmpiricalAffine {
            literals: vec![literal(0, true), literal(2, true)],
            parity: false,
        },
    ];
    let rows = (0..8)
        .map(|mask| {
            (0..3)
                .map(|feature| ((mask >> feature) & 1) as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let features = ColumnMajorMatrix::from_rows(&rows).unwrap();

    for predicate in predicates {
        let tree = TreeNode::Internal {
            predicate,
            left: Box::new(leaf(1)),
            right: Box::new(leaf(0)),
            majority_class: 0,
        };
        let json = to_json(&tree).unwrap();
        let parsed_value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            parsed_value.is_object(),
            "serializer emitted non-JSON: {json}"
        );
        let restored = from_json(&json).unwrap();
        assert_eq!(restored, tree);
        assert_eq!(
            predict_all(&restored, &features),
            predict_all(&tree, &features)
        );
        assert_eq!(to_json(&restored).unwrap(), json);
    }
}

#[test]
fn malformed_json_returns_a_recoverable_error() {
    assert!(from_json("{not valid json").is_err());
}
