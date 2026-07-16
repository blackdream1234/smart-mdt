use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    explain::{
        compile_verified_explanation, verified_explanation_from_json, verified_explanation_to_json,
        ExplanationAudience,
    },
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    tree::TreeNode,
};
use std::collections::BTreeSet;

const THRESHOLD: f64 = 0.123_456_789;

fn literal(feature: u32, positive: bool) -> Literal {
    Literal {
        atom: ThresholdAtom {
            feature,
            threshold_id: 0,
            threshold: THRESHOLD,
            op: ThresholdOp::GreaterEqual,
        },
        positive,
    }
}

fn dataset() -> Dataset {
    let rows = (0..16)
        .map(|mask| {
            (0..4)
                .map(|bit| ((mask >> bit) & 1) as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let labels = rows.iter().map(|row| u32::from(row[0] == 1.0)).collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

fn stump(predicate: Predicate) -> TreeNode {
    TreeNode::Internal {
        predicate,
        majority_class: 0,
        left: Box::new(TreeNode::Leaf {
            class: 1,
            samples: 8,
        }),
        right: Box::new(TreeNode::Leaf {
            class: 0,
            samples: 8,
        }),
    }
}

#[test]
fn verified_json_round_trip_preserves_thresholds_and_axp_scope() {
    let data = dataset();
    let explanation = compile_verified_explanation(
        &stump(Predicate::Unary(literal(0, true))),
        &data,
        1,
        ExplanationAudience::Technical,
    )
    .unwrap();
    let axp_features = explanation
        .sufficient_axp_evidence
        .iter()
        .map(|evidence| evidence.feature_id)
        .collect::<BTreeSet<_>>();
    assert!(explanation
        .human_readable_reasoning_steps
        .iter()
        .flat_map(|step| &step.feature_ids)
        .all(|feature| axp_features.contains(feature)));
    assert_eq!(explanation.full_path[0].thresholds[0].threshold, THRESHOLD);
    let json = verified_explanation_to_json(&explanation).unwrap();
    let restored = verified_explanation_from_json(&json).unwrap();
    assert_eq!(restored, explanation);
    assert_eq!(restored.full_path[0].thresholds[0].threshold, THRESHOLD);
    assert_eq!(verified_explanation_to_json(&restored).unwrap(), json);
}

#[test]
fn affine_parity_wording_is_explicit_and_certified() {
    let explanation = compile_verified_explanation(
        &stump(Predicate::Affine {
            literals: vec![literal(0, true), literal(1, true)],
            rhs: true,
        }),
        &dataset(),
        1,
        ExplanationAudience::Audit,
    )
    .unwrap();
    let wording = &explanation.full_path[0].predicate;
    assert!(wording.contains("Boolean parity"));
    assert!(wording.contains("XOR"));
    assert!(wording.contains("= 1 over GF(2)"));
    assert_eq!(explanation.certificate_backend, "Gf2Gaussian");
}

#[test]
fn horn_antihorn_and_two_cnf_wording_is_unambiguous() {
    let cases = [
        (
            Predicate::HornClause(vec![literal(0, false), literal(1, true)]),
            "Horn clause:",
        ),
        (
            Predicate::AntiHornClause(vec![literal(0, true), literal(1, false)]),
            "AntiHorn clause:",
        ),
        (
            Predicate::Square2Cnf {
                a: literal(0, true),
                b: literal(1, false),
                c: literal(2, true),
                d: literal(3, false),
            },
            "2-CNF:",
        ),
    ];
    for (predicate, expected) in cases {
        let explanation = compile_verified_explanation(
            &stump(predicate),
            &dataset(),
            1,
            ExplanationAudience::Engineering,
        )
        .unwrap();
        assert!(explanation.full_path[0].predicate.contains(expected));
        assert!(explanation.reliability.axp_theorem_verified);
        assert!(explanation.reliability.path_theory_verified);
    }
}
