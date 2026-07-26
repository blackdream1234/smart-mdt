use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    explain::{
        compile_verified_explanation, render_human_explanation, verified_explanation_to_json,
        ExplanationAudience, UncertaintyLevel,
    },
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    tree::TreeNode,
};

fn dataset() -> Dataset {
    Dataset::new(
        ColumnMajorMatrix::from_rows(&[
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ])
        .unwrap(),
        vec![0, 0, 1, 1],
    )
    .unwrap()
}

fn tree() -> TreeNode {
    TreeNode::Internal {
        predicate: Predicate::Unary(Literal {
            atom: ThresholdAtom {
                feature: 0,
                threshold_id: 0,
                threshold: 0.5,
                op: ThresholdOp::GreaterEqual,
            },
            positive: true,
        }),
        majority_class: 0,
        left: Box::new(TreeNode::Leaf {
            class: 1,
            samples: 2,
        }),
        right: Box::new(TreeNode::Leaf {
            class: 0,
            samples: 2,
        }),
    }
}

fn rendered(audience: ExplanationAudience) -> String {
    let explanation = compile_verified_explanation(&tree(), &dataset(), 0, audience).unwrap();
    let json = verified_explanation_to_json(&explanation).unwrap();
    render_human_explanation(&json).unwrap()
}

#[test]
fn all_audience_templates_are_deterministic() {
    for audience in [
        ExplanationAudience::General,
        ExplanationAudience::Clinical,
        ExplanationAudience::Engineering,
        ExplanationAudience::Management,
        ExplanationAudience::Audit,
        ExplanationAudience::Technical,
    ] {
        assert_eq!(rendered(audience), rendered(audience));
    }
}

#[test]
fn clinical_mode_is_non_diagnostic_and_warns_on_low_support() {
    let explanation =
        compile_verified_explanation(&tree(), &dataset(), 0, ExplanationAudience::Clinical)
            .unwrap();
    assert_eq!(explanation.uncertainty_level, UncertaintyLevel::High);
    let text =
        render_human_explanation(&verified_explanation_to_json(&explanation).unwrap()).unwrap();
    assert!(text.contains("model prediction"));
    assert!(text.contains("not a diagnosis"));
    assert!(text.contains("does not establish causation"));
    assert!(text.contains("Professional review is recommended"));
    assert!(text.contains("Leaf support"));
    assert!(text.contains("Uncertainty: high"));
    assert!(!text.contains("caused by"));
}

#[test]
fn renderer_uses_only_verified_json_as_input() {
    let mut explanation =
        compile_verified_explanation(&tree(), &dataset(), 0, ExplanationAudience::Management)
            .unwrap();
    let original =
        render_human_explanation(&verified_explanation_to_json(&explanation).unwrap()).unwrap();
    explanation.prediction = 77;
    let changed =
        render_human_explanation(&verified_explanation_to_json(&explanation).unwrap()).unwrap();
    assert_ne!(original, changed);
    assert!(changed.contains("Model prediction: class 77"));
}
