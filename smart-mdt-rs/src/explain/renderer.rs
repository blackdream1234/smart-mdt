//! Deterministic audience templates rendered exclusively from verified JSON.

use super::{
    verified_explanation_from_json, ExplanationAudience, UncertaintyLevel, VerifiedExplanation,
};
use crate::Result;

/// Renders human text using only fields parsed from `verified_json`.
pub fn render_human_explanation(verified_json: &str) -> Result<String> {
    let explanation = verified_explanation_from_json(verified_json)?;
    Ok(render(&explanation))
}

fn render(explanation: &VerifiedExplanation) -> String {
    let mut lines = Vec::new();
    lines.extend(audience_opening(explanation));
    lines.push(String::new());
    lines.push("Sufficient certified evidence (AXp)".into());
    if explanation.sufficient_axp_evidence.is_empty() {
        lines.push("- No feature values are required because the final tree is constant.".into());
    } else {
        lines.extend(explanation.sufficient_axp_evidence.iter().map(|evidence| {
            format!(
                "- {} (feature {}): {}",
                evidence.feature_name, evidence.feature_id, evidence.observed_value
            )
        }));
    }
    lines.push(String::new());
    lines.push("Verified reasoning".into());
    lines.extend(
        explanation
            .human_readable_reasoning_steps
            .iter()
            .map(|step| format!("- {}", step.text)),
    );
    lines.push(format!(
        "- Full path: {} certified step(s), recorded separately in verified_explanation.json.",
        explanation.full_path.len()
    ));
    lines.push(format!(
        "- Predicate families: {}.",
        if explanation.selected_predicate_families.is_empty() {
            "none (constant leaf)".into()
        } else {
            explanation.selected_predicate_families.join(", ")
        }
    ));
    lines.push(String::new());
    lines.push("Support and uncertainty".into());
    lines.push(format!(
        "- Leaf support: {} training sample(s); {} reference sample(s).",
        explanation.leaf.leaf_sample_support, explanation.leaf.reference_sample_support
    ));
    lines.push(format!(
        "- Class distribution: {}.",
        explanation
            .leaf
            .class_distribution
            .iter()
            .map(|(class, support)| format!("class {class}={support}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    lines.push(format!("- Purity: {:.6}.", explanation.leaf.purity));
    lines.push(format!(
        "- Uncertainty: {}.",
        uncertainty_name(explanation.uncertainty_level)
    ));
    if explanation.uncertainty_level != UncertaintyLevel::Low {
        lines
            .push("- Warning: support or purity is limited; review this output cautiously.".into());
    }
    lines.push(String::new());
    lines.push("Certification".into());
    lines.push(format!(
        "- Backend: {}; path theory: {}.",
        explanation.certificate_backend, explanation.path_theory_state
    ));
    lines.push(format!(
        "- AXp verified: {}; path verified: {}; reliability score: {:.6}.",
        explanation.reliability.axp_theorem_verified,
        explanation.reliability.path_theory_verified,
        explanation.reliability.reliability_score
    ));
    lines.push(String::new());
    lines.push("Closest certified counterfactual".into());
    match &explanation.closest_certified_counterfactual {
        Some(counterfactual) => {
            lines.push(format!(
                "- Reference row {} has model prediction {} under {} / {}.",
                counterfactual.reference_row,
                counterfactual.prediction,
                counterfactual.certificate_backend,
                counterfactual.path_theory_state
            ));
            lines.extend(counterfactual.changed_features.iter().map(|change| {
                format!(
                    "- {} changes from {} to {}.",
                    change.feature_name, change.from_value, change.to_value
                )
            }));
        }
        None => lines.push("- No opposite certified reference-row prediction is available.".into()),
    }
    lines.push(String::new());
    lines.push("Limitations".into());
    lines.extend(
        explanation
            .limitations
            .iter()
            .map(|limitation| format!("- {limitation}")),
    );
    if explanation.audience == ExplanationAudience::Clinical {
        lines.push(
            "- This model prediction is not a diagnosis and does not establish causation.".into(),
        );
        if explanation.uncertainty_level != UncertaintyLevel::Low
            || explanation.leaf.reference_sample_support < 20
        {
            lines.push(
                "- Professional review is recommended because confidence or support is limited."
                    .into(),
            );
        }
    }
    lines.join("\n") + "\n"
}

fn audience_opening(explanation: &VerifiedExplanation) -> Vec<String> {
    match explanation.audience {
        ExplanationAudience::General => vec![
            "CALS-MDT verified explanation".into(),
            format!("The model prediction is class {}.", explanation.prediction),
        ],
        ExplanationAudience::Clinical => vec![
            "CALS-MDT clinical-facing verified explanation".into(),
            format!("The model prediction is class {}.", explanation.prediction),
            "This output supports review of a model prediction; it is not a diagnosis.".into(),
        ],
        ExplanationAudience::Engineering => vec![
            "CALS-MDT engineering verification report".into(),
            format!("Model prediction: class {}.", explanation.prediction),
            format!("Schema: {}.", explanation.schema_version),
        ],
        ExplanationAudience::Management => vec![
            "CALS-MDT decision summary".into(),
            format!("Model prediction: class {}.", explanation.prediction),
            format!(
                "Evidence uses {} feature(s) with {} uncertainty.",
                explanation.sufficient_axp_evidence.len(),
                uncertainty_name(explanation.uncertainty_level)
            ),
        ],
        ExplanationAudience::Audit => vec![
            "CALS-MDT certified audit explanation".into(),
            format!("Model prediction: class {}.", explanation.prediction),
            format!(
                "Certification boundary: {} / {}.",
                explanation.certificate_backend, explanation.path_theory_state
            ),
        ],
        ExplanationAudience::Technical => vec![
            "CALS-MDT technical verified explanation".into(),
            format!("Model prediction: class {}.", explanation.prediction),
            format!("Schema: {}.", explanation.schema_version),
        ],
    }
}

const fn uncertainty_name(level: UncertaintyLevel) -> &'static str {
    match level {
        UncertaintyLevel::Low => "low",
        UncertaintyLevel::Medium => "medium",
        UncertaintyLevel::High => "high",
    }
}
