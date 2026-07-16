//! Verified, machine-readable explanations compiled from a finalized certified tree.

use super::extract_axp_deletion;
use crate::{
    data::Dataset,
    logic::{next_theory_state, Backend, Literal, PathTheoryState, Predicate, ThresholdOp},
    tree::{predict_row, tree_is_certified, TreeNode},
    ClassId, FeatureId, Result, SmartMdtError,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExplanationAudience {
    General,
    Clinical,
    Engineering,
    Management,
    Audit,
    #[default]
    Technical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyLevel {
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AxpEvidence {
    pub feature_id: FeatureId,
    pub feature_name: String,
    pub observed_value: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThresholdEvidence {
    pub feature_id: FeatureId,
    pub feature_name: String,
    pub operator: String,
    pub threshold: f64,
    pub observed_value: f64,
    pub literal_satisfied: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifiedPathStep {
    pub depth: usize,
    pub predicate_family: String,
    pub predicate: String,
    pub outcome: bool,
    pub scope_features: Vec<FeatureId>,
    pub thresholds: Vec<ThresholdEvidence>,
    pub path_theory_state: String,
    pub certificate_backend: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HumanReasoningStep {
    pub text: String,
    pub feature_ids: Vec<FeatureId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualChange {
    pub feature_id: FeatureId,
    pub feature_name: String,
    pub from_value: f64,
    pub to_value: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertifiedCounterfactual {
    pub reference_row: usize,
    pub prediction: ClassId,
    pub changed_features: Vec<CounterfactualChange>,
    pub certificate_backend: String,
    pub path_theory_state: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeafEvidence {
    pub leaf_sample_support: usize,
    pub reference_sample_support: usize,
    pub class_distribution: BTreeMap<ClassId, usize>,
    pub purity: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityEvidence {
    pub axp_theorem_verified: bool,
    pub path_theory_verified: bool,
    pub reference_support_fraction: f64,
    pub reliability_score: f64,
    pub stable_under_unfixed_feature_completion: bool,
}

/// The sole input accepted by the human-language renderer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifiedExplanation {
    pub schema_version: String,
    pub audience: ExplanationAudience,
    pub prediction: ClassId,
    pub sufficient_axp_evidence: Vec<AxpEvidence>,
    pub full_path: Vec<VerifiedPathStep>,
    pub selected_predicate_families: Vec<String>,
    pub human_readable_reasoning_steps: Vec<HumanReasoningStep>,
    pub closest_certified_counterfactual: Option<CertifiedCounterfactual>,
    pub leaf: LeafEvidence,
    pub reliability: ReliabilityEvidence,
    pub uncertainty_level: UncertaintyLevel,
    pub certificate_backend: String,
    pub path_theory_state: String,
    pub limitations: Vec<String>,
}

/// Compiles a verified explanation from a finalized tree and reference dataset.
pub fn compile_verified_explanation(
    tree: &TreeNode,
    reference: &Dataset,
    row: usize,
    audience: ExplanationAudience,
) -> Result<VerifiedExplanation> {
    if row >= reference.features.rows() {
        return Err(SmartMdtError::InvalidInput(format!(
            "explanation row {row} is out of bounds"
        )));
    }
    if !tree_is_certified(tree) {
        return Err(SmartMdtError::TheoremRejected(
            "verified explanations require a path-certified tree".into(),
        ));
    }
    let axp = extract_axp_deletion(tree, &reference.features, row, true);
    if !axp.metadata.theorem_certified {
        return Err(SmartMdtError::TheoremRejected(
            "final-tree AXp was not theorem-certified".into(),
        ));
    }
    let prediction = predict_row(tree, &reference.features, row);
    let mut state = PathTheoryState::Uncommitted;
    let mut outcomes = Vec::new();
    let mut full_path = Vec::new();
    let leaf = trace_path(
        tree,
        reference,
        row,
        0,
        &mut state,
        &mut outcomes,
        &mut full_path,
    )?;
    let sufficient_axp_evidence = axp
        .features
        .iter()
        .map(|&feature| AxpEvidence {
            feature_id: feature,
            feature_name: feature_name(feature),
            observed_value: reference.features.get(row, feature),
        })
        .collect::<Vec<_>>();
    let axp_features = axp.features.iter().copied().collect::<BTreeSet<_>>();
    let mut human_readable_reasoning_steps = sufficient_axp_evidence
        .iter()
        .map(|evidence| HumanReasoningStep {
            text: format!(
                "{} has the observed value {}.",
                evidence.feature_name,
                format_number(evidence.observed_value)
            ),
            feature_ids: vec![evidence.feature_id],
        })
        .collect::<Vec<_>>();
    human_readable_reasoning_steps.extend(
        full_path
            .iter()
            .filter(|step| {
                step.scope_features
                    .iter()
                    .all(|feature| axp_features.contains(feature))
            })
            .map(|step| HumanReasoningStep {
                text: format!(
                    "The certified {} predicate evaluated to {}: {}.",
                    step.predicate_family, step.outcome, step.predicate
                ),
                feature_ids: step.scope_features.clone(),
            }),
    );
    let selected_predicate_families = full_path
        .iter()
        .map(|step| step.predicate_family.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let leaf_evidence = leaf_evidence(tree, leaf, reference, &outcomes, prediction);
    let uncertainty_level = uncertainty(&leaf_evidence);
    let support_fraction =
        leaf_evidence.reference_sample_support as f64 / reference.labels.len().max(1) as f64;
    let reliability_score =
        leaf_evidence.purity * (leaf_evidence.reference_sample_support as f64 / 20.0).min(1.0);
    let counterfactual = closest_counterfactual(tree, reference, row, prediction)?;
    let mut limitations = vec![
        "This explanation describes the model's certified logic, not causation.".into(),
        "AXp sufficiency applies within the represented feature domain.".into(),
        "The closest counterfactual, when present, is selected from reference rows only.".into(),
    ];
    if uncertainty_level != UncertaintyLevel::Low {
        limitations
            .push("Leaf support or purity is limited; interpret the prediction cautiously.".into());
    }
    Ok(VerifiedExplanation {
        schema_version: "cals_compact_explain_v2".into(),
        audience,
        prediction,
        sufficient_axp_evidence,
        full_path,
        selected_predicate_families,
        human_readable_reasoning_steps,
        closest_certified_counterfactual: counterfactual,
        leaf: leaf_evidence,
        reliability: ReliabilityEvidence {
            axp_theorem_verified: true,
            path_theory_verified: true,
            reference_support_fraction: support_fraction,
            reliability_score,
            stable_under_unfixed_feature_completion: true,
        },
        uncertainty_level,
        certificate_backend: backend_name(state.backend()),
        path_theory_state: state.as_str().into(),
        limitations,
    })
}

pub fn verified_explanation_to_json(explanation: &VerifiedExplanation) -> Result<String> {
    serde_json::to_string_pretty(explanation)
        .map_err(|error| SmartMdtError::Json(error.to_string()))
}

pub fn verified_explanation_from_json(json: &str) -> Result<VerifiedExplanation> {
    serde_json::from_str(json).map_err(|error| SmartMdtError::Json(error.to_string()))
}

fn trace_path<'a>(
    tree: &'a TreeNode,
    reference: &Dataset,
    row: usize,
    depth: usize,
    state: &mut PathTheoryState,
    outcomes: &mut Vec<bool>,
    path: &mut Vec<VerifiedPathStep>,
) -> Result<&'a TreeNode> {
    match tree {
        TreeNode::Leaf { .. } => Ok(tree),
        TreeNode::Internal {
            predicate,
            left,
            right,
            ..
        } => {
            let outcome = predicate.eval(&reference.features, row);
            *state = next_theory_state(*state, predicate)?;
            outcomes.push(outcome);
            path.push(VerifiedPathStep {
                depth,
                predicate_family: family_name(predicate),
                predicate: predicate_text(predicate),
                outcome,
                scope_features: predicate.scope_features(),
                thresholds: predicate_literals(predicate)
                    .into_iter()
                    .map(|literal| threshold_evidence(literal, reference, row))
                    .collect(),
                path_theory_state: state.as_str().into(),
                certificate_backend: backend_name(state.backend()),
            });
            trace_path(
                if outcome { left } else { right },
                reference,
                row,
                depth + 1,
                state,
                outcomes,
                path,
            )
        }
    }
}

fn leaf_evidence(
    tree: &TreeNode,
    selected_leaf: &TreeNode,
    reference: &Dataset,
    selected_outcomes: &[bool],
    prediction: ClassId,
) -> LeafEvidence {
    let mut distribution = BTreeMap::new();
    let mut support = 0usize;
    for row in 0..reference.features.rows() {
        if trace_outcomes(tree, &reference.features, row) == selected_outcomes {
            support += 1;
            *distribution.entry(reference.labels[row]).or_default() += 1;
        }
    }
    let predicted_support = distribution.get(&prediction).copied().unwrap_or(0);
    let leaf_sample_support = match selected_leaf {
        TreeNode::Leaf { samples, .. } => *samples,
        TreeNode::Internal { .. } => 0,
    };
    LeafEvidence {
        leaf_sample_support,
        reference_sample_support: support,
        class_distribution: distribution,
        purity: if support == 0 {
            0.0
        } else {
            predicted_support as f64 / support as f64
        },
    }
}

fn closest_counterfactual(
    tree: &TreeNode,
    reference: &Dataset,
    row: usize,
    prediction: ClassId,
) -> Result<Option<CertifiedCounterfactual>> {
    let mut candidates = (0..reference.features.rows())
        .filter(|&candidate| predict_row(tree, &reference.features, candidate) != prediction)
        .map(|candidate| {
            let changes = (0..reference.features.cols() as FeatureId)
                .filter_map(|feature| {
                    let from_value = reference.features.get(row, feature);
                    let to_value = reference.features.get(candidate, feature);
                    (from_value != to_value).then(|| CounterfactualChange {
                        feature_id: feature,
                        feature_name: feature_name(feature),
                        from_value,
                        to_value,
                    })
                })
                .collect::<Vec<_>>();
            (changes.len(), candidate, changes)
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(distance, candidate, _)| (*distance, *candidate));
    let Some((_, candidate, changes)) = candidates.into_iter().next() else {
        return Ok(None);
    };
    let mut state = PathTheoryState::Uncommitted;
    let mut outcomes = Vec::new();
    let mut path = Vec::new();
    trace_path(
        tree,
        reference,
        candidate,
        0,
        &mut state,
        &mut outcomes,
        &mut path,
    )?;
    Ok(Some(CertifiedCounterfactual {
        reference_row: candidate,
        prediction: predict_row(tree, &reference.features, candidate),
        changed_features: changes,
        certificate_backend: backend_name(state.backend()),
        path_theory_state: state.as_str().into(),
    }))
}

fn trace_outcomes(
    tree: &TreeNode,
    features: &crate::data::ColumnMajorMatrix,
    row: usize,
) -> Vec<bool> {
    let mut outcomes = Vec::new();
    let mut current = tree;
    while let TreeNode::Internal {
        predicate,
        left,
        right,
        ..
    } = current
    {
        let outcome = predicate.eval(features, row);
        outcomes.push(outcome);
        current = if outcome { left } else { right };
    }
    outcomes
}

fn predicate_literals(predicate: &Predicate) -> Vec<Literal> {
    match predicate {
        Predicate::Unary(literal) => vec![*literal],
        Predicate::HornClause(literals)
        | Predicate::AntiHornClause(literals)
        | Predicate::Affine { literals, .. }
        | Predicate::EmpiricalAffine { literals, .. } => literals.clone(),
        Predicate::Square2Cnf { a, b, c, d } => vec![*a, *b, *c, *d],
    }
}

fn threshold_evidence(literal: Literal, reference: &Dataset, row: usize) -> ThresholdEvidence {
    let observed_value = reference.features.get(row, literal.atom.feature);
    let operator = match (literal.atom.op, literal.positive) {
        (ThresholdOp::LessThan, true) | (ThresholdOp::GreaterEqual, false) => "<",
        (ThresholdOp::GreaterEqual, true) | (ThresholdOp::LessThan, false) => ">=",
    };
    ThresholdEvidence {
        feature_id: literal.atom.feature,
        feature_name: feature_name(literal.atom.feature),
        operator: operator.into(),
        threshold: literal.atom.threshold,
        observed_value,
        literal_satisfied: literal.eval_value(observed_value),
    }
}

fn predicate_text(predicate: &Predicate) -> String {
    let literal_text = |literal: &Literal| {
        let operator = match (literal.atom.op, literal.positive) {
            (ThresholdOp::LessThan, true) | (ThresholdOp::GreaterEqual, false) => "<",
            (ThresholdOp::GreaterEqual, true) | (ThresholdOp::LessThan, false) => ">=",
        };
        format!(
            "{} {} {}",
            feature_name(literal.atom.feature),
            operator,
            format_number(literal.atom.threshold)
        )
    };
    match predicate {
        Predicate::Unary(literal) => literal_text(literal),
        Predicate::HornClause(literals) => format!(
            "Horn clause: ({})",
            literals
                .iter()
                .map(literal_text)
                .collect::<Vec<_>>()
                .join(" OR ")
        ),
        Predicate::AntiHornClause(literals) => format!(
            "AntiHorn clause: ({})",
            literals
                .iter()
                .map(literal_text)
                .collect::<Vec<_>>()
                .join(" OR ")
        ),
        Predicate::Square2Cnf { a, b, c, d } => format!(
            "2-CNF: ({} OR {}) AND ({} OR {})",
            literal_text(a),
            literal_text(b),
            literal_text(c),
            literal_text(d)
        ),
        Predicate::Affine { literals, rhs } => format!(
            "Boolean parity: ({}) = {} over GF(2)",
            literals
                .iter()
                .map(literal_text)
                .collect::<Vec<_>>()
                .join(" XOR "),
            usize::from(*rhs)
        ),
        Predicate::EmpiricalAffine { literals, parity } => format!(
            "Empirical parity: ({}) = {}",
            literals
                .iter()
                .map(literal_text)
                .collect::<Vec<_>>()
                .join(" XOR "),
            usize::from(*parity)
        ),
    }
}

fn family_name(predicate: &Predicate) -> String {
    match predicate {
        Predicate::Unary(_) => "Unary",
        Predicate::HornClause(_) => "Horn",
        Predicate::AntiHornClause(_) => "AntiHorn",
        Predicate::Square2Cnf { .. } => "Square2CNF",
        Predicate::Affine { .. } => "AffineGf2",
        Predicate::EmpiricalAffine { .. } => "EmpiricalAffine",
    }
    .into()
}

fn backend_name(backend: Backend) -> String {
    format!("{backend:?}")
}

fn feature_name(feature: FeatureId) -> String {
    format!("feature_{feature}")
}

fn format_number(value: f64) -> String {
    value.to_string()
}

fn uncertainty(leaf: &LeafEvidence) -> UncertaintyLevel {
    if leaf.reference_sample_support < 5 || leaf.purity < 0.6 {
        UncertaintyLevel::High
    } else if leaf.reference_sample_support < 20 || leaf.purity < 0.8 {
        UncertaintyLevel::Medium
    } else {
        UncertaintyLevel::Low
    }
}
