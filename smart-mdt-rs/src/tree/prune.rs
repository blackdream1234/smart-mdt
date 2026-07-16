//! Deterministic validation-only cost-complexity pruning.

use super::{predict_row, tree_is_certified, TreeNode};
use crate::{data::ColumnMajorMatrix, data::Dataset, ClassId, Result, SmartMdtError};
use std::{collections::BTreeMap, time::Instant};

/// Selection rule used during bottom-up pruning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PruningSelectionMode {
    CostComplexity,
    Cart,
    /// Accuracy first, then nodes, literals, estimated explanation length.
    #[default]
    EpsilonPareto,
}

/// Additional validation guards for pruning on imbalanced binary data.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassAwarePruningConfig {
    pub enabled: bool,
    pub accuracy_epsilon: f64,
    pub balanced_accuracy_epsilon: f64,
    pub minimum_minority_recall: Option<f64>,
    pub minimum_validation_samples: usize,
    pub minimum_validation_samples_per_class: usize,
    pub root_collapse_majority_threshold: f64,
    pub preserve_subtree_when_evidence_insufficient: bool,
}

impl Default for ClassAwarePruningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            accuracy_epsilon: 0.005,
            balanced_accuracy_epsilon: 0.005,
            minimum_minority_recall: None,
            minimum_validation_samples: 20,
            minimum_validation_samples_per_class: 3,
            root_collapse_majority_threshold: 0.9,
            preserve_subtree_when_evidence_insufficient: true,
        }
    }
}

/// Internal-validation pruning configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct PruningConfig {
    pub enabled: bool,
    pub validation_fraction: f64,
    pub alpha_nodes: f64,
    pub alpha_leaves: f64,
    pub alpha_literals: f64,
    pub alpha_axp: f64,
    pub accuracy_epsilon: f64,
    pub use_one_standard_error_rule: bool,
    pub selection_mode: PruningSelectionMode,
    pub class_aware: ClassAwarePruningConfig,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            validation_fraction: 0.2,
            alpha_nodes: 0.0,
            alpha_leaves: 0.0,
            alpha_literals: 0.0,
            alpha_axp: 0.0,
            accuracy_epsilon: 0.005,
            use_one_standard_error_rule: false,
            selection_mode: PruningSelectionMode::EpsilonPareto,
            class_aware: ClassAwarePruningConfig::default(),
        }
    }
}

/// Deterministic binary classification metrics used only on pruning validation rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PruningClassificationMetrics {
    pub accuracy: f64,
    pub balanced_accuracy: f64,
    pub sensitivity: f64,
    pub specificity: f64,
    pub macro_f1: f64,
    pub minority_recall: f64,
    pub class_support: BTreeMap<ClassId, usize>,
}

/// Stable reason assigned to each bottom-up pruning decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum PruningReason {
    ObjectiveImproved,
    BalancedAccuracyGuard,
    MinorityRecallGuard,
    InsufficientValidationSupport,
    RootCollapseGuard,
    #[default]
    NoChange,
}

impl PruningReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ObjectiveImproved => "objective_improved",
            Self::BalancedAccuracyGuard => "balanced_accuracy_guard",
            Self::MinorityRecallGuard => "minority_recall_guard",
            Self::InsufficientValidationSupport => "insufficient_validation_support",
            Self::RootCollapseGuard => "root_collapse_guard",
            Self::NoChange => "no_change",
        }
    }
}

/// One nested subtree produced by a bottom-up pruning decision.
#[derive(Clone, Debug, PartialEq)]
pub struct PruningPathEntry {
    pub step: usize,
    pub validation_error: f64,
    pub nodes: usize,
    pub leaves: usize,
    pub literals: usize,
    pub estimated_mean_axp_length: f64,
    pub reason: PruningReason,
}

/// Audit metrics for the selected pruned tree.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PruningDiagnostics {
    pub enabled: bool,
    pub grow_samples: usize,
    pub validation_samples: usize,
    pub validation_indices: Vec<usize>,
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub leaves_before: usize,
    pub leaves_after: usize,
    pub literals_before: usize,
    pub literals_after: usize,
    pub estimated_axp_before: f64,
    pub estimated_axp_after: f64,
    pub validation_accuracy_before: f64,
    pub validation_accuracy_after: f64,
    pub validation_metrics_before: PruningClassificationMetrics,
    pub validation_metrics_after: PruningClassificationMetrics,
    pub root_decision_reason: PruningReason,
    pub decision_reason_counts: BTreeMap<PruningReason, usize>,
    pub pruning_time_seconds: f64,
    pub path_certified_after: bool,
    pub pruning_path: Vec<PruningPathEntry>,
}

/// Deterministic stratified split wholly within the caller's training data.
#[derive(Clone, Debug)]
pub struct PruningSplit {
    pub grow: Dataset,
    pub validation: Dataset,
    pub grow_indices: Vec<usize>,
    pub validation_indices: Vec<usize>,
}

#[derive(Default)]
struct PruningAudit {
    path: Vec<PruningPathEntry>,
    step: usize,
    reason_counts: BTreeMap<PruningReason, usize>,
}

pub fn deterministic_pruning_split(
    data: &Dataset,
    validation_fraction: f64,
    seed: u64,
) -> Result<PruningSplit> {
    if data.labels.len() < 4 || !(0.0..1.0).contains(&validation_fraction) {
        return Err(SmartMdtError::InvalidInput(
            "pruning validation split needs at least four rows and a fraction in (0,1)".into(),
        ));
    }
    let mut by_class: BTreeMap<ClassId, Vec<usize>> = BTreeMap::new();
    for (index, &class) in data.labels.iter().enumerate() {
        by_class.entry(class).or_default().push(index);
    }
    let mut validation_indices = Vec::new();
    for indices in by_class.values_mut() {
        indices.sort_by_key(|&index| deterministic_key(index, seed));
        let count = ((indices.len() as f64 * validation_fraction).round() as usize)
            .min(indices.len().saturating_sub(1));
        validation_indices.extend(indices.iter().take(count).copied());
    }
    if validation_indices.is_empty() {
        let fallback = (0..data.labels.len())
            .min_by_key(|&index| deterministic_key(index, seed))
            .ok_or_else(|| SmartMdtError::InvalidInput("empty pruning data".into()))?;
        validation_indices.push(fallback);
    }
    validation_indices.sort_unstable();
    let validation_set = validation_indices
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let grow_indices = (0..data.labels.len())
        .filter(|index| !validation_set.contains(index))
        .collect::<Vec<_>>();
    if grow_indices.is_empty() {
        return Err(SmartMdtError::InvalidInput(
            "pruning split left no grow rows".into(),
        ));
    }
    Ok(PruningSplit {
        grow: select_rows(data, &grow_indices)?,
        validation: select_rows(data, &validation_indices)?,
        grow_indices,
        validation_indices,
    })
}

/// Returns a separately allocated pruned tree and complete audit diagnostics.
pub fn prune_with_validation(
    original: &TreeNode,
    validation: &Dataset,
    config: &PruningConfig,
) -> (TreeNode, PruningDiagnostics) {
    let started = Instant::now();
    let all_rows = (0..validation.labels.len()).collect::<Vec<_>>();
    let mut audit = PruningAudit::default();
    let (pruned, root_reason) =
        prune_node(original, validation, &all_rows, config, &mut audit, true);
    let before_metrics = classification_metrics(original, validation, &all_rows);
    let after_metrics = classification_metrics(&pruned, validation, &all_rows);
    let diagnostics = PruningDiagnostics {
        enabled: true,
        grow_samples: 0,
        validation_samples: validation.labels.len(),
        validation_indices: Vec::new(),
        nodes_before: original.nodes(),
        nodes_after: pruned.nodes(),
        leaves_before: original.leaves(),
        leaves_after: pruned.leaves(),
        literals_before: original.literals(),
        literals_after: pruned.literals(),
        estimated_axp_before: estimated_mean_axp(original, validation, &all_rows),
        estimated_axp_after: estimated_mean_axp(&pruned, validation, &all_rows),
        validation_accuracy_before: before_metrics.accuracy,
        validation_accuracy_after: after_metrics.accuracy,
        validation_metrics_before: before_metrics,
        validation_metrics_after: after_metrics,
        root_decision_reason: root_reason,
        decision_reason_counts: audit.reason_counts,
        pruning_time_seconds: started.elapsed().as_secs_f64(),
        path_certified_after: tree_is_certified(&pruned),
        pruning_path: audit.path,
    };
    (pruned, diagnostics)
}

fn prune_node(
    tree: &TreeNode,
    validation: &Dataset,
    rows: &[usize],
    config: &PruningConfig,
    audit: &mut PruningAudit,
    is_root: bool,
) -> (TreeNode, PruningReason) {
    let TreeNode::Internal {
        predicate,
        left,
        right,
        majority_class,
    } = tree
    else {
        return (tree.clone(), PruningReason::NoChange);
    };
    let (left_rows, right_rows): (Vec<_>, Vec<_>) = rows
        .iter()
        .copied()
        .partition(|&row| predicate.eval(&validation.features, row));
    let (left_pruned, _) = prune_node(left, validation, &left_rows, config, audit, false);
    let (right_pruned, _) = prune_node(right, validation, &right_rows, config, audit, false);
    let candidate = TreeNode::Internal {
        predicate: predicate.clone(),
        left: Box::new(left_pruned),
        right: Box::new(right_pruned),
        majority_class: *majority_class,
    };
    let leaf = TreeNode::Leaf {
        class: *majority_class,
        samples: subtree_samples(tree),
    };
    let decision = pruning_decision(&candidate, &leaf, validation, rows, config, is_root);
    *audit.reason_counts.entry(decision).or_default() += 1;
    if decision == PruningReason::ObjectiveImproved {
        audit.step += 1;
        audit.path.push(PruningPathEntry {
            step: audit.step,
            validation_error: 1.0 - accuracy(&leaf, validation, rows),
            nodes: leaf.nodes(),
            leaves: leaf.leaves(),
            literals: leaf.literals(),
            estimated_mean_axp_length: estimated_mean_axp(&leaf, validation, rows),
            reason: decision,
        });
        (leaf, decision)
    } else {
        (candidate, decision)
    }
}

fn pruning_decision(
    subtree: &TreeNode,
    leaf: &TreeNode,
    validation: &Dataset,
    rows: &[usize],
    config: &PruningConfig,
    is_root: bool,
) -> PruningReason {
    if !legacy_should_prune(subtree, leaf, validation, rows, config) {
        return PruningReason::NoChange;
    }
    if !config.class_aware.enabled {
        return PruningReason::ObjectiveImproved;
    }
    let aware = &config.class_aware;
    let subtree_metrics = classification_metrics(subtree, validation, rows);
    let leaf_metrics = classification_metrics(leaf, validation, rows);
    let support_is_sufficient = rows.len() >= aware.minimum_validation_samples
        && subtree_metrics
            .class_support
            .values()
            .all(|&support| support >= aware.minimum_validation_samples_per_class);
    if !support_is_sufficient && aware.preserve_subtree_when_evidence_insufficient {
        return PruningReason::InsufficientValidationSupport;
    }
    let majority_rate = leaf_metrics
        .class_support
        .values()
        .copied()
        .max()
        .map_or(0.0, |count| count as f64 / rows.len().max(1) as f64);
    if is_root && majority_rate < aware.root_collapse_majority_threshold {
        return PruningReason::RootCollapseGuard;
    }
    if leaf_metrics.accuracy + aware.accuracy_epsilon < subtree_metrics.accuracy {
        return PruningReason::NoChange;
    }
    if leaf_metrics.balanced_accuracy + aware.balanced_accuracy_epsilon
        < subtree_metrics.balanced_accuracy
    {
        return PruningReason::BalancedAccuracyGuard;
    }
    if let Some(minimum) = aware.minimum_minority_recall {
        if leaf_metrics.minority_recall < minimum
            || leaf_metrics.minority_recall + aware.balanced_accuracy_epsilon
                < subtree_metrics.minority_recall
        {
            return PruningReason::MinorityRecallGuard;
        }
    }
    PruningReason::ObjectiveImproved
}

fn legacy_should_prune(
    subtree: &TreeNode,
    leaf: &TreeNode,
    validation: &Dataset,
    rows: &[usize],
    config: &PruningConfig,
) -> bool {
    let subtree_error = 1.0 - accuracy(subtree, validation, rows);
    let leaf_error = 1.0 - accuracy(leaf, validation, rows);
    let standard_error = if config.use_one_standard_error_rule && !rows.is_empty() {
        let accuracy = 1.0 - subtree_error;
        (accuracy * (1.0 - accuracy) / rows.len() as f64).sqrt()
    } else {
        0.0
    };
    match config.selection_mode {
        PruningSelectionMode::EpsilonPareto => {
            leaf_error <= subtree_error + config.accuracy_epsilon + standard_error
        }
        PruningSelectionMode::Cart => {
            leaf_error + config.alpha_leaves * leaf.leaves() as f64
                <= subtree_error + config.alpha_leaves * subtree.leaves() as f64
        }
        PruningSelectionMode::CostComplexity => {
            objective(leaf_error, leaf, validation, rows, config)
                <= objective(subtree_error, subtree, validation, rows, config)
        }
    }
}

/// Computes class-aware metrics without consulting any data outside `validation`.
pub fn classification_metrics(
    tree: &TreeNode,
    validation: &Dataset,
    rows: &[usize],
) -> PruningClassificationMetrics {
    if rows.is_empty() {
        return PruningClassificationMetrics::default();
    }
    let mut support = BTreeMap::new();
    let mut true_positive_by_class = BTreeMap::new();
    let mut predicted_by_class = BTreeMap::new();
    let mut correct = 0usize;
    for &row in rows {
        let actual = validation.labels[row];
        let predicted = predict_row(tree, &validation.features, row);
        *support.entry(actual).or_default() += 1;
        *predicted_by_class.entry(predicted).or_default() += 1;
        if actual == predicted {
            correct += 1;
            *true_positive_by_class.entry(actual).or_default() += 1;
        }
    }
    let recalls = support
        .iter()
        .map(|(&class, &class_support)| {
            true_positive_by_class.get(&class).copied().unwrap_or(0) as f64 / class_support as f64
        })
        .collect::<Vec<_>>();
    let balanced_accuracy = recalls.iter().sum::<f64>() / recalls.len().max(1) as f64;
    let macro_f1 = support
        .iter()
        .map(|(&class, &class_support)| {
            let true_positive = true_positive_by_class.get(&class).copied().unwrap_or(0) as f64;
            let precision =
                true_positive / predicted_by_class.get(&class).copied().unwrap_or(0).max(1) as f64;
            let recall = true_positive / class_support as f64;
            if precision + recall == 0.0 {
                0.0
            } else {
                2.0 * precision * recall / (precision + recall)
            }
        })
        .sum::<f64>()
        / support.len().max(1) as f64;
    let minority_class = support
        .iter()
        .min_by_key(|(class, class_support)| (**class_support, **class))
        .map(|(&class, _)| class);
    let recall_for = |class: ClassId| {
        let class_support = support.get(&class).copied().unwrap_or(0);
        if class_support == 0 {
            0.0
        } else {
            true_positive_by_class.get(&class).copied().unwrap_or(0) as f64 / class_support as f64
        }
    };
    let positive_class = support.keys().copied().max().unwrap_or(1);
    let negative_class = support.keys().copied().min().unwrap_or(0);
    PruningClassificationMetrics {
        accuracy: correct as f64 / rows.len() as f64,
        balanced_accuracy,
        sensitivity: recall_for(positive_class),
        specificity: recall_for(negative_class),
        macro_f1,
        minority_recall: minority_class.map_or(0.0, recall_for),
        class_support: support,
    }
}

fn objective(
    error: f64,
    tree: &TreeNode,
    validation: &Dataset,
    rows: &[usize],
    config: &PruningConfig,
) -> f64 {
    error
        + config.alpha_nodes * tree.nodes().saturating_sub(tree.leaves()) as f64
        + config.alpha_leaves * tree.leaves() as f64
        + config.alpha_literals * tree.literals() as f64
        + config.alpha_axp * estimated_mean_axp(tree, validation, rows)
}

fn accuracy(tree: &TreeNode, validation: &Dataset, rows: &[usize]) -> f64 {
    if rows.is_empty() {
        return 1.0;
    }
    rows.iter()
        .filter(|&&row| predict_row(tree, &validation.features, row) == validation.labels[row])
        .count() as f64
        / rows.len() as f64
}

fn estimated_mean_axp(tree: &TreeNode, validation: &Dataset, rows: &[usize]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    rows.iter()
        .map(|&row| path_literals(tree, &validation.features, row))
        .sum::<usize>() as f64
        / rows.len() as f64
}

fn path_literals(tree: &TreeNode, features: &ColumnMajorMatrix, row: usize) -> usize {
    match tree {
        TreeNode::Leaf { .. } => 0,
        TreeNode::Internal {
            predicate,
            left,
            right,
            ..
        } => {
            predicate.arity()
                + if predicate.eval(features, row) {
                    path_literals(left, features, row)
                } else {
                    path_literals(right, features, row)
                }
        }
    }
}

fn subtree_samples(tree: &TreeNode) -> usize {
    match tree {
        TreeNode::Leaf { samples, .. } => *samples,
        TreeNode::Internal { left, right, .. } => subtree_samples(left) + subtree_samples(right),
    }
}

fn select_rows(data: &Dataset, indices: &[usize]) -> Result<Dataset> {
    let rows = indices
        .iter()
        .map(|&row| {
            (0..data.features.cols() as u32)
                .map(|feature| data.features.get(row, feature))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let labels = indices.iter().map(|&row| data.labels[row]).collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows)?, labels)
}

fn deterministic_key(index: usize, seed: u64) -> u64 {
    let mut value = index as u64 ^ seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
