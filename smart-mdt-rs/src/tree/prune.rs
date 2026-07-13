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
    let mut path = Vec::new();
    let mut step = 0;
    let pruned = prune_node(
        original, validation, &all_rows, config, &mut path, &mut step,
    );
    let before_accuracy = accuracy(original, validation, &all_rows);
    let after_accuracy = accuracy(&pruned, validation, &all_rows);
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
        validation_accuracy_before: before_accuracy,
        validation_accuracy_after: after_accuracy,
        pruning_time_seconds: started.elapsed().as_secs_f64(),
        path_certified_after: tree_is_certified(&pruned),
        pruning_path: path,
    };
    (pruned, diagnostics)
}

fn prune_node(
    tree: &TreeNode,
    validation: &Dataset,
    rows: &[usize],
    config: &PruningConfig,
    path: &mut Vec<PruningPathEntry>,
    step: &mut usize,
) -> TreeNode {
    let TreeNode::Internal {
        predicate,
        left,
        right,
        majority_class,
    } = tree
    else {
        return tree.clone();
    };
    let (left_rows, right_rows): (Vec<_>, Vec<_>) = rows
        .iter()
        .copied()
        .partition(|&row| predicate.eval(&validation.features, row));
    let left_pruned = prune_node(left, validation, &left_rows, config, path, step);
    let right_pruned = prune_node(right, validation, &right_rows, config, path, step);
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
    if should_prune(&candidate, &leaf, validation, rows, config) {
        *step += 1;
        path.push(PruningPathEntry {
            step: *step,
            validation_error: 1.0 - accuracy(&leaf, validation, rows),
            nodes: leaf.nodes(),
            leaves: leaf.leaves(),
            literals: leaf.literals(),
            estimated_mean_axp_length: estimated_mean_axp(&leaf, validation, rows),
        });
        leaf
    } else {
        candidate
    }
}

fn should_prune(
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
