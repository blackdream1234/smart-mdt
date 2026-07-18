//! AXp extraction that is intentionally restricted to the selected final tree.

use super::{extract_axp_deletion, weak_axp_check, AxpResult};
use crate::{
    data::ColumnMajorMatrix,
    tree::{predict_row, tree_is_certified, TreeNode},
};

/// Deterministic AXp summary for a tree after growth, pruning, and selection.
#[derive(Clone, Debug)]
pub struct FinalTreeAxpSummary {
    pub results: Vec<AxpResult>,
    pub mean_length: f64,
    pub max_length: usize,
    pub theorem_certified: bool,
}

/// Extracts AXps only from `tree`, which callers must already have finalized.
pub fn extract_final_tree_axps(
    tree: &TreeNode,
    features: &ColumnMajorMatrix,
    maximum_rows: usize,
    theorem_mode: bool,
) -> FinalTreeAxpSummary {
    let row_count = features.rows().min(maximum_rows);
    let results = (0..row_count)
        .map(|row| extract_axp_deletion(tree, features, row, theorem_mode))
        .collect::<Vec<_>>();
    let total = results
        .iter()
        .map(|result| result.features.len())
        .sum::<usize>();
    let max_length = results
        .iter()
        .map(|result| result.features.len())
        .max()
        .unwrap_or(0);
    let results_verified = results.iter().enumerate().all(|(row, result)| {
        let instance = (0..features.cols())
            .map(|feature| features.get(row, feature as u32))
            .collect::<Vec<_>>();
        let target = predict_row(tree, features, row);
        let sufficient = weak_axp_check(
            tree,
            features,
            &instance,
            target,
            &result.features,
            theorem_mode,
        );
        sufficient.is_weak_axp
            && sufficient.metadata.theorem_certified
            && result.features.iter().all(|removed| {
                let subset = result
                    .features
                    .iter()
                    .copied()
                    .filter(|feature| feature != removed)
                    .collect::<Vec<_>>();
                !weak_axp_check(tree, features, &instance, target, &subset, theorem_mode)
                    .is_weak_axp
            })
    });
    let theorem_certified = tree_is_certified(tree)
        && results
            .iter()
            .all(|result| result.metadata.theorem_certified)
        && results_verified;
    FinalTreeAxpSummary {
        mean_length: if results.is_empty() {
            0.0
        } else {
            total as f64 / results.len() as f64
        },
        max_length,
        theorem_certified,
        results,
    }
}
