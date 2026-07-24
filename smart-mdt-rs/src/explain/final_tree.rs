//! AXp extraction that is intentionally restricted to the selected final tree.

use super::{
    axp_deletion::extract_axp_deletion_with_context,
    weak_axp::{weak_axp_check_with_context, AxpCheckContext},
    AxpResult,
};
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
    pub valid_count: usize,
    pub minimal_count: usize,
    pub theorem_certified: bool,
}

/// Extracts AXps only from `tree`, which callers must already have finalized.
///
/// Builds a single [`AxpCheckContext`] for the whole call (one tree, one
/// reference domain) instead of letting it be rebuilt per row or per
/// feature-deletion trial. See `weak_axp_check_with_context`'s docs for why
/// this matters: without it, the domain-Boolean check alone was rescanning
/// the full held-out row set on effectively every weak-AXp check, which
/// dominated (>99%) measured AXp-extraction time for full-corpus benchmarks.
/// Output is unchanged; only redundant recomputation is removed.
pub fn extract_final_tree_axps(
    tree: &TreeNode,
    features: &ColumnMajorMatrix,
    maximum_rows: usize,
    theorem_mode: bool,
) -> FinalTreeAxpSummary {
    let row_count = features.rows().min(maximum_rows);
    let ctx = AxpCheckContext::new(tree, features, theorem_mode);
    let results = (0..row_count)
        .map(|row| extract_axp_deletion_with_context(&ctx, tree, features, row, theorem_mode))
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
    let verification = results
        .iter()
        .enumerate()
        .map(|(row, result)| {
            let instance = (0..features.cols())
                .map(|feature| features.get(row, feature as u32))
                .collect::<Vec<_>>();
            let target = predict_row(tree, features, row);
            let sufficient = weak_axp_check_with_context(
                &ctx,
                tree,
                features,
                &instance,
                target,
                &result.features,
                theorem_mode,
            );
            let valid = sufficient.is_weak_axp && sufficient.metadata.theorem_certified;
            let minimal = valid
                && result.features.iter().all(|removed| {
                    let subset = result
                        .features
                        .iter()
                        .copied()
                        .filter(|feature| feature != removed)
                        .collect::<Vec<_>>();
                    !weak_axp_check_with_context(
                        &ctx,
                        tree,
                        features,
                        &instance,
                        target,
                        &subset,
                        theorem_mode,
                    )
                    .is_weak_axp
                });
            (valid, minimal)
        })
        .collect::<Vec<_>>();
    let valid_count = verification.iter().filter(|(valid, _)| *valid).count();
    let minimal_count = verification.iter().filter(|(_, minimal)| *minimal).count();
    let theorem_certified = tree_is_certified(tree)
        && results
            .iter()
            .all(|result| result.metadata.theorem_certified)
        && valid_count == results.len()
        && minimal_count == results.len();
    FinalTreeAxpSummary {
        mean_length: if results.is_empty() {
            0.0
        } else {
            total as f64 / results.len() as f64
        },
        max_length,
        valid_count,
        minimal_count,
        theorem_certified,
        results,
    }
}
