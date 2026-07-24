use super::{
    weak_axp::{
        backend_meta, tree_scope_fits_domain, weak_axp_check_with_context, AxpCheckContext,
    },
    AxpResult,
};
use crate::{
    data::ColumnMajorMatrix,
    logic::CertificateMetadata,
    tree::{predict_row, TreeNode},
    FeatureId,
};
use std::time::Instant;

/// Extracts a subset-minimal AXp using the deterministic deletion algorithm.
///
/// Builds its own [`AxpCheckContext`] so tree/domain-invariant facts (see
/// `weak_axp_check_with_context`'s docs) are computed once for this call
/// instead of once per feature-deletion trial. Behavior is identical to
/// calling `weak_axp_check` directly on every trial; only redundant
/// recomputation is removed. Callers that already extract many rows against
/// the same tree/domain (e.g. `extract_final_tree_axps`) should build one
/// `AxpCheckContext` and call `extract_axp_deletion_with_context` instead, to
/// also avoid rebuilding the context on every row.
pub fn extract_axp_deletion(
    tree: &TreeNode,
    domain: &ColumnMajorMatrix,
    row: usize,
    theorem_mode: bool,
) -> AxpResult {
    let ctx = AxpCheckContext::new(tree, domain, theorem_mode);
    extract_axp_deletion_with_context(&ctx, tree, domain, row, theorem_mode)
}

/// Same semantics as [`extract_axp_deletion`], but reuses a precomputed
/// [`AxpCheckContext`] instead of rebuilding it for this call.
pub(crate) fn extract_axp_deletion_with_context(
    ctx: &AxpCheckContext,
    tree: &TreeNode,
    domain: &ColumnMajorMatrix,
    row: usize,
    theorem_mode: bool,
) -> AxpResult {
    let start = Instant::now();
    if row >= domain.rows() || !tree_scope_fits_domain(tree, domain.cols()) {
        let meta = backend_meta(tree, theorem_mode);
        let reason = if row >= domain.rows() {
            "AXp extraction row is out of bounds"
        } else {
            "AXp extraction tree scope is out of bounds"
        };
        return AxpResult::new(
            Vec::new(),
            0,
            0,
            CertificateMetadata::rejected(theorem_mode, meta.language_family, reason),
            start.elapsed(),
        );
    }
    let instance: Vec<f64> = (0..domain.cols())
        .map(|j| domain.get(row, j as u32))
        .collect();
    let target = predict_row(tree, domain, row);
    let mut selected: Vec<FeatureId> = (0..domain.cols() as u32).collect();
    let mut checks = 0;
    let mut paths = 0;
    let mut last = None;
    for f in 0..domain.cols() as u32 {
        let trial: Vec<_> = selected.iter().copied().filter(|x| *x != f).collect();
        let r =
            weak_axp_check_with_context(ctx, tree, domain, &instance, target, &trial, theorem_mode);
        checks += 1;
        paths += r.opposite_paths_checked;
        last = Some(r.metadata.clone());
        if r.is_weak_axp {
            selected = trial;
        }
    }
    let meta = last.unwrap_or_else(|| {
        weak_axp_check_with_context(
            ctx,
            tree,
            domain,
            &instance,
            target,
            &selected,
            theorem_mode,
        )
        .metadata
    });
    AxpResult::new(selected, checks, paths, meta, start.elapsed())
}

#[cfg(test)]
mod cached_context_regression_tests {
    //! Proves `extract_axp_deletion` (builds its own context) and
    //! `extract_axp_deletion_with_context` (reuses a caller-supplied context,
    //! as `extract_final_tree_axps` does across many rows) agree exactly on
    //! the extracted AXp and its certificate, for the same tree/domain/row.
    //! `elapsed_micros` is excluded from comparison since it is a timing
    //! measurement, not part of the AXp result.
    use super::*;
    use crate::logic::{Literal, Predicate, ThresholdAtom, ThresholdOp};

    fn ge(feature: FeatureId) -> Literal {
        Literal {
            atom: ThresholdAtom {
                feature,
                threshold_id: 0,
                threshold: 0.5,
                op: ThresholdOp::GreaterEqual,
            },
            positive: true,
        }
    }

    fn and_tree() -> TreeNode {
        TreeNode::Internal {
            predicate: Predicate::Unary(ge(0)),
            majority_class: 0,
            left: Box::new(TreeNode::Internal {
                predicate: Predicate::Unary(ge(1)),
                majority_class: 0,
                left: Box::new(TreeNode::Leaf {
                    class: 1,
                    samples: 1,
                }),
                right: Box::new(TreeNode::Leaf {
                    class: 0,
                    samples: 1,
                }),
            }),
            right: Box::new(TreeNode::Leaf {
                class: 0,
                samples: 2,
            }),
        }
    }

    fn binary_domain(n: usize) -> ColumnMajorMatrix {
        let rows: Vec<Vec<f64>> = (0..(1usize << n))
            .map(|m| (0..n).map(|j| ((m >> j) & 1) as f64).collect())
            .collect();
        ColumnMajorMatrix::from_rows(&rows).unwrap()
    }

    fn assert_same_result(a: &AxpResult, b: &AxpResult, case: &str) {
        assert_eq!(a.features, b.features, "{case}: features differ");
        assert_eq!(a.weak_checks, b.weak_checks, "{case}: weak_checks differ");
        assert_eq!(
            a.opposite_paths_checked, b.opposite_paths_checked,
            "{case}: opposite_paths_checked differ"
        );
        assert_eq!(a.metadata, b.metadata, "{case}: metadata differs");
    }

    #[test]
    fn cached_context_matches_uncached_for_every_row() {
        let tree = and_tree();
        let domain = binary_domain(2);
        let ctx = AxpCheckContext::new(&tree, &domain, true);
        for row in 0..domain.rows() {
            let uncached = extract_axp_deletion(&tree, &domain, row, true);
            let cached = extract_axp_deletion_with_context(&ctx, &tree, &domain, row, true);
            assert_same_result(&uncached, &cached, &format!("row {row}"));
        }
    }

    #[test]
    fn cached_context_matches_uncached_on_out_of_bounds_row() {
        let tree = and_tree();
        let domain = binary_domain(2);
        let ctx = AxpCheckContext::new(&tree, &domain, true);
        let uncached = extract_axp_deletion(&tree, &domain, domain.rows(), true);
        let cached = extract_axp_deletion_with_context(&ctx, &tree, &domain, domain.rows(), true);
        assert_same_result(&uncached, &cached, "out of bounds row");
        assert!(!uncached.metadata.theorem_certified);
    }

    #[test]
    fn cached_context_matches_uncached_in_empirical_mode() {
        let tree = and_tree();
        let domain = binary_domain(2);
        let ctx = AxpCheckContext::new(&tree, &domain, false);
        for row in 0..domain.rows() {
            let uncached = extract_axp_deletion(&tree, &domain, row, false);
            let cached = extract_axp_deletion_with_context(&ctx, &tree, &domain, row, false);
            assert_same_result(&uncached, &cached, &format!("empirical row {row}"));
        }
    }
}
