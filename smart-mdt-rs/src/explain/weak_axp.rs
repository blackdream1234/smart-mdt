use super::{path_blocking::certified_opposite_completion_exists, WeakAxpResult};
use crate::{
    data::ColumnMajorMatrix,
    logic::{Backend, CertificateMetadata, LanguageFamily, PathCertificate, PathTheoryState},
    tree::{predict_row, tree_path_theory_states, TreeNode},
    ClassId, FeatureId,
};

pub(super) fn backend_meta(tree: &TreeNode, theorem_mode: bool) -> CertificateMetadata {
    let Ok(states) = tree_path_theory_states(tree) else {
        return CertificateMetadata::rejected(
            theorem_mode,
            LanguageFamily::EmpiricalMixed,
            "incompatible theories occur on a root-to-leaf path",
        );
    };
    if states.len() != 1 {
        return CertificateMetadata::new(
            theorem_mode,
            LanguageFamily::SmartCertified,
            Backend::PathCertified,
            PathCertificate::PathTheory,
        );
    }
    match states[0] {
        PathTheoryState::Uncommitted => CertificateMetadata::new(
            theorem_mode,
            LanguageFamily::Unary,
            Backend::StructuralHorn,
            PathCertificate::HornCnf,
        ),
        PathTheoryState::Horn => CertificateMetadata::new(
            theorem_mode,
            LanguageFamily::Horn,
            Backend::StructuralHorn,
            PathCertificate::HornCnf,
        ),
        PathTheoryState::AntiHorn => CertificateMetadata::new(
            theorem_mode,
            LanguageFamily::AntiHorn,
            Backend::StructuralAntiHorn,
            PathCertificate::AntiHornCnf,
        ),
        PathTheoryState::TwoSat => CertificateMetadata::new(
            theorem_mode,
            LanguageFamily::Square2Cnf,
            Backend::TwoSat,
            PathCertificate::TwoCnf,
        ),
        PathTheoryState::AffineGf2 => CertificateMetadata::new(
            theorem_mode,
            LanguageFamily::Affine,
            Backend::Gf2Gaussian,
            PathCertificate::AffineGf2,
        ),
    }
}

fn count_opposite_leaves(tree: &TreeNode, target: ClassId) -> usize {
    match tree {
        TreeNode::Leaf { class, .. } => usize::from(*class != target),
        TreeNode::Internal { left, right, .. } => {
            count_opposite_leaves(left, target) + count_opposite_leaves(right, target)
        }
    }
}

fn is_binary_instance(instance: &[f64]) -> bool {
    instance.iter().all(|v| *v == 0.0 || *v == 1.0)
}

fn is_binary_domain(domain: &ColumnMajorMatrix) -> bool {
    domain.rows() > 0
        && (0..domain.cols() as FeatureId).all(|feature| {
            domain
                .column(feature)
                .iter()
                .all(|v| *v == 0.0 || *v == 1.0)
        })
}

pub(crate) fn tree_scope_fits_domain(tree: &TreeNode, feature_count: usize) -> bool {
    match tree {
        TreeNode::Leaf { .. } => true,
        TreeNode::Internal {
            predicate,
            left,
            right,
            ..
        } => {
            predicate
                .scope_features()
                .iter()
                .all(|feature| (*feature as usize) < feature_count)
                && tree_scope_fits_domain(left, feature_count)
                && tree_scope_fits_domain(right, feature_count)
        }
    }
}

fn assignment_matrix(values: &[f64]) -> Option<ColumnMajorMatrix> {
    let row = values.to_vec();
    ColumnMajorMatrix::from_rows(&[row]).ok()
}

/// Tree/domain-invariant facts reused across many weak-AXp checks against the
/// same tree and reference domain.
///
/// `weak_axp_check` recomputed `backend_meta` (a function of `tree` and
/// `theorem_mode` only) and `is_binary_domain` (a full rescan of `domain`,
/// independent of `instance`/`selected_features`) on every single call.
/// Deletion-based AXp extraction calls it once per feature per row, and
/// full-corpus benchmarking calls it for every held-out row, so a fixed
/// `domain` was rescanned millions of times over instead of once. Profiling
/// on `letter.dl8` (224 features, depth-5 tree) showed this rescan alone
/// accounted for 99.8% of total AXp-extraction time, versus 0.1% for the
/// actual certified solve. Precomputing these once per tree/domain pair
/// changes no output: `weak_axp_check` still recomputes them fresh, so its
/// public behavior is identical; this context only lets repeat callers skip
/// redundant recomputation of values that cannot have changed.
pub(crate) struct AxpCheckContext {
    meta: CertificateMetadata,
    domain_is_binary: bool,
}

impl AxpCheckContext {
    pub(crate) fn new(tree: &TreeNode, domain: &ColumnMajorMatrix, theorem_mode: bool) -> Self {
        Self {
            meta: backend_meta(tree, theorem_mode),
            domain_is_binary: is_binary_domain(domain),
        }
    }
}

/// Checks weak AXp by blocking all opposite-class leaves.
///
/// In theorem mode, every opposite leaf is blocked with the certified solver for
/// that path's tractable Boolean theory. Non-Boolean reference domains are
/// rejected because dataset-row enumeration is not a theorem proof. Outside
/// theorem mode, the historical finite-completion/data-row checks remain
/// available without a certification claim.
pub fn weak_axp_check(
    tree: &TreeNode,
    domain: &ColumnMajorMatrix,
    instance: &[f64],
    target_class: ClassId,
    selected_features: &[FeatureId],
    theorem_mode: bool,
) -> WeakAxpResult {
    let ctx = AxpCheckContext::new(tree, domain, theorem_mode);
    weak_axp_check_with_context(
        &ctx,
        tree,
        domain,
        instance,
        target_class,
        selected_features,
        theorem_mode,
    )
}

/// Same semantics as [`weak_axp_check`], but reuses a precomputed
/// [`AxpCheckContext`] instead of recomputing tree/domain-invariant facts.
/// Produces byte-identical results to `weak_axp_check` for the same
/// `tree`/`domain`/`theorem_mode` that built `ctx`.
pub(crate) fn weak_axp_check_with_context(
    ctx: &AxpCheckContext,
    tree: &TreeNode,
    domain: &ColumnMajorMatrix,
    instance: &[f64],
    target_class: ClassId,
    selected_features: &[FeatureId],
    theorem_mode: bool,
) -> WeakAxpResult {
    let meta = ctx.meta.clone();
    let opposite_paths = count_opposite_leaves(tree, target_class);
    if domain.cols() != instance.len()
        || selected_features
            .iter()
            .any(|&feature| feature as usize >= instance.len())
        || !tree_scope_fits_domain(tree, domain.cols())
    {
        return WeakAxpResult {
            is_weak_axp: false,
            metadata: CertificateMetadata::rejected(
                theorem_mode,
                meta.language_family,
                "AXp checking requires matching dimensions, in-bounds selected features, and in-bounds tree scope",
            ),
            opposite_paths_checked: opposite_paths,
        };
    }
    if theorem_mode && !meta.theorem_certified {
        return WeakAxpResult {
            is_weak_axp: false,
            metadata: meta,
            opposite_paths_checked: opposite_paths,
        };
    }

    if theorem_mode {
        if !is_binary_instance(instance) || !ctx.domain_is_binary {
            return WeakAxpResult {
                is_weak_axp: false,
                metadata: CertificateMetadata::rejected(
                    true,
                    meta.language_family,
                    "theorem AXp checking requires a non-empty Boolean reference domain and in-bounds tree scope",
                ),
                opposite_paths_checked: opposite_paths,
            };
        }
        return match certified_opposite_completion_exists(
            tree,
            instance,
            target_class,
            selected_features,
        ) {
            Ok(has_opposite_completion) => WeakAxpResult {
                is_weak_axp: !has_opposite_completion,
                metadata: meta,
                opposite_paths_checked: opposite_paths,
            },
            Err(reason) => WeakAxpResult {
                is_weak_axp: false,
                metadata: CertificateMetadata::rejected(true, meta.language_family, reason),
                opposite_paths_checked: opposite_paths,
            },
        };
    }

    let mut has_opposite_completion = false;
    if ctx.domain_is_binary && is_binary_instance(instance) && instance.len() <= 20 {
        let n = instance.len();
        for mask in 0..(1usize << n) {
            let mut completion = vec![0.0; n];
            for (j, v) in completion.iter_mut().enumerate() {
                *v = if (mask >> j) & 1 == 1 { 1.0 } else { 0.0 };
            }
            if selected_features
                .iter()
                .all(|&f| completion[f as usize] == instance[f as usize])
            {
                if let Some(x) = assignment_matrix(&completion) {
                    if predict_row(tree, &x, 0) != target_class {
                        has_opposite_completion = true;
                        break;
                    }
                }
            }
        }
    } else {
        for i in 0..domain.rows() {
            let agree = selected_features
                .iter()
                .all(|&f| domain.get(i, f) == instance[f as usize]);
            if agree && predict_row(tree, domain, i) != target_class {
                has_opposite_completion = true;
                break;
            }
        }
    }

    WeakAxpResult {
        is_weak_axp: !has_opposite_completion,
        metadata: meta,
        opposite_paths_checked: opposite_paths,
    }
}

#[cfg(test)]
mod cached_context_regression_tests {
    //! Proves the `AxpCheckContext`-cached path is behaviorally identical to
    //! the always-recompute path it replaced in hot callers. Root cause: see
    //! the doc comment on `AxpCheckContext`. These tests directly compare
    //! `weak_axp_check` (recomputes `backend_meta`/`is_binary_domain` fresh
    //! every call) against `weak_axp_check_with_context` (reuses a
    //! precomputed context) across certified-success, non-Boolean-domain
    //! rejection, dimension-mismatch rejection, and empirical-mode cases.
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

    fn assert_same_result(a: &WeakAxpResult, b: &WeakAxpResult, case: &str) {
        assert_eq!(a.is_weak_axp, b.is_weak_axp, "{case}: is_weak_axp differs");
        assert_eq!(a.metadata, b.metadata, "{case}: metadata differs");
        assert_eq!(
            a.opposite_paths_checked, b.opposite_paths_checked,
            "{case}: opposite_paths_checked differs"
        );
    }

    #[test]
    fn cached_context_matches_uncached_on_certified_success_and_rejection() {
        let tree = and_tree();
        let domain = binary_domain(2);
        let instance = vec![1.0, 1.0];
        let ctx = AxpCheckContext::new(&tree, &domain, true);
        for selected in [vec![], vec![0], vec![1], vec![0, 1]] {
            let uncached = weak_axp_check(&tree, &domain, &instance, 1, &selected, true);
            let cached =
                weak_axp_check_with_context(&ctx, &tree, &domain, &instance, 1, &selected, true);
            assert_same_result(&uncached, &cached, "certified success/rejection");
        }
    }

    #[test]
    fn cached_context_matches_uncached_on_non_boolean_domain_rejection() {
        let tree = TreeNode::Internal {
            predicate: Predicate::Unary(ge(0)),
            majority_class: 0,
            left: Box::new(TreeNode::Leaf {
                class: 1,
                samples: 1,
            }),
            right: Box::new(TreeNode::Leaf {
                class: 0,
                samples: 1,
            }),
        };
        let domain = ColumnMajorMatrix::from_rows(&[vec![0.0], vec![2.0]]).unwrap();
        let instance = vec![2.0];
        let uncached = weak_axp_check(&tree, &domain, &instance, 1, &[], true);
        let ctx = AxpCheckContext::new(&tree, &domain, true);
        let cached = weak_axp_check_with_context(&ctx, &tree, &domain, &instance, 1, &[], true);
        assert_same_result(&uncached, &cached, "non-boolean domain rejection");
        assert!(!uncached.is_weak_axp);
    }

    #[test]
    fn cached_context_matches_uncached_on_dimension_mismatch_rejection() {
        let tree = and_tree();
        let domain = binary_domain(2);
        let instance = vec![1.0]; // wrong length vs domain.cols() == 2
        let uncached = weak_axp_check(&tree, &domain, &instance, 1, &[], true);
        let ctx = AxpCheckContext::new(&tree, &domain, true);
        let cached = weak_axp_check_with_context(&ctx, &tree, &domain, &instance, 1, &[], true);
        assert_same_result(&uncached, &cached, "dimension mismatch");
        assert!(!uncached.is_weak_axp);
    }

    #[test]
    fn cached_context_matches_uncached_in_empirical_mode() {
        let tree = and_tree();
        let domain = binary_domain(2);
        let instance = vec![1.0, 1.0];
        for selected in [vec![], vec![0], vec![1], vec![0, 1]] {
            let uncached = weak_axp_check(&tree, &domain, &instance, 1, &selected, false);
            let ctx = AxpCheckContext::new(&tree, &domain, false);
            let cached =
                weak_axp_check_with_context(&ctx, &tree, &domain, &instance, 1, &selected, false);
            assert_same_result(&uncached, &cached, "empirical mode");
        }
    }
}
