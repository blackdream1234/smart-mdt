use super::{path_blocking::certified_opposite_completion_exists, WeakAxpResult};
use crate::{
    data::ColumnMajorMatrix,
    logic::{
        Backend, CertificateMetadata, ComplementCheck, DomainRegime, LanguageFamily, PathCheck,
        PathTheoryState, StructuralCheck, TheoremCertificate, TheoremSource,
    },
    tree::{predict_row, tree_path_theory_states, TreeNode},
    ClassId, FeatureId,
};

pub(super) fn backend_meta(
    tree: &TreeNode,
    theorem_mode: bool,
    assumptions_supported: bool,
) -> CertificateMetadata {
    let Ok(states) = tree_path_theory_states(tree) else {
        return CertificateMetadata::rejected(
            theorem_mode,
            LanguageFamily::EmpiricalMixed,
            "incompatible theories occur on a root-to-leaf path",
        );
    };
    let (language_family, theorem_id, structural_check, complement_check, backend, path_check) =
        if states.len() != 1 {
            (
                LanguageFamily::SmartCertified,
                TheoremSource::Proposition1,
                StructuralCheck::PathCompatibleExactRelations,
                ComplementCheck::PerNodeVerified,
                Backend::PathCertified,
                PathCheck::PerPathTheoryValidated,
            )
        } else {
            match states[0] {
                PathTheoryState::Uncommitted => (
                    LanguageFamily::Unary,
                    TheoremSource::UnaryBaseline,
                    StructuralCheck::UnaryRelation,
                    ComplementCheck::UnaryNegation,
                    Backend::StructuralHorn,
                    PathCheck::HornCnfValidated,
                ),
                PathTheoryState::Horn => (
                    LanguageFamily::Horn,
                    TheoremSource::Theorem3,
                    StructuralCheck::StarNestedHorn,
                    ComplementCheck::StarNestedConstruction,
                    Backend::StructuralHorn,
                    PathCheck::HornCnfValidated,
                ),
                PathTheoryState::AntiHorn => (
                    LanguageFamily::AntiHorn,
                    TheoremSource::Theorem4,
                    StructuralCheck::StarNestedAntiHorn,
                    ComplementCheck::StarNestedConstruction,
                    Backend::StructuralAntiHorn,
                    PathCheck::AntiHornCnfValidated,
                ),
                PathTheoryState::TwoSat => (
                    LanguageFamily::Square2Cnf,
                    TheoremSource::Theorem6,
                    StructuralCheck::Square2CnfFormI,
                    ComplementCheck::Square2CnfDualForm,
                    Backend::TwoSat,
                    PathCheck::TwoCnfValidated,
                ),
                PathTheoryState::AffineGf2 => (
                    LanguageFamily::Affine,
                    TheoremSource::Theorem5,
                    StructuralCheck::SingleGf2Equation,
                    ComplementCheck::Gf2RhsFlip,
                    Backend::Gf2Gaussian,
                    PathCheck::Gf2SystemValidated,
                ),
            }
        };
    CertificateMetadata::from_theorem(
        theorem_mode,
        TheoremCertificate {
            domain_regime: DomainRegime::Boolean,
            language_family,
            theorem_id,
            structural_check,
            complement_check,
            backend,
            assumptions_supported,
            path_check,
        },
    )
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
    let mut meta = backend_meta(tree, theorem_mode, false);
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
        if !is_binary_instance(instance) || !is_binary_domain(domain) {
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
        meta = backend_meta(tree, theorem_mode, true);
        if !meta.theorem_certified {
            return WeakAxpResult {
                is_weak_axp: false,
                metadata: meta,
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
    if is_binary_domain(domain) && is_binary_instance(instance) && instance.len() <= 20 {
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
