use super::WeakAxpResult;
use crate::{
    data::ColumnMajorMatrix,
    logic::{Backend, CertificateMetadata, LanguageFamily, PathCertificate, PathTheoryState},
    tree::{predict_row, tree_path_theory_states, TreeNode},
    ClassId, FeatureId,
};

fn backend_meta(tree: &TreeNode, theorem_mode: bool) -> CertificateMetadata {
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

fn assignment_matrix(values: &[f64]) -> Option<ColumnMajorMatrix> {
    let row = values.to_vec();
    ColumnMajorMatrix::from_rows(&[row]).ok()
}

/// Checks weak AXp by blocking all opposite-class leaves.
///
/// For binary domains, this performs the direct finite-completion semantics: every
/// Boolean completion agreeing with the selected features is predicted and any
/// opposite prediction witnesses that `selected_features` is not weak. For
/// non-binary data, it conservatively checks completions present as rows in the
/// supplied domain matrix, which is useful for dataset-backed smoke tests but is
/// not reported as a stronger formal guarantee.
pub fn weak_axp_check(
    tree: &TreeNode,
    domain: &ColumnMajorMatrix,
    instance: &[f64],
    target_class: ClassId,
    selected_features: &[FeatureId],
    theorem_mode: bool,
) -> WeakAxpResult {
    let meta = backend_meta(tree, theorem_mode);
    let opposite_paths = count_opposite_leaves(tree, target_class);
    if theorem_mode && !meta.theorem_certified {
        return WeakAxpResult {
            is_weak_axp: false,
            metadata: meta,
            opposite_paths_checked: opposite_paths,
        };
    }

    let mut has_opposite_completion = false;
    if is_binary_instance(instance) && instance.len() <= 20 {
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
