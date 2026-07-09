use super::WeakAxpResult;
use crate::{
    data::ColumnMajorMatrix,
    logic::{CertificateMetadata, LanguageFamily},
    tree::{predict_row, TreeNode},
    ClassId, FeatureId,
};

fn backend_meta(tree: &TreeNode, theorem_mode: bool) -> CertificateMetadata {
    fn fam(t: &TreeNode, acc: &mut Vec<LanguageFamily>) {
        if let TreeNode::Internal {
            predicate,
            left,
            right,
            ..
        } = t
        {
            acc.push(predicate.language());
            fam(left, acc);
            fam(right, acc);
        }
    }
    let mut fs = Vec::new();
    fam(tree, &mut fs);
    let f = fs.first().copied().unwrap_or(LanguageFamily::Unary);
    let same = fs.iter().all(|x| *x == f);
    let meta = match f {
        LanguageFamily::Unary | LanguageFamily::Horn => CertificateMetadata::new(
            theorem_mode,
            f,
            crate::logic::Backend::StructuralHorn,
            crate::logic::PathCertificate::HornCnf,
        ),
        LanguageFamily::AntiHorn => CertificateMetadata::new(
            theorem_mode,
            f,
            crate::logic::Backend::StructuralAntiHorn,
            crate::logic::PathCertificate::AntiHornCnf,
        ),
        LanguageFamily::Square2Cnf => CertificateMetadata::new(
            theorem_mode,
            f,
            crate::logic::Backend::TwoSat,
            crate::logic::PathCertificate::TwoCnf,
        ),
        LanguageFamily::Affine => CertificateMetadata::new(
            theorem_mode,
            f,
            crate::logic::Backend::Gf2Gaussian,
            crate::logic::PathCertificate::AffineGf2,
        ),
        _ => CertificateMetadata::rejected(theorem_mode, f, "empirical path"),
    };
    if theorem_mode && !same {
        CertificateMetadata::rejected(
            true,
            LanguageFamily::EmpiricalMixed,
            "mixed paths are not theorem-certified",
        )
    } else {
        meta
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
