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
/// Checks weak AXp by enumerating finite completions induced by training-domain row values.
pub fn weak_axp_check(
    tree: &TreeNode,
    domain: &ColumnMajorMatrix,
    instance: &[f64],
    target_class: ClassId,
    selected_features: &[FeatureId],
    theorem_mode: bool,
) -> WeakAxpResult {
    let meta = backend_meta(tree, theorem_mode);
    if theorem_mode && !meta.theorem_certified {
        return WeakAxpResult {
            is_weak_axp: false,
            metadata: meta,
            opposite_paths_checked: 0,
        };
    }
    let mut rows = Vec::new();
    for i in 0..domain.rows() {
        let agree = selected_features
            .iter()
            .all(|&f| domain.get(i, f) == instance[f as usize]);
        if agree {
            rows.push(i);
        }
    }
    let opposite = rows
        .iter()
        .filter(|&&i| predict_row(tree, domain, i) != target_class)
        .count();
    WeakAxpResult {
        is_weak_axp: opposite == 0,
        metadata: meta,
        opposite_paths_checked: opposite,
    }
}
