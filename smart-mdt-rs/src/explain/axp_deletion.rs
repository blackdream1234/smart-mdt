use super::{
    weak_axp::{backend_meta, tree_scope_fits_domain},
    weak_axp_check, AxpResult,
};
use crate::{
    data::ColumnMajorMatrix,
    logic::CertificateMetadata,
    tree::{predict_row, TreeNode},
    FeatureId,
};
use std::time::Instant;
/// Extracts a subset-minimal AXp using the deterministic deletion algorithm.
pub fn extract_axp_deletion(
    tree: &TreeNode,
    domain: &ColumnMajorMatrix,
    row: usize,
    theorem_mode: bool,
) -> AxpResult {
    let start = Instant::now();
    if row >= domain.rows() || !tree_scope_fits_domain(tree, domain.cols()) {
        let meta = backend_meta(tree, theorem_mode, false);
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
        let r = weak_axp_check(tree, domain, &instance, target, &trial, theorem_mode);
        checks += 1;
        paths += r.opposite_paths_checked;
        last = Some(r.metadata.clone());
        if r.is_weak_axp {
            selected = trial;
        }
    }
    let meta = last.unwrap_or_else(|| {
        weak_axp_check(tree, domain, &instance, target, &selected, theorem_mode).metadata
    });
    AxpResult::new(selected, checks, paths, meta, start.elapsed())
}
