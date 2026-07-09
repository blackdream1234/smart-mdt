use super::{weak_axp_check, AxpResult};
use crate::{
    data::ColumnMajorMatrix,
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
