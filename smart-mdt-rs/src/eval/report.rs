use crate::logic::{Backend, LanguageFamily};
/// Benchmark row with theorem metadata.
#[derive(Clone, Debug)]
pub struct ResultRow {
    pub method: String,
    pub accuracy: f64,
    pub tree_nodes: usize,
    pub leaves: usize,
    pub max_depth_reached: usize,
    pub theorem_certified: bool,
    pub language_family: LanguageFamily,
    pub backend: Backend,
    pub git_sha: String,
    pub config: String,
}
/// True iff row is allowed in theorem-certified table.
pub fn theorem_table_filter(r: &ResultRow) -> bool {
    r.theorem_certified
        && r.language_family.theorem_table_allowed()
        && matches!(
            r.backend,
            Backend::StructuralHorn | Backend::StructuralAntiHorn | Backend::TwoSat
        )
        && !matches!(
            r.method.as_str(),
            "affine" | "bestpn" | "best-certified" | "empirical-mixed" | "tuned-experimental"
        )
}
