use crate::logic::{Backend, LanguageFamily};
/// Benchmark row with theorem metadata.
#[derive(Clone, Debug)]
pub struct ResultRow {
    pub dataset: String,
    pub run: usize,
    pub depth: usize,
    pub method: String,
    pub accuracy: f64,
    pub train_time: f64,
    pub predict_time: f64,
    pub tree_nodes: usize,
    pub leaves: usize,
    pub max_depth_reached: usize,
    pub mean_axp_length: f64,
    pub axp_time: f64,
    pub theorem_certified: bool,
    pub language_family: LanguageFamily,
    pub backend: Backend,
    pub git_sha: String,
    pub config: String,
    pub random_state: u64,
    pub n_runs: usize,
    pub train_test_split_protocol: String,
}
/// True iff row is allowed in theorem-certified table.
pub fn theorem_table_filter(r: &ResultRow) -> bool {
    r.theorem_certified
        && matches!(
            r.method.as_str(),
            "unary" | "horn" | "antihorn" | "square2cnf"
        )
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

/// Benchmark warning row.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkWarning {
    pub dataset: String,
    pub run: String,
    pub depth: String,
    pub method: String,
    pub warning_type: String,
    pub message: String,
    pub value: String,
}
