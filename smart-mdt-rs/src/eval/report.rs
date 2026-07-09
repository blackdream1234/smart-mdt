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
/// True iff row is allowed in the theorem-certified table.
///
/// Each certified method is bound to its one certified backend. Affine is
/// admitted only with the GF(2) backend, so empirical affine (family
/// `EmpiricalAffine`, backend `Affine`) is excluded even though it shares the
/// `affine` method name.
pub fn theorem_table_filter(r: &ResultRow) -> bool {
    if !r.theorem_certified || !r.language_family.theorem_table_allowed() {
        return false;
    }
    match r.method.as_str() {
        "unary" | "horn" => matches!(r.backend, Backend::StructuralHorn),
        "antihorn" => matches!(r.backend, Backend::StructuralAntiHorn),
        "square2cnf" => matches!(r.backend, Backend::TwoSat),
        "affine" => matches!(r.backend, Backend::Gf2Gaussian),
        _ => false,
    }
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
