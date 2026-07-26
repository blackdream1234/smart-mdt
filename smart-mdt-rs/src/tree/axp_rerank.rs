//! Explanation-aware reranking metadata for a bounded certified shortlist.

use crate::logic::{Backend, LanguageFamily, PathTheoryState};

#[derive(Clone, Debug, PartialEq)]
pub struct AxpRerankConfig {
    pub enabled: bool,
    pub shortlist_size: usize,
    pub validation_samples: usize,
    pub weight_mean_axp: f64,
    pub weight_max_axp: f64,
    pub timeout_ms: Option<u64>,
}

impl Default for AxpRerankConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            shortlist_size: 8,
            validation_samples: 8,
            weight_mean_axp: 0.001,
            weight_max_axp: 0.0005,
            timeout_ms: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AxpCandidateDiagnostics {
    pub canonical_predicate: String,
    pub family: LanguageFamily,
    pub path_theory_state: PathTheoryState,
    pub backend: Backend,
    pub original_score: f64,
    pub rerank_score: f64,
    pub mean_axp_length: Option<f64>,
    pub max_axp_length: Option<usize>,
    pub validation_samples: usize,
    pub timed_out: bool,
    pub theorem_certified: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AxpRerankDiagnostics {
    pub enabled: bool,
    pub shortlist_considered: usize,
    pub candidates_evaluated: usize,
    pub timeout_count: usize,
    pub elapsed_seconds: f64,
    pub candidates: Vec<AxpCandidateDiagnostics>,
}
