//! Deterministic parallel-search configuration and diagnostics.

/// Controls bounded Rayon evaluation within one training invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParallelConfig {
    pub enabled: bool,
    pub threads: Option<usize>,
    pub parallel_candidates: bool,
    pub parallel_beam_states: bool,
    pub minimum_parallel_work: usize,
}

impl ParallelConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            threads: Some(1),
            parallel_candidates: false,
            parallel_beam_states: false,
            minimum_parallel_work: usize::MAX,
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threads: None,
            parallel_candidates: true,
            parallel_beam_states: true,
            minimum_parallel_work: 2,
        }
    }
}

/// Work observed by the deterministic parallel evaluator.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParallelDiagnostics {
    pub configured_threads: usize,
    pub family_tasks: usize,
    pub candidate_batches_parallelized: usize,
    pub beam_batches_parallelized: usize,
    pub serial_fallbacks: usize,
}
