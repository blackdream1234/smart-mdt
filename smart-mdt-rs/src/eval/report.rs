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
    /// Distinct theory states reached by root-to-leaf paths.
    pub path_theory_state: String,
    /// Distinct certified backends used by root-to-leaf paths.
    pub path_backend: String,
    /// Whether every root-to-leaf path passed theory-state validation.
    pub path_certified: bool,
    pub git_sha: String,
    pub config: String,
    pub random_state: u64,
    pub n_runs: usize,
    pub train_test_split_protocol: String,
    pub search_strategy: String,
    pub score_profile: String,
    pub candidate_beam_width: usize,
    pub tree_beam_width: usize,
    pub lookahead_depth: usize,
    pub node_budget: usize,
    pub pruning_enabled: bool,
    pub nodes_before_prune: usize,
    pub nodes_after_prune: usize,
    pub leaves_before_prune: usize,
    pub leaves_after_prune: usize,
    pub literals_before_prune: usize,
    pub literals_after_prune: usize,
    pub validation_accuracy_before_prune: f64,
    pub validation_accuracy_after_prune: f64,
    pub validation_balanced_accuracy_before_prune: f64,
    pub validation_balanced_accuracy_after_prune: f64,
    pub validation_sensitivity_before_prune: f64,
    pub validation_sensitivity_after_prune: f64,
    pub validation_specificity_before_prune: f64,
    pub validation_specificity_after_prune: f64,
    pub validation_macro_f1_before_prune: f64,
    pub validation_macro_f1_after_prune: f64,
    pub validation_minority_recall_before_prune: f64,
    pub validation_minority_recall_after_prune: f64,
    pub validation_class_support: String,
    pub pruning_root_reason: String,
    pub pruning_reason_counts: String,
    pub candidate_count: usize,
    pub candidate_pruned_count: usize,
    pub branch_and_bound_fallback_count: usize,
    pub nodes_using_greedy_selection: usize,
    pub nodes_using_selective_lookahead: usize,
    pub branch_and_bound_activation_count: usize,
    pub branch_and_bound_avoided_count: usize,
    pub cache_activation_count: usize,
    pub estimated_work_saved: usize,
    pub predicate_mask_cache_hits: usize,
    pub predicate_mask_cache_misses: usize,
    pub candidate_cache_hits: usize,
    pub candidate_cache_misses: usize,
    pub subtree_cache_hits: usize,
    pub subtree_cache_misses: usize,
    pub parallel_threads: usize,
    pub compatible_family_count: usize,
    pub selected_family_counts: String,
    pub path_violation_count: usize,
    pub max_axp_length: usize,
    pub total_fit_time: f64,
    pub search_time: f64,
    pub pruning_time: f64,
    pub axp_rerank_time: f64,
    pub empirical_fallback_used: bool,
    pub incompatible_cached_subtree_reused: bool,
    pub all_predicates_backend_allowed: bool,
    pub theorem_rejection_reason: String,
}

impl Default for ResultRow {
    fn default() -> Self {
        Self {
            dataset: String::new(),
            run: 0,
            depth: 0,
            method: String::new(),
            accuracy: 0.0,
            train_time: 0.0,
            predict_time: 0.0,
            tree_nodes: 0,
            leaves: 0,
            max_depth_reached: 0,
            mean_axp_length: 0.0,
            axp_time: 0.0,
            theorem_certified: false,
            language_family: LanguageFamily::Unary,
            backend: Backend::None,
            path_theory_state: String::new(),
            path_backend: String::new(),
            path_certified: false,
            git_sha: String::new(),
            config: String::new(),
            random_state: 0,
            n_runs: 0,
            train_test_split_protocol: String::new(),
            search_strategy: String::new(),
            score_profile: String::new(),
            candidate_beam_width: 0,
            tree_beam_width: 0,
            lookahead_depth: 0,
            node_budget: 0,
            pruning_enabled: false,
            nodes_before_prune: 0,
            nodes_after_prune: 0,
            leaves_before_prune: 0,
            leaves_after_prune: 0,
            literals_before_prune: 0,
            literals_after_prune: 0,
            validation_accuracy_before_prune: 0.0,
            validation_accuracy_after_prune: 0.0,
            validation_balanced_accuracy_before_prune: 0.0,
            validation_balanced_accuracy_after_prune: 0.0,
            validation_sensitivity_before_prune: 0.0,
            validation_sensitivity_after_prune: 0.0,
            validation_specificity_before_prune: 0.0,
            validation_specificity_after_prune: 0.0,
            validation_macro_f1_before_prune: 0.0,
            validation_macro_f1_after_prune: 0.0,
            validation_minority_recall_before_prune: 0.0,
            validation_minority_recall_after_prune: 0.0,
            validation_class_support: String::new(),
            pruning_root_reason: String::new(),
            pruning_reason_counts: String::new(),
            candidate_count: 0,
            candidate_pruned_count: 0,
            branch_and_bound_fallback_count: 0,
            nodes_using_greedy_selection: 0,
            nodes_using_selective_lookahead: 0,
            branch_and_bound_activation_count: 0,
            branch_and_bound_avoided_count: 0,
            cache_activation_count: 0,
            estimated_work_saved: 0,
            predicate_mask_cache_hits: 0,
            predicate_mask_cache_misses: 0,
            candidate_cache_hits: 0,
            candidate_cache_misses: 0,
            subtree_cache_hits: 0,
            subtree_cache_misses: 0,
            parallel_threads: 0,
            compatible_family_count: 0,
            selected_family_counts: String::new(),
            path_violation_count: 0,
            max_axp_length: 0,
            total_fit_time: 0.0,
            search_time: 0.0,
            pruning_time: 0.0,
            axp_rerank_time: 0.0,
            empirical_fallback_used: false,
            incompatible_cached_subtree_reused: false,
            all_predicates_backend_allowed: false,
            theorem_rejection_reason: String::new(),
        }
    }
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
        "smart_certified" => {
            r.path_certified
                && matches!(r.language_family, LanguageFamily::SmartCertified)
                && matches!(r.backend, Backend::PathCertified)
                && !r.path_backend.is_empty()
                && r.path_backend.split('|').all(|backend| {
                    matches!(
                        backend,
                        "StructuralHorn" | "StructuralAntiHorn" | "TwoSat" | "Gf2Gaussian"
                    )
                })
        }
        "cals" => {
            r.path_certified
                && r.path_violation_count == 0
                && !r.empirical_fallback_used
                && !r.incompatible_cached_subtree_reused
                && r.all_predicates_backend_allowed
                && r.theorem_rejection_reason.is_empty()
                && matches!(r.language_family, LanguageFamily::SmartCertified)
                && matches!(r.backend, Backend::PathCertified)
                && !r.path_backend.is_empty()
                && r.path_backend.split('|').all(|backend| {
                    matches!(
                        backend,
                        "StructuralHorn" | "StructuralAntiHorn" | "TwoSat" | "Gf2Gaussian"
                    )
                })
        }
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
    /// Number of benchmark rows represented by this warning.
    pub affected_rows: usize,
    /// Sorted, pipe-delimited run identifiers, or `all` for metadata warnings.
    pub runs: String,
    /// Sorted, pipe-delimited depth identifiers, or `all` for metadata warnings.
    pub depths: String,
    pub warning_type: String,
    /// Stable human-readable explanation of why the warning was emitted.
    pub reason: String,
    /// Backward-compatible alias retained in the benchmark CSV.
    pub message: String,
    pub value: String,
}
