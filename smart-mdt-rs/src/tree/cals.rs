//! Unified CALS-MDT configuration profiles.

use super::{
    AdaptiveLanguageConfig, AxpRerankConfig, CacheConfig, ClassAwarePruningConfig,
    ConditionalCandidateSearchConfig, LearnerConfig, ParallelConfig, PruningConfig,
    SelectiveLookaheadConfig, TreeSearchConfig, TreeSearchStrategy,
};
use crate::search::{BranchAndBoundConfig, SplitScoreConfig};

#[derive(Clone, Debug, PartialEq)]
pub struct CalsConfig {
    pub scoring: SplitScoreConfig,
    pub branch_and_bound: BranchAndBoundConfig,
    pub cache: CacheConfig,
    pub tree_search: TreeSearchConfig,
    pub conditional_search: ConditionalCandidateSearchConfig,
    pub parallel: ParallelConfig,
    pub pruning: PruningConfig,
    pub adaptive_language: AdaptiveLanguageConfig,
    pub axp_rerank: AxpRerankConfig,
}

impl CalsConfig {
    /// Recommended accuracy-first thesis profile.
    pub fn thesis() -> Self {
        let branch_and_bound = BranchAndBoundConfig {
            enabled: true,
            exhaustive_fallback: true,
            top_k: 16,
            ..BranchAndBoundConfig::default()
        };
        let tree_search = TreeSearchConfig {
            strategy: TreeSearchStrategy::SparseLookahead,
            lookahead_depth: 2,
            candidate_beam_width: 16,
            tree_beam_width: 8,
            max_expansions: 1_000,
            ..TreeSearchConfig::default()
        };
        let pruning = PruningConfig {
            enabled: true,
            accuracy_epsilon: 0.005,
            ..PruningConfig::default()
        };
        let adaptive_language = AdaptiveLanguageConfig {
            enabled: true,
            total_candidate_budget: 64,
            ..AdaptiveLanguageConfig::default()
        };
        let parallel = ParallelConfig {
            enabled: true,
            ..ParallelConfig::default()
        };
        Self {
            scoring: SplitScoreConfig::sparse_certified(),
            branch_and_bound,
            cache: CacheConfig::all_enabled(),
            tree_search,
            conditional_search: ConditionalCandidateSearchConfig::default(),
            parallel,
            pruning,
            adaptive_language,
            axp_rerank: AxpRerankConfig::default(),
        }
    }

    /// Fast, serial, explanation-first profile with class-aware pruning guards.
    pub fn compact_explain() -> Self {
        let tree_search = TreeSearchConfig {
            strategy: TreeSearchStrategy::SelectiveLookahead,
            candidate_beam_width: 16,
            tree_beam_width: 4,
            lookahead_depth: 2,
            max_expansions: 500,
            selective: SelectiveLookaheadConfig {
                enabled: true,
                candidate_beam_width: 14,
                ..SelectiveLookaheadConfig::default()
            },
            ..TreeSearchConfig::default()
        };
        let pruning = PruningConfig {
            enabled: true,
            accuracy_epsilon: 0.04,
            class_aware: ClassAwarePruningConfig {
                enabled: true,
                accuracy_epsilon: 0.04,
                balanced_accuracy_epsilon: 0.4,
                minimum_minority_recall: None,
                minimum_validation_samples: 1,
                minimum_validation_samples_per_class: 1,
                root_collapse_majority_threshold: 0.9,
                preserve_subtree_when_evidence_insufficient: true,
            },
            ..PruningConfig::default()
        };
        Self {
            scoring: SplitScoreConfig::sparse_certified(),
            branch_and_bound: BranchAndBoundConfig {
                enabled: true,
                exhaustive_fallback: true,
                top_k: 8,
                ..BranchAndBoundConfig::default()
            },
            cache: CacheConfig {
                enabled: true,
                node_statistics: false,
                predicate_masks: true,
                candidate_pools: true,
                best_subtrees: false,
                lookahead: false,
                max_entries: 10_000,
                approximate_byte_limit: 128 * 1024 * 1024,
            },
            tree_search,
            conditional_search: ConditionalCandidateSearchConfig {
                enabled: true,
                branch_and_bound_candidate_threshold: 64,
                candidate_cache_minimum_expected_reuse: 2,
            },
            parallel: ParallelConfig::disabled(),
            pruning,
            adaptive_language: AdaptiveLanguageConfig {
                enabled: true,
                total_candidate_budget: 56,
                ..AdaptiveLanguageConfig::default()
            },
            axp_rerank: AxpRerankConfig::default(),
        }
    }

    pub fn learner_config(&self, max_depth: usize, random_seed: u64) -> LearnerConfig {
        LearnerConfig {
            max_depth,
            max_candidates_per_node: self.tree_search.candidate_beam_width.max(1),
            beam_width: self.tree_search.candidate_beam_width.max(1),
            split_score: self.scoring.clone(),
            branch_and_bound: self.branch_and_bound.clone(),
            cache: self.cache.clone(),
            tree_search: self.tree_search.clone(),
            conditional_search: self.conditional_search.clone(),
            parallel: self.parallel.clone(),
            pruning: self.pruning.clone(),
            adaptive_language: self.adaptive_language.clone(),
            axp_rerank: self.axp_rerank.clone(),
            language_policy: super::LanguagePolicy::SmartCertified,
            theorem_mode: true,
            random_seed,
            ..LearnerConfig::default()
        }
    }
}

impl Default for CalsConfig {
    fn default() -> Self {
        Self::thesis()
    }
}
