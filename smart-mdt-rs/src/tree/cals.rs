//! Unified CALS-MDT configuration profiles.

use super::{
    AdaptiveLanguageConfig, AxpRerankConfig, CacheConfig, LearnerConfig, ParallelConfig,
    PruningConfig, TreeSearchConfig, TreeSearchStrategy,
};
use crate::search::{BranchAndBoundConfig, SplitScoreConfig};

#[derive(Clone, Debug, PartialEq)]
pub struct CalsConfig {
    pub scoring: SplitScoreConfig,
    pub branch_and_bound: BranchAndBoundConfig,
    pub cache: CacheConfig,
    pub tree_search: TreeSearchConfig,
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
            parallel,
            pruning,
            adaptive_language,
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
