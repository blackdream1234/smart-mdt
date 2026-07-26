//! Per-fit collision-safe memoization and transposition-table types.

use super::TreeNode;
use crate::{data::BitSet, logic::PathTheoryState, search::SplitCandidate};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheConfig {
    pub enabled: bool,
    pub node_statistics: bool,
    pub predicate_masks: bool,
    pub candidate_pools: bool,
    pub best_subtrees: bool,
    pub lookahead: bool,
    pub max_entries: usize,
    pub approximate_byte_limit: usize,
}

impl CacheConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            node_statistics: false,
            predicate_masks: false,
            candidate_pools: false,
            best_subtrees: false,
            lookahead: false,
            max_entries: 0,
            approximate_byte_limit: 0,
        }
    }

    pub fn all_enabled() -> Self {
        Self {
            enabled: true,
            node_statistics: true,
            predicate_masks: true,
            candidate_pools: true,
            best_subtrees: true,
            lookahead: true,
            max_entries: 20_000,
            approximate_byte_limit: 256 * 1024 * 1024,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            predicate_masks: true,
            node_statistics: false,
            candidate_pools: false,
            best_subtrees: false,
            lookahead: false,
            max_entries: 20_000,
            approximate_byte_limit: 256 * 1024 * 1024,
        }
    }
}

/// Equality checks include the full row words and full configuration keys.
/// Numeric IDs accelerate diagnostics only and cannot cause a false match.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SearchStateKey {
    pub row_mask_words: Arc<[u64]>,
    pub depth_remaining: u16,
    pub node_budget: u16,
    pub theory_state: PathTheoryState,
    pub scoring_config_id: u64,
    pub candidate_config_id: u64,
    pub scoring_config_key: Arc<str>,
    pub candidate_config_key: Arc<str>,
}

impl SearchStateKey {
    pub fn new(
        rows: &BitSet,
        depth_remaining: usize,
        node_budget: usize,
        theory_state: PathTheoryState,
        scoring_config_key: impl Into<Arc<str>>,
        candidate_config_key: impl Into<Arc<str>>,
    ) -> Self {
        let scoring_config_key = scoring_config_key.into();
        let candidate_config_key = candidate_config_key.into();
        Self {
            row_mask_words: Arc::from(rows.words().to_vec()),
            depth_remaining: depth_remaining.min(u16::MAX as usize) as u16,
            node_budget: node_budget.min(u16::MAX as usize) as u16,
            theory_state,
            scoring_config_id: stable_id(&scoring_config_key),
            candidate_config_id: stable_id(&candidate_config_key),
            scoring_config_key,
            candidate_config_key,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeStatistics {
    pub sample_count: usize,
    pub class_counts: Vec<usize>,
    pub majority_class: u32,
}

#[derive(Clone, Debug)]
pub struct CachedSubtree {
    pub tree: Arc<TreeNode>,
    pub training_error: f64,
    pub validation_error: Option<f64>,
    pub node_count: usize,
    pub leaf_count: usize,
    pub literal_count: usize,
    pub estimated_axp_length: Option<f64>,
    pub objective: f64,
    pub path_certified: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheLevelDiagnostics {
    pub hits: usize,
    pub misses: usize,
    pub insertions: usize,
    pub evictions: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheDiagnostics {
    pub node_statistics: CacheLevelDiagnostics,
    pub predicate_masks: CacheLevelDiagnostics,
    pub candidate_pools: CacheLevelDiagnostics,
    pub best_subtrees: CacheLevelDiagnostics,
    pub lookahead: CacheLevelDiagnostics,
    pub approximate_memory_bytes: usize,
}

#[derive(Clone, Debug)]
struct CacheEntry<V> {
    value: V,
    last_access: u64,
    approximate_bytes: usize,
}

/// Deterministic bounded LRU map. Callers provide conservative byte estimates.
#[derive(Clone, Debug)]
pub struct BoundedCache<K, V> {
    entries: BTreeMap<K, CacheEntry<V>>,
    clock: u64,
    approximate_bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl<K: Ord + Clone, V: Clone> BoundedCache<K, V> {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            clock: 0,
            approximate_bytes: 0,
            max_entries,
            max_bytes,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        self.clock = self.clock.wrapping_add(1);
        let entry = self.entries.get_mut(key)?;
        entry.last_access = self.clock;
        Some(entry.value.clone())
    }

    pub fn insert(&mut self, key: K, value: V, approximate_bytes: usize) -> usize {
        self.clock = self.clock.wrapping_add(1);
        if let Some(previous) = self.entries.remove(&key) {
            self.approximate_bytes = self
                .approximate_bytes
                .saturating_sub(previous.approximate_bytes);
        }
        self.approximate_bytes = self.approximate_bytes.saturating_add(approximate_bytes);
        self.entries.insert(
            key,
            CacheEntry {
                value,
                last_access: self.clock,
                approximate_bytes,
            },
        );
        let mut evictions = 0;
        while self.entries.len() > self.max_entries || self.approximate_bytes > self.max_bytes {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(key, entry)| (entry.last_access, *key))
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.approximate_bytes = self
                    .approximate_bytes
                    .saturating_sub(removed.approximate_bytes);
                evictions += 1;
            }
        }
        evictions
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn approximate_bytes(&self) -> usize {
        self.approximate_bytes
    }
}

pub type NodeStatisticsCache = BoundedCache<SearchStateKey, NodeStatistics>;
pub type CandidatePoolCache = BoundedCache<SearchStateKey, Vec<SplitCandidate>>;
pub type BestSubtreeCache = BoundedCache<SearchStateKey, CachedSubtree>;
pub type LookaheadCache = BoundedCache<SearchStateKey, f64>;

fn stable_id(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}
