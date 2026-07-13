//! Incremental bitset-backed training views and candidate statistics.

use super::{
    BeamSearchDiagnostics, BestSubtreeCache, CacheConfig, CacheDiagnostics, CachedSubtree,
    CandidatePoolCache, LanguagePolicy, LookaheadCache, NodeStatistics, NodeStatisticsCache,
    SearchStateKey,
};
use crate::{
    data::{is_boolean_column, predicate_mask, BitSet, Dataset},
    logic::{
        candidate_is_compatible, Literal, PathTheoryState, Predicate, ThresholdAtom, ThresholdOp,
    },
    search::{
        gini, information_gain, score_split, BranchAndBoundDiagnostics, SplitCandidate,
        SplitScoreConfig, SplitScoreInput,
    },
    ClassId, FeatureId, Result,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
};

/// Immutable row-mask view of one recursive training node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeView {
    pub rows: BitSet,
    pub depth: usize,
    pub theory_state: PathTheoryState,
}

impl NodeView {
    pub fn root(dataset: &Dataset) -> Self {
        Self {
            rows: BitSet::ones(dataset.labels.len()),
            depth: 0,
            theory_state: PathTheoryState::Uncommitted,
        }
    }
}

/// Snapshot of allocation and incremental-statistics counters.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainingDiagnostics {
    pub dataset_subset_allocations_avoided: usize,
    pub predicate_mask_cache_hits: usize,
    pub predicate_mask_cache_misses: usize,
    pub count_operations: usize,
    pub row_rescans_avoided: usize,
    pub branch_and_bound: BranchAndBoundDiagnostics,
    pub cache: CacheDiagnostics,
    pub beam_search: BeamSearchDiagnostics,
}

#[derive(Debug, Default)]
struct AtomicTrainingDiagnostics {
    dataset_subset_allocations_avoided: AtomicUsize,
    predicate_mask_cache_hits: AtomicUsize,
    predicate_mask_cache_misses: AtomicUsize,
    count_operations: AtomicUsize,
    row_rescans_avoided: AtomicUsize,
}

/// Per-fit immutable dataset plus reusable masks and incremental statistics.
#[derive(Debug)]
pub struct TrainingContext {
    pub dataset: Arc<Dataset>,
    pub class_masks: Vec<BitSet>,
    pub boolean_column_masks: Vec<Option<BitSet>>,
    pub feature_domains: Vec<Vec<f64>>,
    pub unary_literal_masks: RwLock<BTreeMap<String, Arc<BitSet>>>,
    pub predicate_mask_cache: RwLock<super::BoundedCache<String, Arc<BitSet>>>,
    pub cache_config: CacheConfig,
    node_statistics_cache: RwLock<NodeStatisticsCache>,
    candidate_pool_cache: RwLock<CandidatePoolCache>,
    best_subtree_cache: RwLock<BestSubtreeCache>,
    lookahead_cache: RwLock<LookaheadCache>,
    cache_diagnostics: RwLock<CacheDiagnostics>,
    diagnostics: AtomicTrainingDiagnostics,
    branch_and_bound_diagnostics: RwLock<BranchAndBoundDiagnostics>,
    beam_search_diagnostics: RwLock<BeamSearchDiagnostics>,
}

impl TrainingContext {
    /// Creates one context for an entire fit. Recursive nodes share this root dataset.
    pub fn new(dataset: Arc<Dataset>) -> Self {
        Self::with_cache_config(dataset, CacheConfig::default())
    }

    pub fn with_cache_config(dataset: Arc<Dataset>, cache_config: CacheConfig) -> Self {
        let classes = dataset.class_count().max(2);
        let mut class_masks = vec![BitSet::zeros(dataset.labels.len()); classes];
        for (row, &class) in dataset.labels.iter().enumerate() {
            class_masks[class as usize].set(row, true);
        }

        let mut boolean_column_masks = Vec::with_capacity(dataset.features.cols());
        let mut feature_domains = Vec::with_capacity(dataset.features.cols());
        for feature in 0..dataset.features.cols() as FeatureId {
            let mut values = dataset.features.column(feature).to_vec();
            values.sort_by(f64::total_cmp);
            values.dedup();
            feature_domains.push(values);
            if is_boolean_column(&dataset.features, feature) {
                let mut mask = BitSet::zeros(dataset.labels.len());
                for (row, &value) in dataset.features.column(feature).iter().enumerate() {
                    mask.set(row, value == 1.0);
                }
                boolean_column_masks.push(Some(mask));
            } else {
                boolean_column_masks.push(None);
            }
        }

        let max_entries = cache_config.max_entries;
        let max_bytes = cache_config.approximate_byte_limit;
        Self {
            dataset,
            class_masks,
            boolean_column_masks,
            feature_domains,
            unary_literal_masks: RwLock::new(BTreeMap::new()),
            predicate_mask_cache: RwLock::new(super::BoundedCache::new(max_entries, max_bytes)),
            cache_config,
            node_statistics_cache: RwLock::new(NodeStatisticsCache::new(max_entries, max_bytes)),
            candidate_pool_cache: RwLock::new(CandidatePoolCache::new(max_entries, max_bytes)),
            best_subtree_cache: RwLock::new(BestSubtreeCache::new(max_entries, max_bytes)),
            lookahead_cache: RwLock::new(LookaheadCache::new(max_entries, max_bytes)),
            cache_diagnostics: RwLock::new(CacheDiagnostics::default()),
            diagnostics: AtomicTrainingDiagnostics::default(),
            branch_and_bound_diagnostics: RwLock::new(BranchAndBoundDiagnostics::default()),
            beam_search_diagnostics: RwLock::new(BeamSearchDiagnostics::default()),
        }
    }

    pub fn root_view(&self) -> NodeView {
        NodeView::root(&self.dataset)
    }

    pub fn sample_count(&self, node: &NodeView) -> usize {
        self.diagnostics
            .count_operations
            .fetch_add(1, Ordering::Relaxed);
        node.rows.count_ones()
    }

    pub fn class_counts(&self, node: &NodeView) -> Result<Vec<usize>> {
        self.diagnostics
            .count_operations
            .fetch_add(self.class_masks.len(), Ordering::Relaxed);
        self.diagnostics
            .row_rescans_avoided
            .fetch_add(node.rows.count_ones(), Ordering::Relaxed);
        self.class_masks
            .iter()
            .map(|class| node.rows.intersection_count(class))
            .collect()
    }

    pub fn majority_class(&self, node: &NodeView) -> Result<ClassId> {
        Ok(self
            .class_counts(node)?
            .into_iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .map_or(0, |(class, _)| class as ClassId))
    }

    /// Returns a full-dataset predicate mask, computing it only on the first use.
    pub fn full_predicate_mask(&self, predicate: &Predicate) -> Arc<BitSet> {
        let key = predicate_key(predicate);
        let cache_enabled = self.cache_config.enabled && self.cache_config.predicate_masks;
        if cache_enabled {
            if let Some(mask) = self
                .predicate_mask_cache
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&key)
            {
                self.diagnostics
                    .predicate_mask_cache_hits
                    .fetch_add(1, Ordering::Relaxed);
                self.diagnostics
                    .row_rescans_avoided
                    .fetch_add(self.dataset.labels.len(), Ordering::Relaxed);
                self.update_cache_diagnostics(|diagnostics| {
                    diagnostics.predicate_masks.hits += 1;
                });
                return mask;
            }
        }

        self.diagnostics
            .predicate_mask_cache_misses
            .fetch_add(1, Ordering::Relaxed);
        self.update_cache_diagnostics(|diagnostics| {
            diagnostics.predicate_masks.misses += 1;
        });
        let computed = Arc::new(predicate_mask(&self.dataset.features, predicate));
        if cache_enabled {
            let approximate_bytes = key.len() + std::mem::size_of_val(computed.words());
            let evictions = self
                .predicate_mask_cache
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(key.clone(), computed.clone(), approximate_bytes);
            self.update_cache_diagnostics(|diagnostics| {
                diagnostics.predicate_masks.insertions += 1;
                diagnostics.predicate_masks.evictions += evictions;
            });
            self.refresh_cache_memory();
        }
        if matches!(predicate, Predicate::Unary(_)) {
            self.unary_literal_masks
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry(key)
                .or_insert_with(|| computed.clone());
        }
        computed
    }

    pub fn predicate_mask(&self, node: &NodeView, predicate: &Predicate) -> Result<BitSet> {
        node.rows.and(&self.full_predicate_mask(predicate))
    }

    pub fn split_masks(&self, node: &NodeView, predicate: &Predicate) -> Result<(BitSet, BitSet)> {
        let full = self.full_predicate_mask(predicate);
        Ok((node.rows.and(&full)?, node.rows.and_not(&full)?))
    }

    pub fn child_class_counts(&self, child_rows: &BitSet) -> Result<Vec<usize>> {
        self.diagnostics
            .count_operations
            .fetch_add(self.class_masks.len(), Ordering::Relaxed);
        self.class_masks
            .iter()
            .map(|class| child_rows.intersection_count(class))
            .collect()
    }

    pub fn balance(&self, true_rows: &BitSet, false_rows: &BitSet) -> f64 {
        self.diagnostics
            .count_operations
            .fetch_add(2, Ordering::Relaxed);
        let left = true_rows.count_ones();
        let right = false_rows.count_ones();
        left.min(right) as f64 / (left + right).max(1) as f64
    }

    pub fn record_child_views(&self) {
        self.diagnostics
            .dataset_subset_allocations_avoided
            .fetch_add(2, Ordering::Relaxed);
    }

    pub fn diagnostics(&self) -> TrainingDiagnostics {
        TrainingDiagnostics {
            dataset_subset_allocations_avoided: self
                .diagnostics
                .dataset_subset_allocations_avoided
                .load(Ordering::Relaxed),
            predicate_mask_cache_hits: self
                .diagnostics
                .predicate_mask_cache_hits
                .load(Ordering::Relaxed),
            predicate_mask_cache_misses: self
                .diagnostics
                .predicate_mask_cache_misses
                .load(Ordering::Relaxed),
            count_operations: self.diagnostics.count_operations.load(Ordering::Relaxed),
            row_rescans_avoided: self.diagnostics.row_rescans_avoided.load(Ordering::Relaxed),
            branch_and_bound: self
                .branch_and_bound_diagnostics
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone(),
            cache: self.cache_diagnostics(),
            beam_search: self
                .beam_search_diagnostics
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone(),
        }
    }

    pub fn record_beam_search(&self, diagnostics: BeamSearchDiagnostics) {
        *self
            .beam_search_diagnostics
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = diagnostics;
    }

    pub fn record_branch_and_bound(&self, current: &BranchAndBoundDiagnostics) {
        let mut total = self
            .branch_and_bound_diagnostics
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        total.partial_states_created += current.partial_states_created;
        total.partial_states_expanded += current.partial_states_expanded;
        total.partial_states_pruned += current.partial_states_pruned;
        total.exhaustive_fallback_count += current.exhaustive_fallback_count;
        total.complete_candidates_evaluated += current.complete_candidates_evaluated;
        total.best_bound = total.best_bound.max(current.best_bound);
        total.kth_best_threshold = current.kth_best_threshold;
        total.percentage_reduction = if total.partial_states_created == 0 {
            0.0
        } else {
            100.0 * total.partial_states_pruned as f64 / total.partial_states_created as f64
        };
    }

    fn update_cache_diagnostics(&self, update: impl FnOnce(&mut CacheDiagnostics)) {
        update(
            &mut self
                .cache_diagnostics
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
    }

    fn refresh_cache_memory(&self) {
        let bytes = self
            .predicate_mask_cache
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .approximate_bytes()
            + self
                .node_statistics_cache
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .approximate_bytes()
            + self
                .candidate_pool_cache
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .approximate_bytes()
            + self
                .best_subtree_cache
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .approximate_bytes()
            + self
                .lookahead_cache
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .approximate_bytes();
        self.update_cache_diagnostics(|diagnostics| {
            diagnostics.approximate_memory_bytes = bytes;
        });
    }

    pub fn cache_diagnostics(&self) -> CacheDiagnostics {
        self.cache_diagnostics
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn node_statistics_cached(
        &self,
        key: &SearchStateKey,
        node: &NodeView,
    ) -> Result<NodeStatistics> {
        let enabled = self.cache_config.enabled && self.cache_config.node_statistics;
        if enabled {
            if let Some(statistics) = self
                .node_statistics_cache
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(key)
            {
                self.update_cache_diagnostics(|diagnostics| {
                    diagnostics.node_statistics.hits += 1;
                });
                return Ok(statistics);
            }
        }
        self.update_cache_diagnostics(|diagnostics| {
            diagnostics.node_statistics.misses += 1;
        });
        let class_counts = self.class_counts(node)?;
        let statistics = NodeStatistics {
            sample_count: node.rows.count_ones(),
            majority_class: class_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, count)| **count)
                .map_or(0, |(class, _)| class as ClassId),
            class_counts,
        };
        if enabled {
            let bytes = key.row_mask_words.len() * std::mem::size_of::<u64>()
                + statistics.class_counts.len() * std::mem::size_of::<usize>()
                + 64;
            let evictions = self
                .node_statistics_cache
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(key.clone(), statistics.clone(), bytes);
            self.update_cache_diagnostics(|diagnostics| {
                diagnostics.node_statistics.insertions += 1;
                diagnostics.node_statistics.evictions += evictions;
            });
            self.refresh_cache_memory();
        }
        Ok(statistics)
    }

    pub fn candidate_pool_cached(&self, key: &SearchStateKey) -> Option<Vec<SplitCandidate>> {
        if !(self.cache_config.enabled && self.cache_config.candidate_pools) {
            self.update_cache_diagnostics(|diagnostics| {
                diagnostics.candidate_pools.misses += 1;
            });
            return None;
        }
        let found = self
            .candidate_pool_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(key);
        self.update_cache_diagnostics(|diagnostics| {
            if found.is_some() {
                diagnostics.candidate_pools.hits += 1;
            } else {
                diagnostics.candidate_pools.misses += 1;
            }
        });
        found
    }

    pub fn insert_candidate_pool(&self, key: SearchStateKey, candidates: Vec<SplitCandidate>) {
        if !(self.cache_config.enabled && self.cache_config.candidate_pools) {
            return;
        }
        let bytes = key.row_mask_words.len() * std::mem::size_of::<u64>()
            + candidates.len() * std::mem::size_of::<SplitCandidate>();
        let evictions = self
            .candidate_pool_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, candidates, bytes);
        self.update_cache_diagnostics(|diagnostics| {
            diagnostics.candidate_pools.insertions += 1;
            diagnostics.candidate_pools.evictions += evictions;
        });
        self.refresh_cache_memory();
    }

    pub fn best_subtree_cached(&self, key: &SearchStateKey) -> Option<CachedSubtree> {
        if !(self.cache_config.enabled && self.cache_config.best_subtrees) {
            self.update_cache_diagnostics(|diagnostics| {
                diagnostics.best_subtrees.misses += 1;
            });
            return None;
        }
        let found = self
            .best_subtree_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(key);
        self.update_cache_diagnostics(|diagnostics| {
            if found.is_some() {
                diagnostics.best_subtrees.hits += 1;
            } else {
                diagnostics.best_subtrees.misses += 1;
            }
        });
        found
    }

    pub fn insert_best_subtree(&self, key: SearchStateKey, subtree: CachedSubtree) {
        if !(self.cache_config.enabled && self.cache_config.best_subtrees) {
            return;
        }
        let bytes = key.row_mask_words.len() * std::mem::size_of::<u64>()
            + subtree.node_count * std::mem::size_of::<crate::tree::TreeNode>();
        let evictions = self
            .best_subtree_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, subtree, bytes);
        self.update_cache_diagnostics(|diagnostics| {
            diagnostics.best_subtrees.insertions += 1;
            diagnostics.best_subtrees.evictions += evictions;
        });
        self.refresh_cache_memory();
    }

    pub fn lookahead_cached(&self, key: &SearchStateKey) -> Option<f64> {
        if !(self.cache_config.enabled && self.cache_config.lookahead) {
            self.update_cache_diagnostics(|diagnostics| {
                diagnostics.lookahead.misses += 1;
            });
            return None;
        }
        let found = self
            .lookahead_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(key);
        self.update_cache_diagnostics(|diagnostics| {
            if found.is_some() {
                diagnostics.lookahead.hits += 1;
            } else {
                diagnostics.lookahead.misses += 1;
            }
        });
        found
    }

    pub fn insert_lookahead(&self, key: SearchStateKey, objective: f64) {
        if !(self.cache_config.enabled && self.cache_config.lookahead) {
            return;
        }
        let bytes = key.row_mask_words.len() * std::mem::size_of::<u64>() + 64;
        let evictions = self
            .lookahead_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, objective, bytes);
        self.update_cache_diagnostics(|diagnostics| {
            diagnostics.lookahead.insertions += 1;
            diagnostics.lookahead.evictions += evictions;
        });
        self.refresh_cache_memory();
    }

    fn node_values(&self, node: &NodeView, feature: FeatureId) -> Vec<f64> {
        let mut values = Vec::new();
        for row in 0..node.rows.len() {
            if node.rows.get(row) {
                values.push(self.dataset.features.get(row, feature));
            }
        }
        values.sort_by(f64::total_cmp);
        values.dedup();
        values
    }

    fn score_candidate(
        &self,
        node: &NodeView,
        predicate: Predicate,
        score_config: &SplitScoreConfig,
    ) -> Result<Option<(SplitCandidate, BitSet)>> {
        let (true_rows, false_rows) = self.split_masks(node, &predicate)?;
        let left = true_rows.count_ones();
        let right = false_rows.count_ones();
        if left == 0 || right == 0 {
            return Ok(None);
        }
        let parent_counts = self.class_counts(node)?;
        let left_counts = self.child_class_counts(&true_rows)?;
        let right_counts = self.child_class_counts(&false_rows)?;
        let gain = information_gain(&parent_counts, &left_counts, &right_counts);
        let total = left + right;
        let fragmentation = 1.0 - 2.0 * left.min(right) as f64 / total as f64;
        let estimated_subtree_cost =
            (left as f64 * gini(&left_counts) + right as f64 * gini(&right_counts)) / total as f64;
        let instability = (left as f64 / total as f64 - 0.5).abs() * 2.0;
        Ok(Some((
            SplitCandidate {
                score: score_split(
                    SplitScoreInput {
                        information_gain: gain,
                        true_count: left,
                        false_count: right,
                        literal_count: predicate.arity(),
                        family: predicate.language(),
                        fragmentation,
                        estimated_subtree_cost,
                        instability,
                    },
                    score_config,
                ),
                predicate,
                left_count: left,
                right_count: right,
            },
            true_rows,
        )))
    }

    /// Scores only candidates admissible under the node's current path theory.
    /// Incompatible and degenerate candidates return before numerical scoring.
    pub fn score_if_admissible(
        &self,
        node: &NodeView,
        predicate: Predicate,
        score_config: &SplitScoreConfig,
    ) -> Result<Option<crate::search::CandidateScore>> {
        if !candidate_is_compatible(node.theory_state, &predicate) {
            return Ok(None);
        }
        Ok(self
            .score_candidate(node, predicate, score_config)?
            .map(|(candidate, _)| candidate.score))
    }

    fn ranked_literals(
        &self,
        node: &NodeView,
        score_config: &SplitScoreConfig,
    ) -> Result<Vec<Literal>> {
        let mut literals = Vec::new();
        for feature in 0..self.dataset.features.cols() as FeatureId {
            let values = self.node_values(node, feature);
            for window in values.windows(2) {
                let atom = ThresholdAtom {
                    feature,
                    threshold_id: 0,
                    threshold: (window[0] + window[1]) / 2.0,
                    op: ThresholdOp::GreaterEqual,
                };
                literals.push(Literal {
                    atom,
                    positive: true,
                });
                literals.push(Literal {
                    atom,
                    positive: false,
                });
            }
        }
        let mut scored = Vec::with_capacity(literals.len());
        for literal in literals {
            let gain = self
                .score_candidate(node, Predicate::Unary(literal), score_config)?
                .map_or(f64::NEG_INFINITY, |(candidate, _)| {
                    candidate.score.predictive_gain
                });
            scored.push((literal, gain));
        }
        scored.sort_by(|(_, left), (_, right)| right.total_cmp(left));
        Ok(scored.into_iter().map(|(literal, _)| literal).collect())
    }

    fn generate_unary(
        &self,
        node: &NodeView,
        min_leaf: usize,
        score_config: &SplitScoreConfig,
    ) -> Result<Vec<SplitCandidate>> {
        let mut output = Vec::new();
        for feature in 0..self.dataset.features.cols() as FeatureId {
            let values = self.node_values(node, feature);
            for window in values.windows(2) {
                let predicate = Predicate::Unary(Literal {
                    atom: ThresholdAtom {
                        feature,
                        threshold_id: 0,
                        threshold: (window[0] + window[1]) / 2.0,
                        op: ThresholdOp::LessThan,
                    },
                    positive: true,
                });
                if let Some((candidate, _)) = self.score_candidate(node, predicate, score_config)? {
                    if candidate.left_count >= min_leaf && candidate.right_count >= min_leaf {
                        output.push(candidate);
                    }
                }
            }
        }
        Ok(output)
    }

    fn generate_clause_family(
        &self,
        node: &NodeView,
        min_leaf: usize,
        beam: usize,
        horn: bool,
        score_config: &SplitScoreConfig,
    ) -> Result<Vec<SplitCandidate>> {
        let selected: Vec<_> = self
            .ranked_literals(node, score_config)?
            .into_iter()
            .take(beam.max(2))
            .collect();
        let mut seen_masks = BTreeSet::new();
        let mut output = Vec::new();
        for first in 0..selected.len() {
            for second in first + 1..selected.len() {
                let a = selected[first];
                let b = selected[second];
                if same_atom_opposite_polarity(a, b) {
                    continue;
                }
                let literals = vec![a, b];
                let wrong_polarity = if horn {
                    literals.iter().filter(|literal| literal.positive).count() > 1
                } else {
                    literals.iter().filter(|literal| !literal.positive).count() > 1
                };
                if wrong_polarity {
                    continue;
                }
                let predicate = if horn {
                    Predicate::HornClause(literals)
                } else {
                    Predicate::AntiHornClause(literals)
                };
                if let Some((candidate, mask)) =
                    self.score_candidate(node, predicate, score_config)?
                {
                    if candidate.left_count < min_leaf || candidate.right_count < min_leaf {
                        continue;
                    }
                    if seen_masks.insert(mask.words().to_vec()) {
                        output.push(candidate);
                    }
                }
            }
        }
        output.sort_by(|a, b| b.score.final_score.total_cmp(&a.score.final_score));
        Ok(output)
    }

    fn generate_square2cnf(
        &self,
        node: &NodeView,
        min_leaf: usize,
        beam: usize,
        score_config: &SplitScoreConfig,
    ) -> Result<Vec<SplitCandidate>> {
        let selected: Vec<_> = self
            .ranked_literals(node, score_config)?
            .into_iter()
            .take(beam.max(4))
            .collect();
        let mut clauses = Vec::new();
        for first in 0..selected.len() {
            for second in first + 1..selected.len() {
                let a = selected[first];
                let b = selected[second];
                if !same_atom_opposite_polarity(a, b) {
                    let predicate = Predicate::Square2Cnf { a, b, c: a, d: b };
                    let gain = self
                        .score_candidate(node, predicate, score_config)?
                        .map_or(f64::NEG_INFINITY, |(candidate, _)| {
                            candidate.score.predictive_gain
                        });
                    clauses.push((a, b, gain));
                }
            }
        }
        clauses.sort_by(|(_, _, left), (_, _, right)| right.total_cmp(left));

        let mut seen_masks = BTreeSet::new();
        let mut output = Vec::new();
        for (index, &(a, b, _)) in clauses.iter().enumerate() {
            self.consider_square(
                node,
                min_leaf,
                Predicate::Square2Cnf { a, b, c: a, d: b },
                &mut seen_masks,
                &mut output,
                score_config,
            )?;
            for &(c, d, _) in clauses.iter().skip(index + 1) {
                self.consider_square(
                    node,
                    min_leaf,
                    Predicate::Square2Cnf { a, b, c, d },
                    &mut seen_masks,
                    &mut output,
                    score_config,
                )?;
            }
        }
        output.sort_by(|a, b| b.score.final_score.total_cmp(&a.score.final_score));
        Ok(output)
    }

    fn consider_square(
        &self,
        node: &NodeView,
        min_leaf: usize,
        predicate: Predicate,
        seen_masks: &mut BTreeSet<Vec<u64>>,
        output: &mut Vec<SplitCandidate>,
        score_config: &SplitScoreConfig,
    ) -> Result<()> {
        if let Some((candidate, mask)) = self.score_candidate(node, predicate, score_config)? {
            if candidate.left_count >= min_leaf
                && candidate.right_count >= min_leaf
                && seen_masks.insert(mask.words().to_vec())
            {
                output.push(candidate);
            }
        }
        Ok(())
    }

    fn generate_affine(
        &self,
        node: &NodeView,
        min_leaf: usize,
        beam: usize,
        score_config: &SplitScoreConfig,
    ) -> Result<Vec<SplitCandidate>> {
        let mut ranked = Vec::new();
        for feature in 0..self.dataset.features.cols() as FeatureId {
            if self.boolean_column_masks[feature as usize].is_none() {
                continue;
            }
            let predicate = Predicate::Unary(boolean_literal(feature));
            let gain = self
                .score_candidate(node, predicate, score_config)?
                .map_or(f64::NEG_INFINITY, |(candidate, _)| {
                    candidate.score.predictive_gain
                });
            ranked.push((feature, gain));
        }
        ranked.sort_by(|(left_feature, left), (right_feature, right)| {
            right.total_cmp(left).then(left_feature.cmp(right_feature))
        });
        let pool: Vec<_> = ranked
            .into_iter()
            .map(|(feature, _)| feature)
            .take(beam.max(3).max(2))
            .collect();
        let mut seen_masks = BTreeSet::new();
        let mut output = Vec::new();
        for arity in 2..=3 {
            for combination in combinations(pool.len(), arity) {
                let literals: Vec<_> = combination
                    .iter()
                    .map(|&index| boolean_literal(pool[index]))
                    .collect();
                for rhs in [false, true] {
                    let predicate = Predicate::Affine {
                        literals: literals.clone(),
                        rhs,
                    };
                    if let Some((candidate, mask)) =
                        self.score_candidate(node, predicate, score_config)?
                    {
                        if candidate.left_count >= min_leaf
                            && candidate.right_count >= min_leaf
                            && seen_masks.insert(mask.words().to_vec())
                        {
                            output.push(candidate);
                        }
                    }
                }
            }
        }
        output.sort_by(|a, b| b.score.final_score.total_cmp(&a.score.final_score));
        Ok(output)
    }

    /// Generates the same bounded per-node families without materializing a dataset subset.
    pub fn generate_candidates(
        &self,
        node: &NodeView,
        policy: LanguagePolicy,
        min_leaf: usize,
        beam: usize,
        score_config: &SplitScoreConfig,
    ) -> Result<Vec<SplitCandidate>> {
        let mut output = Vec::new();
        match policy {
            LanguagePolicy::UnaryOnly => {
                output.extend(self.generate_unary(node, min_leaf, score_config)?)
            }
            LanguagePolicy::HornOnly => output.extend(self.generate_clause_family(
                node,
                min_leaf,
                beam,
                true,
                score_config,
            )?),
            LanguagePolicy::AntiHornOnly => output.extend(self.generate_clause_family(
                node,
                min_leaf,
                beam,
                false,
                score_config,
            )?),
            LanguagePolicy::Square2CnfOnly => {
                output.extend(self.generate_square2cnf(node, min_leaf, beam, score_config)?)
            }
            LanguagePolicy::AffineOnly => {
                output.extend(self.generate_affine(node, min_leaf, beam, score_config)?)
            }
            LanguagePolicy::SmartCertified => {
                // Compatibility is enforced before family generation and therefore
                // before any candidate receives a numerical score.
                output.extend(self.generate_unary(node, min_leaf, score_config)?);
                match node.theory_state {
                    PathTheoryState::Uncommitted => {
                        output.extend(self.generate_clause_family(
                            node,
                            min_leaf,
                            beam,
                            true,
                            score_config,
                        )?);
                        output.extend(self.generate_clause_family(
                            node,
                            min_leaf,
                            beam,
                            false,
                            score_config,
                        )?);
                        output.extend(self.generate_square2cnf(
                            node,
                            min_leaf,
                            beam,
                            score_config,
                        )?);
                        output.extend(self.generate_affine(node, min_leaf, beam, score_config)?);
                    }
                    PathTheoryState::Horn => output.extend(self.generate_clause_family(
                        node,
                        min_leaf,
                        beam,
                        true,
                        score_config,
                    )?),
                    PathTheoryState::AntiHorn => output.extend(self.generate_clause_family(
                        node,
                        min_leaf,
                        beam,
                        false,
                        score_config,
                    )?),
                    PathTheoryState::TwoSat => output.extend(self.generate_square2cnf(
                        node,
                        min_leaf,
                        beam,
                        score_config,
                    )?),
                    PathTheoryState::AffineGf2 => {
                        output.extend(self.generate_affine(node, min_leaf, beam, score_config)?)
                    }
                }
            }
            LanguagePolicy::CertifiedOnly | LanguagePolicy::BestCertifiedPerNode => {
                output.extend(self.generate_unary(node, min_leaf, score_config)?);
                output.extend(self.generate_clause_family(
                    node,
                    min_leaf,
                    beam,
                    true,
                    score_config,
                )?);
                output.extend(self.generate_clause_family(
                    node,
                    min_leaf,
                    beam,
                    false,
                    score_config,
                )?);
                output.extend(self.generate_square2cnf(node, min_leaf, beam, score_config)?);
            }
            LanguagePolicy::EmpiricalMixed | LanguagePolicy::TunedExperimental => {
                output.extend(self.generate_unary(node, min_leaf, score_config)?);
            }
        }
        Ok(output)
    }
}

fn predicate_key(predicate: &Predicate) -> String {
    format!("{predicate:?}")
}

fn same_atom_opposite_polarity(a: Literal, b: Literal) -> bool {
    a.atom.feature == b.atom.feature
        && a.atom.threshold == b.atom.threshold
        && a.atom.op == b.atom.op
        && a.positive != b.positive
}

fn boolean_literal(feature: FeatureId) -> Literal {
    Literal {
        atom: ThresholdAtom {
            feature,
            threshold_id: 0,
            threshold: 0.5,
            op: ThresholdOp::GreaterEqual,
        },
        positive: true,
    }
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut output = Vec::new();
    if k == 0 || k > n {
        return output;
    }
    let mut indices: Vec<usize> = (0..k).collect();
    loop {
        output.push(indices.clone());
        let mut index = k - 1;
        while indices[index] == index + n - k {
            if index == 0 {
                return output;
            }
            index -= 1;
        }
        indices[index] += 1;
        for next in index + 1..k {
            indices[next] = indices[next - 1] + 1;
        }
    }
}
