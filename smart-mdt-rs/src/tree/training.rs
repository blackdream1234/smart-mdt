//! Incremental bitset-backed training views and candidate statistics.

use super::LanguagePolicy;
use crate::{
    data::{is_boolean_column, predicate_mask, BitSet, Dataset},
    logic::{
        candidate_is_compatible, Literal, PathTheoryState, Predicate, ThresholdAtom, ThresholdOp,
    },
    search::{
        gini, information_gain, score_split, SplitCandidate, SplitScoreConfig, SplitScoreInput,
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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrainingDiagnostics {
    pub dataset_subset_allocations_avoided: usize,
    pub predicate_mask_cache_hits: usize,
    pub predicate_mask_cache_misses: usize,
    pub count_operations: usize,
    pub row_rescans_avoided: usize,
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
    pub predicate_mask_cache: RwLock<BTreeMap<String, Arc<BitSet>>>,
    diagnostics: AtomicTrainingDiagnostics,
}

impl TrainingContext {
    /// Creates one context for an entire fit. Recursive nodes share this root dataset.
    pub fn new(dataset: Arc<Dataset>) -> Self {
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

        Self {
            dataset,
            class_masks,
            boolean_column_masks,
            feature_domains,
            unary_literal_masks: RwLock::new(BTreeMap::new()),
            predicate_mask_cache: RwLock::new(BTreeMap::new()),
            diagnostics: AtomicTrainingDiagnostics::default(),
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
        if let Some(mask) = self
            .predicate_mask_cache
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .cloned()
        {
            self.diagnostics
                .predicate_mask_cache_hits
                .fetch_add(1, Ordering::Relaxed);
            self.diagnostics
                .row_rescans_avoided
                .fetch_add(self.dataset.labels.len(), Ordering::Relaxed);
            return mask;
        }

        self.diagnostics
            .predicate_mask_cache_misses
            .fetch_add(1, Ordering::Relaxed);
        let computed = Arc::new(predicate_mask(&self.dataset.features, predicate));
        let mut cache = self
            .predicate_mask_cache
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mask = cache
            .entry(key.clone())
            .or_insert_with(|| computed.clone())
            .clone();
        if matches!(predicate, Predicate::Unary(_)) {
            self.unary_literal_masks
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry(key)
                .or_insert_with(|| mask.clone());
        }
        mask
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
        }
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
