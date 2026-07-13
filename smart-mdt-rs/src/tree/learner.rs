use super::{
    deterministic_pruning_split, predict_row, prune_with_validation, AdaptiveLanguageConfig,
    BeamSearchDiagnostics, CacheConfig, CachedSubtree, CandidateGenerationConfig, FrontierLeaf,
    NodeView, ParallelConfig, PartialTree, PartialTreeState, PruningConfig, SearchStateKey,
    TrainingContext, TrainingDiagnostics, TreeNode, TreeSearchConfig, TreeSearchStrategy,
};
use crate::{
    data::Dataset,
    logic::{candidate_is_compatible, next_theory_state, PathTheoryState},
    search::{
        exact_branch_and_bound_top_k, BranchAndBoundConfig, SplitCandidate, SplitScoreConfig,
    },
    ClassId, Result, SmartMdtError,
};
use std::{cmp::Ordering, sync::Arc, time::Instant};
/// Language policy for learner search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanguagePolicy {
    CertifiedOnly,
    HornOnly,
    AntiHornOnly,
    Square2CnfOnly,
    AffineOnly,
    UnaryOnly,
    /// Chooses among all certified families while preserving one tractable
    /// theory on every root-to-leaf path.
    SmartCertified,
    BestCertifiedPerNode,
    EmpiricalMixed,
    TunedExperimental,
}
/// Learner configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnerConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_candidates_per_node: usize,
    pub beam_width: usize,
    pub split_score: SplitScoreConfig,
    pub branch_and_bound: BranchAndBoundConfig,
    pub cache: CacheConfig,
    pub tree_search: TreeSearchConfig,
    pub parallel: ParallelConfig,
    pub pruning: PruningConfig,
    pub adaptive_language: AdaptiveLanguageConfig,
    pub language_policy: LanguagePolicy,
    pub theorem_mode: bool,
    pub random_seed: u64,
}
impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_candidates_per_node: 64,
            beam_width: 8,
            split_score: SplitScoreConfig::default(),
            branch_and_bound: BranchAndBoundConfig::default(),
            cache: CacheConfig::default(),
            tree_search: TreeSearchConfig::default(),
            parallel: ParallelConfig::default(),
            pruning: PruningConfig::default(),
            adaptive_language: AdaptiveLanguageConfig::default(),
            language_policy: LanguagePolicy::BestCertifiedPerNode,
            theorem_mode: true,
            random_seed: 42,
        }
    }
}
/// Learns a CGS-MDT tree.
pub fn learn(data: &Dataset, cfg: &LearnerConfig) -> Result<TreeNode> {
    learn_with_diagnostics(data, cfg).map(|(tree, _)| tree)
}

/// Learns a tree and returns incremental-training diagnostics for this fit.
pub fn learn_with_diagnostics(
    data: &Dataset,
    cfg: &LearnerConfig,
) -> Result<(TreeNode, TrainingDiagnostics)> {
    if cfg.theorem_mode
        && matches!(
            cfg.language_policy,
            LanguagePolicy::EmpiricalMixed | LanguagePolicy::TunedExperimental
        )
    {
        return Err(SmartMdtError::TheoremRejected(
            "empirical policy in theorem mode".into(),
        ));
    }
    let pruning_split = if cfg.pruning.enabled {
        deterministic_pruning_split(data, cfg.pruning.validation_fraction, cfg.random_seed).ok()
    } else {
        None
    };
    let grow_data = pruning_split
        .as_ref()
        .map_or_else(|| data.clone(), |split| split.grow.clone());
    let context = TrainingContext::with_cache_config(Arc::new(grow_data), cfg.cache.clone());
    let grown_tree = match cfg.tree_search.strategy {
        TreeSearchStrategy::Greedy => build(&context, cfg, context.root_view())?,
        TreeSearchStrategy::SparseLookahead | TreeSearchStrategy::GlobalBeam => {
            build_with_tree_search(&context, cfg)?
        }
    };
    let tree = if let Some(split) = pruning_split {
        let (tree, mut diagnostics) =
            prune_with_validation(&grown_tree, &split.validation, &cfg.pruning);
        diagnostics.grow_samples = split.grow.labels.len();
        diagnostics.validation_samples = split.validation.labels.len();
        diagnostics.validation_indices = split.validation_indices;
        context.record_pruning(diagnostics);
        tree
    } else {
        grown_tree
    };
    context.record_selected_tree(&tree);
    Ok((tree, context.diagnostics()))
}
fn candidates(
    context: &TrainingContext,
    node: &NodeView,
    state_key: &SearchStateKey,
    cfg: &LearnerConfig,
) -> Result<Vec<SplitCandidate>> {
    if let Some(cached) = context.candidate_pool_cached(state_key) {
        return Ok(cached);
    }
    let mut generated = context.generate_candidates_adaptive(
        node,
        CandidateGenerationConfig {
            policy: cfg.language_policy,
            min_leaf: cfg.min_samples_leaf,
            beam: candidate_generation_width(cfg),
            score: &cfg.split_score,
            parallel: &cfg.parallel,
            adaptive: &cfg.adaptive_language,
        },
    )?;
    if cfg.language_policy == LanguagePolicy::SmartCertified {
        generated
            .retain(|candidate| candidate_is_compatible(node.theory_state, &candidate.predicate));
    }
    let mut branch_config = cfg.branch_and_bound.clone();
    branch_config.top_k = if branch_config.enabled {
        branch_config.top_k.min(cfg.max_candidates_per_node)
    } else {
        cfg.max_candidates_per_node
    };
    let (selected, diagnostics) =
        exact_branch_and_bound_top_k(generated, &branch_config, &cfg.split_score, |predicate| {
            Arc::<[u64]>::from(context.full_predicate_mask(predicate).words().to_vec())
        });
    context.record_branch_and_bound(&diagnostics);
    context.insert_candidate_pool(state_key.clone(), selected.clone());
    Ok(selected)
}
fn build(context: &TrainingContext, cfg: &LearnerConfig, node: NodeView) -> Result<TreeNode> {
    let state_key = search_state_key(&node, cfg, cfg.tree_search.node_budget);
    if let Some(cached) = context.best_subtree_cached(&state_key) {
        return Ok((*cached.tree).clone());
    }
    let statistics = context.node_statistics_cached(&state_key, &node)?;
    let sample_count = statistics.sample_count;
    let class_counts = statistics.class_counts;
    let majority_class = statistics.majority_class;
    if node.depth >= cfg.max_depth
        || sample_count < cfg.min_samples_split
        || class_counts.iter().filter(|&&count| count > 0).count() <= 1
    {
        let tree = TreeNode::Leaf {
            class: majority_class,
            samples: sample_count,
        };
        cache_subtree(context, &node, state_key, &tree);
        return Ok(tree);
    }
    let cand = candidates(context, &node, &state_key, cfg)?;
    let Some(best) = cand.into_iter().next() else {
        let tree = TreeNode::Leaf {
            class: majority_class,
            samples: sample_count,
        };
        cache_subtree(context, &node, state_key, &tree);
        return Ok(tree);
    };
    if cfg.theorem_mode && !best.predicate.language().theorem_table_allowed() {
        return Err(SmartMdtError::TheoremRejected(
            "non-certified predicate selected".into(),
        ));
    }
    let (left_rows, right_rows) = context.split_masks(&node, &best.predicate)?;
    let child_state = if cfg.language_policy == LanguagePolicy::SmartCertified {
        next_theory_state(node.theory_state, &best.predicate)?
    } else {
        node.theory_state
    };
    // Each recursive call receives its own copy, so descendants of one branch
    // cannot commit the sibling branch to a theory.
    context.record_child_views();
    let left = build(
        context,
        cfg,
        NodeView {
            rows: left_rows,
            depth: node.depth + 1,
            theory_state: child_state,
        },
    )?;
    let right = build(
        context,
        cfg,
        NodeView {
            rows: right_rows,
            depth: node.depth + 1,
            theory_state: child_state,
        },
    )?;
    let tree = TreeNode::Internal {
        predicate: best.predicate,
        left: Box::new(left),
        right: Box::new(right),
        majority_class,
    };
    cache_subtree(context, &node, state_key, &tree);
    Ok(tree)
}

fn candidate_generation_width(cfg: &LearnerConfig) -> usize {
    match cfg.tree_search.strategy {
        TreeSearchStrategy::Greedy => cfg.beam_width,
        TreeSearchStrategy::SparseLookahead | TreeSearchStrategy::GlobalBeam => {
            cfg.tree_search.candidate_beam_width
        }
    }
}

fn search_state_key(node: &NodeView, cfg: &LearnerConfig, node_budget: usize) -> SearchStateKey {
    SearchStateKey::new(
        &node.rows,
        cfg.max_depth.saturating_sub(node.depth),
        node_budget,
        node.theory_state,
        format!("{:?}", cfg.split_score),
        format!(
            "policy={:?};min_split={};min_leaf={};candidate_cap={};candidate_beam={};branch={:?};tree_search={:?};parallel={:?};pruning={:?};adaptive={:?}",
            cfg.language_policy,
            cfg.min_samples_split,
            cfg.min_samples_leaf,
            cfg.max_candidates_per_node,
            candidate_generation_width(cfg),
            cfg.branch_and_bound,
            cfg.tree_search,
            cfg.parallel,
            cfg.pruning,
            cfg.adaptive_language,
        ),
    )
}

fn build_with_tree_search(context: &TrainingContext, cfg: &LearnerConfig) -> Result<TreeNode> {
    // A complete greedy tree is the deterministic anytime incumbent. This also
    // guarantees that widening the tree beam cannot worsen the configured
    // complete-tree training objective.
    let greedy = build(context, cfg, context.root_view())?;
    let root = context.root_view();
    let root_counts = context.class_counts(&root)?;
    let root_majority = majority(&root_counts);
    let root_leaf = PartialTree::Leaf {
        node_id: 0,
        class: root_majority,
        samples: root.rows.count_ones(),
    };
    let root_terminal =
        node_is_terminal(context, cfg, &root, &root_counts) || cfg.tree_search.node_budget < 3;
    let root_mistakes = majority_mistakes(&root_counts);
    let root_tree: TreeNode = root_leaf.clone().into();
    let mut best = if greedy.nodes() <= cfg.tree_search.node_budget {
        (complete_tree_objective(context, &greedy), greedy)
    } else {
        (complete_tree_objective(context, &root_tree), root_tree)
    };
    if cfg.tree_search.tree_beam_width == 1
        && cfg.tree_search.node_budget == usize::MAX
        && cfg.tree_search.max_expansions == usize::MAX
        && cfg.tree_search.time_budget_ms.is_none()
    {
        context.record_beam_search(BeamSearchDiagnostics {
            states_generated: 1,
            completed_states: 1,
            maximum_live_states: 1,
            ..BeamSearchDiagnostics::default()
        });
        return Ok(best.1);
    }
    let mut beam = vec![PartialTreeState {
        tree: root_leaf,
        frontier: if root_terminal {
            Vec::new()
        } else {
            vec![FrontierLeaf {
                node_id: 0,
                rows: root.rows,
                depth: 0,
                theory_state: PathTheoryState::Uncommitted,
                majority_class: root_majority,
            }]
        },
        training_error_lower_bound: if root_terminal {
            root_mistakes as f64 / context.dataset.labels.len().max(1) as f64
        } else {
            0.0
        },
        complexity_cost: 0.0,
        objective_lower_bound: if root_terminal {
            root_mistakes as f64 / context.dataset.labels.len().max(1) as f64
        } else {
            0.0
        },
        expanded_nodes: 0,
        generated_order: 0,
    }];
    let started = Instant::now();
    let mut generated_order = 1u64;
    let mut next_node_id = 1usize;
    let mut expansions = 0usize;
    let mut sparse_greedy = false;
    let mut diagnostics = BeamSearchDiagnostics {
        states_generated: 1,
        maximum_live_states: 1,
        ..BeamSearchDiagnostics::default()
    };

    loop {
        for state in &beam {
            if state.frontier.is_empty() {
                diagnostics.completed_states += 1;
                consider_partial_incumbent(context, state, &mut best);
            }
        }
        if beam.iter().all(|state| state.frontier.is_empty()) {
            break;
        }
        if expansions >= cfg.tree_search.max_expansions {
            diagnostics.expansion_budget_reached = true;
            break;
        }
        if cfg
            .tree_search
            .time_budget_ms
            .is_some_and(|limit| started.elapsed().as_millis() >= u128::from(limit))
        {
            diagnostics.time_budget_reached = true;
            break;
        }

        let mut next = Vec::new();
        for state in beam {
            if state.frontier.is_empty() {
                next.push(state);
                continue;
            }
            if expansions >= cfg.tree_search.max_expansions {
                next.push(state);
                continue;
            }
            let frontier_index = select_frontier(context, cfg, &state)?;
            let frontier = state.frontier[frontier_index].clone();
            if cfg.tree_search.strategy == TreeSearchStrategy::SparseLookahead
                && frontier.depth >= cfg.tree_search.lookahead_depth
            {
                sparse_greedy = true;
            }
            let node = NodeView {
                rows: frontier.rows.clone(),
                depth: frontier.depth,
                theory_state: frontier.theory_state,
            };
            let remaining_budget = cfg
                .tree_search
                .node_budget
                .saturating_sub(state.tree.nodes());
            let key = search_state_key(&node, cfg, remaining_budget);
            let mut ranked = candidates(context, &node, &key, cfg)?;
            let before_filter = ranked.len();
            ranked.retain(|candidate| {
                candidate_is_compatible(frontier.theory_state, &candidate.predicate)
            });
            diagnostics.path_incompatible_candidates_rejected += before_filter - ranked.len();
            let candidate_width = if sparse_greedy {
                1
            } else {
                cfg.tree_search.candidate_beam_width.max(1)
            };
            ranked.truncate(candidate_width);
            expansions += 1;
            diagnostics.states_expanded += 1;

            if ranked.is_empty()
                || state.tree.nodes().saturating_add(2) > cfg.tree_search.node_budget
            {
                let mut terminal = state;
                terminal.frontier.remove(frontier_index);
                add_completed_leaf_error(context, &frontier.rows, &mut terminal);
                next.push(terminal);
                continue;
            }

            for candidate in ranked {
                let child_state = next_theory_state(frontier.theory_state, &candidate.predicate)?;
                let (left_rows, right_rows) = context.split_masks(&node, &candidate.predicate)?;
                let left_counts = context.child_class_counts(&left_rows)?;
                let right_counts = context.child_class_counts(&right_rows)?;
                let left_majority = majority(&left_counts);
                let right_majority = majority(&right_counts);
                let left_id = next_node_id;
                let right_id = next_node_id + 1;
                next_node_id += 2;
                let mut child = state.clone();
                child.frontier.remove(frontier_index);
                let replacement = PartialTree::Internal {
                    predicate: candidate.predicate,
                    left: Box::new(PartialTree::Leaf {
                        node_id: left_id,
                        class: left_majority,
                        samples: left_rows.count_ones(),
                    }),
                    right: Box::new(PartialTree::Leaf {
                        node_id: right_id,
                        class: right_majority,
                        samples: right_rows.count_ones(),
                    }),
                    majority_class: frontier.majority_class,
                };
                debug_assert!(child.tree.replace_leaf(frontier.node_id, replacement));
                child.expanded_nodes += 1;
                child.generated_order = generated_order;
                generated_order += 1;
                let child_depth = frontier.depth + 1;
                let child_node_count = child.tree.nodes();
                append_child_frontier_or_error(
                    context,
                    cfg,
                    &mut child,
                    ChildSearchView {
                        node_id: left_id,
                        rows: left_rows,
                        depth: child_depth,
                        theory_state: child_state,
                        majority_class: left_majority,
                        counts: left_counts,
                        current_node_count: child_node_count,
                    },
                );
                append_child_frontier_or_error(
                    context,
                    cfg,
                    &mut child,
                    ChildSearchView {
                        node_id: right_id,
                        rows: right_rows,
                        depth: child_depth,
                        theory_state: child_state,
                        majority_class: right_majority,
                        counts: right_counts,
                        current_node_count: child_node_count,
                    },
                );
                update_partial_objective(context, &mut child);
                diagnostics.states_generated += 1;
                next.push(child);
            }
        }
        next.sort_by(compare_partial_states);
        let width = if sparse_greedy {
            1
        } else {
            cfg.tree_search.tree_beam_width.max(1)
        };
        if next.len() > width {
            diagnostics.states_pruned += next.len() - width;
            next.truncate(width);
        }
        diagnostics.maximum_live_states = diagnostics.maximum_live_states.max(next.len());
        beam = next;
    }

    // Every partial state is already a valid complete prediction tree because
    // unexpanded frontier entries are represented by majority leaves.
    for state in &beam {
        consider_partial_incumbent(context, state, &mut best);
    }
    context.record_beam_search(diagnostics);
    if cfg.theorem_mode && !tree_is_certified(&best.1) {
        return Err(SmartMdtError::TheoremRejected(
            "partial-tree search produced an incompatible path".into(),
        ));
    }
    Ok(best.1)
}

fn node_is_terminal(
    _context: &TrainingContext,
    cfg: &LearnerConfig,
    node: &NodeView,
    counts: &[usize],
) -> bool {
    node.depth >= cfg.max_depth
        || node.rows.count_ones() < cfg.min_samples_split
        || counts.iter().filter(|&&count| count > 0).count() <= 1
}

fn majority(counts: &[usize]) -> ClassId {
    counts
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| **count)
        .map_or(0, |(class, _)| class as ClassId)
}

fn majority_mistakes(counts: &[usize]) -> usize {
    counts.iter().sum::<usize>() - counts.iter().copied().max().unwrap_or(0)
}

struct ChildSearchView {
    node_id: usize,
    rows: crate::data::BitSet,
    depth: usize,
    theory_state: PathTheoryState,
    majority_class: ClassId,
    counts: Vec<usize>,
    current_node_count: usize,
}

fn append_child_frontier_or_error(
    context: &TrainingContext,
    cfg: &LearnerConfig,
    state: &mut PartialTreeState,
    child: ChildSearchView,
) {
    let view = NodeView {
        rows: child.rows.clone(),
        depth: child.depth,
        theory_state: child.theory_state,
    };
    if !node_is_terminal(context, cfg, &view, &child.counts)
        && child.current_node_count.saturating_add(2) <= cfg.tree_search.node_budget
    {
        state.frontier.push(FrontierLeaf {
            node_id: child.node_id,
            rows: child.rows,
            depth: child.depth,
            theory_state: child.theory_state,
            majority_class: child.majority_class,
        });
    } else {
        state.training_error_lower_bound +=
            majority_mistakes(&child.counts) as f64 / context.dataset.labels.len().max(1) as f64;
    }
}

fn add_completed_leaf_error(
    context: &TrainingContext,
    rows: &crate::data::BitSet,
    state: &mut PartialTreeState,
) {
    let view = NodeView {
        rows: rows.clone(),
        depth: 0,
        theory_state: PathTheoryState::Uncommitted,
    };
    if let Ok(counts) = context.class_counts(&view) {
        state.training_error_lower_bound +=
            majority_mistakes(&counts) as f64 / context.dataset.labels.len().max(1) as f64;
    }
    update_partial_objective(context, state);
}

fn update_partial_objective(context: &TrainingContext, state: &mut PartialTreeState) {
    let internal_nodes = state.tree.nodes().saturating_sub(1) / 2;
    state.complexity_cost =
        internal_nodes as f64 / context.dataset.labels.len().max(1) as f64 * 1e-9;
    state.objective_lower_bound = state.training_error_lower_bound + state.complexity_cost;
}

fn select_frontier(
    context: &TrainingContext,
    cfg: &LearnerConfig,
    state: &PartialTreeState,
) -> Result<usize> {
    let mut scores = Vec::with_capacity(state.frontier.len());
    for (index, frontier) in state.frontier.iter().enumerate() {
        let view = NodeView {
            rows: frontier.rows.clone(),
            depth: frontier.depth,
            theory_state: frontier.theory_state,
        };
        let counts = context.class_counts(&view)?;
        let primary = match cfg.tree_search.frontier_selection {
            super::FrontierSelection::HighestError => majority_mistakes(&counts) as f64,
            super::FrontierSelection::MostSamples => frontier.rows.count_ones() as f64,
            super::FrontierSelection::BestPotentialGain => {
                let remaining = cfg
                    .tree_search
                    .node_budget
                    .saturating_sub(state.tree.nodes());
                let key = search_state_key(&view, cfg, remaining);
                candidates(context, &view, &key, cfg)?
                    .first()
                    .map_or(f64::NEG_INFINITY, |candidate| candidate.score.final_score)
            }
        };
        scores.push((index, primary, frontier.depth, frontier.node_id));
    }
    scores.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    Ok(scores[0].0)
}

fn compare_partial_states(left: &PartialTreeState, right: &PartialTreeState) -> Ordering {
    left.objective_lower_bound
        .total_cmp(&right.objective_lower_bound)
        .then_with(|| left.complexity_cost.total_cmp(&right.complexity_cost))
        .then_with(|| left.expanded_nodes.cmp(&right.expanded_nodes))
        .then_with(|| left.tree.canonical_key().cmp(&right.tree.canonical_key()))
        .then_with(|| left.generated_order.cmp(&right.generated_order))
}

fn complete_tree_objective(context: &TrainingContext, tree: &TreeNode) -> f64 {
    let mistakes = context
        .dataset
        .labels
        .iter()
        .enumerate()
        .filter(|(row, label)| predict_row(tree, &context.dataset.features, *row) != **label)
        .count();
    let error = mistakes as f64 / context.dataset.labels.len().max(1) as f64;
    let internal_nodes = tree.nodes().saturating_sub(1) / 2;
    error + internal_nodes as f64 / context.dataset.labels.len().max(1) as f64 * 1e-9
}

fn consider_partial_incumbent(
    context: &TrainingContext,
    state: &PartialTreeState,
    best: &mut (f64, TreeNode),
) {
    let candidate: TreeNode = state.tree.clone().into();
    if !tree_is_certified(&candidate) {
        return;
    }
    let objective = complete_tree_objective(context, &candidate);
    let ordering = objective
        .total_cmp(&best.0)
        .then_with(|| candidate.nodes().cmp(&best.1.nodes()))
        .then_with(|| candidate.literals().cmp(&best.1.literals()));
    if ordering == Ordering::Less {
        *best = (objective, candidate);
    }
}

fn cache_subtree(
    context: &TrainingContext,
    node: &NodeView,
    state_key: SearchStateKey,
    tree: &TreeNode,
) {
    if !(context.cache_config.enabled && context.cache_config.best_subtrees) {
        return;
    }
    let mut mistakes = 0usize;
    for row in 0..node.rows.len() {
        if node.rows.get(row)
            && predict_row(tree, &context.dataset.features, row) != context.dataset.labels[row]
        {
            mistakes += 1;
        }
    }
    let samples = node.rows.count_ones();
    let training_error = mistakes as f64 / samples.max(1) as f64;
    context.insert_best_subtree(
        state_key,
        CachedSubtree {
            tree: Arc::new(tree.clone()),
            training_error,
            validation_error: None,
            node_count: tree.nodes(),
            leaf_count: tree.leaves(),
            literal_count: tree.literals(),
            estimated_axp_length: None,
            objective: training_error,
            path_certified: tree_is_certified(tree),
        },
    );
}
/// Returns the distinct theory states reached by certified root-to-leaf paths.
pub fn tree_path_theory_states(tree: &TreeNode) -> Result<Vec<PathTheoryState>> {
    fn visit(
        tree: &TreeNode,
        state: PathTheoryState,
        leaves: &mut Vec<PathTheoryState>,
    ) -> Result<()> {
        match tree {
            TreeNode::Leaf { .. } => leaves.push(state),
            TreeNode::Internal {
                predicate,
                left,
                right,
                ..
            } => {
                let next = next_theory_state(state, predicate)?;
                visit(left, next, leaves)?;
                visit(right, next, leaves)?;
            }
        }
        Ok(())
    }

    let mut states = Vec::new();
    visit(tree, PathTheoryState::Uncommitted, &mut states)?;
    states.sort_unstable();
    states.dedup();
    Ok(states)
}

/// Stable path-level metadata for benchmark and debug output.
pub fn tree_path_theory_metadata(tree: &TreeNode) -> (String, String, bool) {
    match tree_path_theory_states(tree) {
        Ok(states) => {
            let state = states
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("|");
            let mut backends = states.iter().map(|s| s.backend()).collect::<Vec<_>>();
            backends.sort_by_key(|backend| format!("{backend:?}"));
            backends.dedup();
            let backend = backends
                .iter()
                .map(|backend| format!("{backend:?}"))
                .collect::<Vec<_>>()
                .join("|");
            (state, backend, true)
        }
        Err(_) => ("incompatible".into(), "Unsupported".into(), false),
    }
}

/// Returns true iff every root-to-leaf path stays within one tractable theory.
pub fn tree_is_certified(tree: &TreeNode) -> bool {
    tree_path_theory_states(tree).is_ok()
}
