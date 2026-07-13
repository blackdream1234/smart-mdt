use super::{
    predict_row, CacheConfig, CachedSubtree, NodeView, SearchStateKey, TrainingContext,
    TrainingDiagnostics, TreeNode,
};
use crate::{
    data::Dataset,
    logic::{candidate_is_compatible, next_theory_state, PathTheoryState},
    search::{
        exact_branch_and_bound_top_k, BranchAndBoundConfig, SplitCandidate, SplitScoreConfig,
    },
    Result, SmartMdtError,
};
use std::sync::Arc;
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
    let context = TrainingContext::with_cache_config(Arc::new(data.clone()), cfg.cache.clone());
    let tree = build(&context, cfg, context.root_view())?;
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
    let mut generated = context.generate_candidates(
        node,
        cfg.language_policy,
        cfg.min_samples_leaf,
        cfg.beam_width,
        &cfg.split_score,
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
    let state_key = search_state_key(&node, cfg);
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

fn search_state_key(node: &NodeView, cfg: &LearnerConfig) -> SearchStateKey {
    SearchStateKey::new(
        &node.rows,
        cfg.max_depth.saturating_sub(node.depth),
        u16::MAX as usize,
        node.theory_state,
        format!("{:?}", cfg.split_score),
        format!(
            "policy={:?};min_split={};min_leaf={};candidate_cap={};beam={};branch={:?}",
            cfg.language_policy,
            cfg.min_samples_split,
            cfg.min_samples_leaf,
            cfg.max_candidates_per_node,
            cfg.beam_width,
            cfg.branch_and_bound
        ),
    )
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
