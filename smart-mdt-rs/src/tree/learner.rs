use super::{NodeView, TrainingContext, TrainingDiagnostics, TreeNode};
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
    let context = TrainingContext::new(Arc::new(data.clone()));
    let tree = build(&context, cfg, context.root_view())?;
    Ok((tree, context.diagnostics()))
}
fn candidates(
    context: &TrainingContext,
    node: &NodeView,
    cfg: &LearnerConfig,
) -> Result<Vec<SplitCandidate>> {
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
    Ok(selected)
}
fn build(context: &TrainingContext, cfg: &LearnerConfig, node: NodeView) -> Result<TreeNode> {
    let sample_count = context.sample_count(&node);
    let class_counts = context.class_counts(&node)?;
    let majority_class = context.majority_class(&node)?;
    if node.depth >= cfg.max_depth
        || sample_count < cfg.min_samples_split
        || class_counts.iter().filter(|&&count| count > 0).count() <= 1
    {
        return Ok(TreeNode::Leaf {
            class: majority_class,
            samples: sample_count,
        });
    }
    let cand = candidates(context, &node, cfg)?;
    let Some(best) = cand.into_iter().next() else {
        return Ok(TreeNode::Leaf {
            class: majority_class,
            samples: sample_count,
        });
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
    Ok(TreeNode::Internal {
        predicate: best.predicate,
        left: Box::new(left),
        right: Box::new(right),
        majority_class,
    })
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
