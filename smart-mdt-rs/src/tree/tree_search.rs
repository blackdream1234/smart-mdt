//! Deterministic search state for sparse lookahead and global tree beams.

use super::TreeNode;
use crate::{data::BitSet, logic::PathTheoryState, ClassId};

/// Strategy used to construct a complete tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TreeSearchStrategy {
    /// Historical recursive best-candidate induction.
    #[default]
    Greedy,
    /// Retain several partial trees near the root, then finish greedily.
    SparseLookahead,
    /// Apply bounded lookahead only at eligible ambiguous, low-gain, or large nodes.
    SelectiveLookahead,
    /// Retain a bounded beam of complete partial-tree alternatives.
    GlobalBeam,
}

/// Node-local gate and widths for selective lookahead.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectiveLookaheadConfig {
    pub enabled: bool,
    pub maximum_depth: usize,
    pub ambiguity_margin: f64,
    pub minimum_confident_gain: f64,
    pub large_node_threshold: usize,
    pub candidate_beam_width: usize,
    pub tree_beam_width: usize,
}

impl Default for SelectiveLookaheadConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            maximum_depth: 2,
            ambiguity_margin: 0.01,
            minimum_confident_gain: 0.02,
            large_node_threshold: 500,
            candidate_beam_width: 8,
            tree_beam_width: 4,
        }
    }
}

/// Gates expensive candidate selection and memoization by estimated work/reuse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditionalCandidateSearchConfig {
    pub enabled: bool,
    pub branch_and_bound_candidate_threshold: usize,
    pub candidate_cache_minimum_expected_reuse: usize,
}

impl Default for ConditionalCandidateSearchConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            branch_and_bound_candidate_threshold: 64,
            candidate_cache_minimum_expected_reuse: 2,
        }
    }
}

/// Auditable decisions made by conditional candidate search.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConditionalSearchDiagnostics {
    pub branch_and_bound_activation_count: usize,
    pub branch_and_bound_avoided_count: usize,
    pub cache_activation_count: usize,
    pub estimated_work_saved: usize,
}

/// Deterministic policy for choosing the next open leaf in a partial tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FrontierSelection {
    /// Prefer the leaf with the most current majority-class mistakes.
    #[default]
    HighestError,
    /// Prefer the leaf containing the most training rows.
    MostSamples,
    /// Prefer the leaf with the largest best admissible local gain.
    BestPotentialGain,
}

/// Bounded anytime tree-search configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct TreeSearchConfig {
    pub strategy: TreeSearchStrategy,
    pub tree_beam_width: usize,
    pub candidate_beam_width: usize,
    pub lookahead_depth: usize,
    pub max_expansions: usize,
    /// Maximum total nodes, including leaves. `usize::MAX` disables the cap.
    pub node_budget: usize,
    pub time_budget_ms: Option<u64>,
    pub frontier_selection: FrontierSelection,
    pub selective: SelectiveLookaheadConfig,
}

impl Default for TreeSearchConfig {
    fn default() -> Self {
        Self {
            strategy: TreeSearchStrategy::Greedy,
            tree_beam_width: 8,
            candidate_beam_width: 8,
            lookahead_depth: 2,
            max_expansions: usize::MAX,
            node_budget: usize::MAX,
            time_budget_ms: None,
            frontier_selection: FrontierSelection::HighestError,
            selective: SelectiveLookaheadConfig::default(),
        }
    }
}

/// One open majority leaf that may still be replaced by a certified split.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontierLeaf {
    pub node_id: usize,
    pub rows: BitSet,
    pub depth: usize,
    pub theory_state: PathTheoryState,
    pub majority_class: ClassId,
}

/// Immutable-by-cloning representation used by the bounded global search.
#[derive(Clone, Debug, PartialEq)]
pub enum PartialTree {
    Leaf {
        node_id: usize,
        class: ClassId,
        samples: usize,
    },
    Internal {
        predicate: crate::logic::Predicate,
        left: Box<PartialTree>,
        right: Box<PartialTree>,
        majority_class: ClassId,
    },
}

impl PartialTree {
    pub(crate) fn replace_leaf(&mut self, node_id: usize, replacement: PartialTree) -> bool {
        match self {
            Self::Leaf { node_id: id, .. } if *id == node_id => {
                *self = replacement;
                true
            }
            Self::Leaf { .. } => false,
            Self::Internal { left, right, .. } => {
                left.replace_leaf(node_id, replacement.clone())
                    || right.replace_leaf(node_id, replacement)
            }
        }
    }

    pub(crate) fn nodes(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Internal { left, right, .. } => 1 + left.nodes() + right.nodes(),
        }
    }

    pub(crate) fn canonical_key(&self) -> String {
        match self {
            Self::Leaf { class, .. } => format!("L{class}"),
            Self::Internal {
                predicate,
                left,
                right,
                ..
            } => format!(
                "I({predicate:?},{},{})",
                left.canonical_key(),
                right.canonical_key()
            ),
        }
    }
}

impl From<PartialTree> for TreeNode {
    fn from(value: PartialTree) -> Self {
        match value {
            PartialTree::Leaf { class, samples, .. } => Self::Leaf { class, samples },
            PartialTree::Internal {
                predicate,
                left,
                right,
                majority_class,
            } => Self::Internal {
                predicate,
                left: Box::new((*left).into()),
                right: Box::new((*right).into()),
                majority_class,
            },
        }
    }
}

/// One item in the global partial-tree beam.
#[derive(Clone, Debug, PartialEq)]
pub struct PartialTreeState {
    pub tree: PartialTree,
    pub frontier: Vec<FrontierLeaf>,
    pub training_error_lower_bound: f64,
    pub complexity_cost: f64,
    pub objective_lower_bound: f64,
    pub expanded_nodes: usize,
    pub generated_order: u64,
}

/// Auditable counters for one sparse-lookahead or global-beam fit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BeamSearchDiagnostics {
    pub states_generated: usize,
    pub states_expanded: usize,
    pub states_pruned: usize,
    pub completed_states: usize,
    pub maximum_live_states: usize,
    pub expansion_budget_reached: bool,
    pub time_budget_reached: bool,
    pub path_incompatible_candidates_rejected: usize,
    pub nodes_using_greedy_selection: usize,
    pub nodes_using_selective_lookahead: usize,
}
