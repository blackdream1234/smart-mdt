use super::TreeNode;
use crate::{
    data::{ColumnMajorMatrix, Dataset},
    logic::{candidate_is_compatible, next_theory_state, PathTheoryState},
    search::{
        affine::generate_affine, antihorn::generate_antihorn, horn::generate_horn,
        square2cnf::generate_square2cnf, top_k, unary::generate_unary, SplitCandidate,
    },
    ClassId, Result, SmartMdtError,
};
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
            language_policy: LanguagePolicy::BestCertifiedPerNode,
            theorem_mode: true,
            random_seed: 42,
        }
    }
}
/// Learns a CGS-MDT tree.
pub fn learn(data: &Dataset, cfg: &LearnerConfig) -> Result<TreeNode> {
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
    build(data, cfg, 0, PathTheoryState::Uncommitted)
}
fn majority(labels: &[ClassId]) -> ClassId {
    let mut m = std::collections::BTreeMap::new();
    for &l in labels {
        *m.entry(l).or_insert(0usize) += 1;
    }
    m.into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(l, _)| l)
        .unwrap_or(0)
}
fn subset(data: &Dataset, rows: &[usize]) -> Result<Dataset> {
    let matrix_rows: Vec<Vec<f64>> = rows
        .iter()
        .map(|&i| {
            (0..data.features.cols())
                .map(|j| data.features.get(i, j as u32))
                .collect()
        })
        .collect();
    let labels = rows.iter().map(|&i| data.labels[i]).collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&matrix_rows)?, labels)
}
fn candidates(
    data: &Dataset,
    cfg: &LearnerConfig,
    path_state: PathTheoryState,
) -> Vec<SplitCandidate> {
    let mut xs = Vec::new();
    match cfg.language_policy {
        LanguagePolicy::UnaryOnly => xs.extend(generate_unary(data, cfg.min_samples_leaf)),
        LanguagePolicy::HornOnly => {
            xs.extend(generate_horn(data, cfg.min_samples_leaf, cfg.beam_width))
        }
        LanguagePolicy::AntiHornOnly => xs.extend(generate_antihorn(
            data,
            cfg.min_samples_leaf,
            cfg.beam_width,
        )),
        LanguagePolicy::Square2CnfOnly => xs.extend(generate_square2cnf(
            data,
            cfg.min_samples_leaf,
            cfg.beam_width,
        )),
        LanguagePolicy::AffineOnly => {
            xs.extend(generate_affine(data, cfg.min_samples_leaf, cfg.beam_width))
        }
        LanguagePolicy::SmartCertified => {
            xs.extend(generate_unary(data, cfg.min_samples_leaf));
            xs.extend(generate_horn(data, cfg.min_samples_leaf, cfg.beam_width));
            xs.extend(generate_antihorn(
                data,
                cfg.min_samples_leaf,
                cfg.beam_width,
            ));
            xs.extend(generate_square2cnf(
                data,
                cfg.min_samples_leaf,
                cfg.beam_width,
            ));
            xs.extend(generate_affine(data, cfg.min_samples_leaf, cfg.beam_width));
            xs.retain(|candidate| candidate_is_compatible(path_state, &candidate.predicate));
        }
        LanguagePolicy::CertifiedOnly | LanguagePolicy::BestCertifiedPerNode => {
            xs.extend(generate_unary(data, cfg.min_samples_leaf));
            xs.extend(generate_horn(data, cfg.min_samples_leaf, cfg.beam_width));
            xs.extend(generate_antihorn(
                data,
                cfg.min_samples_leaf,
                cfg.beam_width,
            ));
            xs.extend(generate_square2cnf(
                data,
                cfg.min_samples_leaf,
                cfg.beam_width,
            ));
        }
        LanguagePolicy::EmpiricalMixed | LanguagePolicy::TunedExperimental => {
            xs.extend(generate_unary(data, cfg.min_samples_leaf))
        }
    }
    top_k(xs, cfg.max_candidates_per_node)
}
fn build(
    data: &Dataset,
    cfg: &LearnerConfig,
    depth: usize,
    path_state: PathTheoryState,
) -> Result<TreeNode> {
    let maj = majority(&data.labels);
    if depth >= cfg.max_depth
        || data.labels.len() < cfg.min_samples_split
        || data.labels.iter().all(|&l| l == data.labels[0])
    {
        return Ok(TreeNode::Leaf {
            class: maj,
            samples: data.labels.len(),
        });
    }
    let cand = candidates(data, cfg, path_state);
    let Some(best) = cand.into_iter().next() else {
        return Ok(TreeNode::Leaf {
            class: maj,
            samples: data.labels.len(),
        });
    };
    if cfg.theorem_mode && !best.predicate.language().theorem_table_allowed() {
        return Err(SmartMdtError::TheoremRejected(
            "non-certified predicate selected".into(),
        ));
    }
    let mut lrows = Vec::new();
    let mut rrows = Vec::new();
    for i in 0..data.labels.len() {
        if best.predicate.eval(&data.features, i) {
            lrows.push(i)
        } else {
            rrows.push(i)
        }
    }
    let child_state = if cfg.language_policy == LanguagePolicy::SmartCertified {
        next_theory_state(path_state, &best.predicate)?
    } else {
        path_state
    };
    // Each recursive call receives its own copy, so descendants of one branch
    // cannot commit the sibling branch to a theory.
    let left = build(&subset(data, &lrows)?, cfg, depth + 1, child_state)?;
    let right = build(&subset(data, &rrows)?, cfg, depth + 1, child_state)?;
    Ok(TreeNode::Internal {
        predicate: best.predicate,
        left: Box::new(left),
        right: Box::new(right),
        majority_class: maj,
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
