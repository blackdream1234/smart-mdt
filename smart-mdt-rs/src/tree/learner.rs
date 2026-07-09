use super::TreeNode;
use crate::{
    data::{ColumnMajorMatrix, Dataset},
    logic::LanguageFamily,
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
    build(data, cfg, 0)
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
fn candidates(data: &Dataset, cfg: &LearnerConfig) -> Vec<SplitCandidate> {
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
fn build(data: &Dataset, cfg: &LearnerConfig, depth: usize) -> Result<TreeNode> {
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
    let cand = candidates(data, cfg);
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
    let left = build(&subset(data, &lrows)?, cfg, depth + 1)?;
    let right = build(&subset(data, &rrows)?, cfg, depth + 1)?;
    Ok(TreeNode::Internal {
        predicate: best.predicate,
        left: Box::new(left),
        right: Box::new(right),
        majority_class: maj,
    })
}
/// Returns true if all internal predicates are theorem-table allowed.
pub fn tree_is_certified(tree: &TreeNode) -> bool {
    match tree {
        TreeNode::Leaf { .. } => true,
        TreeNode::Internal {
            predicate,
            left,
            right,
            ..
        } => {
            matches!(
                predicate.language(),
                LanguageFamily::Unary
                    | LanguageFamily::Horn
                    | LanguageFamily::AntiHorn
                    | LanguageFamily::Square2Cnf
                    | LanguageFamily::Affine
            ) && tree_is_certified(left)
                && tree_is_certified(right)
        }
    }
}
