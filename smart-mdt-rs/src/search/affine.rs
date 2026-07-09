use crate::{
    data::{class_counts, is_boolean_column, predicate_mask, BitSet, Dataset},
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    search::{final_score, information_gain, ScoreWeights, SplitCandidate},
    FeatureId,
};
use std::collections::BTreeSet;

/// Configuration for certified affine candidate generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineConfig {
    /// Maximum arity of generated affine equations (2 and 3 by default, 4 optional).
    pub max_arity: usize,
}
impl Default for AffineConfig {
    fn default() -> Self {
        Self { max_arity: 3 }
    }
}

/// Diagnostics for affine candidate generation in the learner path.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AffineSearchDiagnostics {
    pub candidate_count_before_filtering: usize,
    pub candidate_count_after_filtering: usize,
    pub boolean_feature_count: usize,
    pub best_final_score: f64,
    pub best_gain: f64,
    pub selected_predicate: String,
    pub leaf_reason: String,
}

/// Generates certified Boolean affine candidates with the default configuration.
pub fn generate_affine(data: &Dataset, min_leaf: usize, beam: usize) -> Vec<SplitCandidate> {
    generate_affine_with_config(data, min_leaf, beam, AffineConfig::default())
}

/// Generates certified Boolean affine candidates with an explicit configuration.
pub fn generate_affine_with_config(
    data: &Dataset,
    min_leaf: usize,
    beam: usize,
    cfg: AffineConfig,
) -> Vec<SplitCandidate> {
    generate_affine_with_diagnostics(data, min_leaf, beam, cfg).0
}

/// Generates affine candidates and diagnostics using the same path as the learner.
///
/// Only Boolean features participate, so every generated predicate has an
/// all-Boolean scope and is theorem-certifiable by the GF(2) backend.
pub fn generate_affine_with_diagnostics(
    data: &Dataset,
    min_leaf: usize,
    beam: usize,
    cfg: AffineConfig,
) -> (Vec<SplitCandidate>, AffineSearchDiagnostics) {
    let features = ranked_boolean_features(data);
    let boolean_feature_count = features.len();
    let pool: Vec<FeatureId> = features
        .into_iter()
        .take(beam.max(cfg.max_arity).max(2))
        .collect();

    let mut before = 0usize;
    let mut seen_masks = BTreeSet::new();
    let mut out = Vec::new();
    let max_arity = cfg.max_arity.clamp(2, 4);
    for k in 2..=max_arity {
        for combo in combinations(pool.len(), k) {
            let literals: Vec<Literal> = combo.iter().map(|&i| boolean_literal(pool[i])).collect();
            for rhs in [false, true] {
                before += 1;
                let p = Predicate::Affine {
                    literals: literals.clone(),
                    rhs,
                };
                let m = predicate_mask(&data.features, &p);
                let l = m.count_ones();
                let r = data.labels.len().saturating_sub(l);
                if l < min_leaf || r < min_leaf {
                    continue;
                }
                if !seen_masks.insert(mask_signature(&m)) {
                    continue;
                }
                out.push(score_candidate(data, p, &m, l, r));
            }
        }
    }

    out.sort_by(|a, b| b.score.final_score.total_cmp(&a.score.final_score));
    let best = out.first();
    let diag = AffineSearchDiagnostics {
        candidate_count_before_filtering: before,
        candidate_count_after_filtering: out.len(),
        boolean_feature_count,
        best_final_score: best.map_or(0.0, |c| c.score.final_score),
        best_gain: best.map_or(0.0, |c| c.score.predictive_gain),
        selected_predicate: best.map_or_else(String::new, |c| format!("{:?}", c.predicate)),
        leaf_reason: if out.is_empty() {
            "no_affine_candidate_after_filtering".into()
        } else {
            String::new()
        },
    };
    (out, diag)
}

/// Canonical Boolean literal for a feature: true iff the feature value is 1.
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

/// Ranks Boolean features by descending unary gain, tie-broken by feature index.
fn ranked_boolean_features(data: &Dataset) -> Vec<FeatureId> {
    let mut features: Vec<FeatureId> = (0..data.features.cols() as FeatureId)
        .filter(|&f| is_boolean_column(&data.features, f))
        .collect();
    features.sort_by(|&a, &b| {
        feature_gain(data, b)
            .total_cmp(&feature_gain(data, a))
            .then(a.cmp(&b))
    });
    features
}

fn feature_gain(data: &Dataset, feature: FeatureId) -> f64 {
    let p = Predicate::Unary(boolean_literal(feature));
    let m = predicate_mask(&data.features, &p);
    let l = m.count_ones();
    let r = data.labels.len().saturating_sub(l);
    score_candidate(data, p, &m, l, r).score.predictive_gain
}

fn score_candidate(data: &Dataset, p: Predicate, m: &BitSet, l: usize, r: usize) -> SplitCandidate {
    let classes = data.class_count().max(2);
    let mut parent = vec![0; classes];
    for &y in &data.labels {
        parent[y as usize] += 1;
    }
    let lc = class_counts(&data.labels, m, classes);
    let rc: Vec<_> = parent.iter().zip(&lc).map(|(a, b)| a - b).collect();
    let gain = information_gain(&parent, &lc, &rc);
    let k = p.arity() as f64;
    SplitCandidate {
        predicate: p,
        score: final_score(gain, k, k, k, true, ScoreWeights::default()),
        left_count: l,
        right_count: r,
    }
}

/// Enumerates all `k`-subsets of `0..n` in lexicographic order (deterministic).
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        let mut i = k - 1;
        while idx[i] == i + n - k {
            if i == 0 {
                return out;
            }
            i -= 1;
        }
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

fn mask_signature(mask: &BitSet) -> Vec<bool> {
    (0..mask.len()).map(|i| mask.get(i)).collect()
}
