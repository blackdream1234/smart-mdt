use crate::logic::{LanguageFamily, Predicate};

/// Configurable candidate-score profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SplitScoreProfile {
    InformationGain,
    GainRatio,
    SparseCertified,
    LookaheadReady,
}

/// Weights for the admissible-candidate objective.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitScoreConfig {
    pub profile: SplitScoreProfile,
    pub gain_weight: f64,
    pub gain_ratio_weight: f64,
    pub balance_weight: f64,
    pub literal_penalty: f64,
    pub family_complexity_penalty: f64,
    pub child_fragmentation_penalty: f64,
    pub estimated_subtree_penalty: f64,
    pub instability_penalty: f64,
    pub tie_epsilon: f64,
}

impl SplitScoreConfig {
    /// Ranking-compatible replacement for the historical certificate-guided score.
    pub fn information_gain() -> Self {
        Self {
            profile: SplitScoreProfile::InformationGain,
            gain_weight: 1.0,
            gain_ratio_weight: 0.0,
            balance_weight: 0.0,
            // Historical ranking subtracted 0.02 + 0.05 + 0.001 per literal.
            // Its +0.1 certificate bonus was constant after hard admissibility
            // filtering and is intentionally removed.
            literal_penalty: 0.071,
            family_complexity_penalty: 0.0,
            child_fragmentation_penalty: 0.0,
            estimated_subtree_penalty: 0.0,
            instability_penalty: 0.0,
            tie_epsilon: 1e-12,
        }
    }

    pub fn gain_ratio() -> Self {
        Self {
            profile: SplitScoreProfile::GainRatio,
            gain_weight: 0.25,
            gain_ratio_weight: 0.75,
            balance_weight: 0.0,
            literal_penalty: 0.005,
            family_complexity_penalty: 0.0,
            child_fragmentation_penalty: 0.0,
            estimated_subtree_penalty: 0.0,
            instability_penalty: 0.0,
            tie_epsilon: 1e-12,
        }
    }

    pub fn sparse_certified() -> Self {
        Self {
            profile: SplitScoreProfile::SparseCertified,
            gain_weight: 1.0,
            gain_ratio_weight: 0.03,
            balance_weight: 0.01,
            literal_penalty: 0.004,
            family_complexity_penalty: 0.002,
            child_fragmentation_penalty: 0.005,
            estimated_subtree_penalty: 0.01,
            instability_penalty: 0.0,
            tie_epsilon: 1e-9,
        }
    }

    pub fn lookahead_ready() -> Self {
        Self {
            profile: SplitScoreProfile::LookaheadReady,
            estimated_subtree_penalty: 0.025,
            ..Self::sparse_certified()
        }
    }
}

impl Default for SplitScoreConfig {
    fn default() -> Self {
        Self::information_gain()
    }
}

/// Cheap node-local statistics used to score one already-admissible candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitScoreInput {
    pub information_gain: f64,
    pub true_count: usize,
    pub false_count: usize,
    pub literal_count: usize,
    pub family: LanguageFamily,
    pub fragmentation: f64,
    pub estimated_subtree_cost: f64,
    pub instability: f64,
}

/// Multi-objective split score and auditable weighted components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateScore {
    pub score_profile: SplitScoreProfile,
    pub predictive_gain: f64,
    pub gain_ratio: f64,
    pub balance: f64,
    pub balance_component: f64,
    pub literal_penalty: f64,
    pub family_penalty: f64,
    pub fragmentation_penalty: f64,
    pub estimated_subtree_penalty: f64,
    pub instability_penalty: f64,
    // Compatibility aliases retained for existing diagnostics consumers.
    pub complexity_penalty: f64,
    pub explanation_risk: f64,
    pub estimated_cost: f64,
    /// Always zero: certification is a hard filter, never a score bonus.
    pub certificate_bonus: f64,
    pub final_score: f64,
}

/// Scoring policy retained for the public research API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoringPolicy {
    InformationGain,
    GiniGain,
    GainRatio,
    CertificateGuided,
    ExplanationAware,
}

/// Historical weights retained for callers of `final_score`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreWeights {
    pub lambda_size: f64,
    pub lambda_axp: f64,
    pub lambda_time: f64,
    pub lambda_cert: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            lambda_size: 0.02,
            lambda_axp: 0.05,
            lambda_time: 0.001,
            lambda_cert: 0.1,
        }
    }
}

/// Computes entropy.
pub fn entropy(counts: &[usize]) -> f64 {
    let n: usize = counts.iter().sum();
    if n == 0 {
        return 0.0;
    }
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n as f64;
            -p * p.log2()
        })
        .sum()
}

/// Computes information gain.
pub fn information_gain(parent: &[usize], left: &[usize], right: &[usize]) -> f64 {
    let n: usize = parent.iter().sum();
    let l: usize = left.iter().sum();
    let r: usize = right.iter().sum();
    entropy(parent) - l as f64 / n as f64 * entropy(left) - r as f64 / n as f64 * entropy(right)
}

/// Computes Gini impurity.
pub fn gini(c: &[usize]) -> f64 {
    let n: usize = c.iter().sum();
    if n == 0 {
        return 0.0;
    }
    1.0 - c
        .iter()
        .map(|&x| {
            let p = x as f64 / n as f64;
            p * p
        })
        .sum::<f64>()
}

/// Computes the configured score after certification/path filtering.
pub fn score_split(input: SplitScoreInput, config: &SplitScoreConfig) -> CandidateScore {
    let total = input.true_count + input.false_count;
    let split_entropy = entropy(&[input.true_count, input.false_count]);
    let gain_ratio = if split_entropy > 0.0 {
        input.information_gain / split_entropy
    } else {
        0.0
    };
    let balance = if total == 0 {
        0.0
    } else {
        2.0 * input.true_count.min(input.false_count) as f64 / total as f64
    };
    let balance_component = config.balance_weight * balance;
    let literal_penalty = config.literal_penalty * input.literal_count as f64;
    let family_penalty = config.family_complexity_penalty * family_complexity(input.family);
    let fragmentation_penalty = config.child_fragmentation_penalty * input.fragmentation;
    let estimated_subtree_penalty = config.estimated_subtree_penalty * input.estimated_subtree_cost;
    let instability_penalty = config.instability_penalty * input.instability;
    let final_score = config.gain_weight * input.information_gain
        + config.gain_ratio_weight * gain_ratio
        + balance_component
        - literal_penalty
        - family_penalty
        - fragmentation_penalty
        - estimated_subtree_penalty
        - instability_penalty;
    CandidateScore {
        score_profile: config.profile,
        predictive_gain: input.information_gain,
        gain_ratio,
        balance,
        balance_component,
        literal_penalty,
        family_penalty,
        fragmentation_penalty,
        estimated_subtree_penalty,
        instability_penalty,
        complexity_penalty: input.literal_count as f64,
        explanation_risk: input.fragmentation,
        estimated_cost: input.estimated_subtree_cost,
        certificate_bonus: 0.0,
        final_score,
    }
}

/// Compatibility wrapper for existing family generators.
pub fn final_score(
    gain: f64,
    complexity: f64,
    risk: f64,
    cost: f64,
    _cert: bool,
    weights: ScoreWeights,
) -> CandidateScore {
    let config = SplitScoreConfig {
        literal_penalty: weights.lambda_size + weights.lambda_axp + weights.lambda_time,
        ..SplitScoreConfig::information_gain()
    };
    score_split(
        SplitScoreInput {
            information_gain: gain,
            true_count: 1,
            false_count: 1,
            literal_count: complexity.round().max(0.0) as usize,
            family: LanguageFamily::Unary,
            fragmentation: risk,
            estimated_subtree_cost: cost,
            instability: 0.0,
        },
        &config,
    )
}

pub fn family_order(family: LanguageFamily) -> u8 {
    match family {
        LanguageFamily::Unary => 0,
        LanguageFamily::Horn => 1,
        LanguageFamily::AntiHorn => 2,
        LanguageFamily::Square2Cnf => 3,
        LanguageFamily::Affine => 4,
        LanguageFamily::SmartCertified => 5,
        LanguageFamily::EmpiricalAffine => 6,
        LanguageFamily::EmpiricalMixed => 7,
        LanguageFamily::TunedExperimental => 8,
    }
}

pub fn family_complexity(family: LanguageFamily) -> f64 {
    match family {
        LanguageFamily::Unary => 1.0,
        LanguageFamily::Horn | LanguageFamily::AntiHorn => 1.25,
        LanguageFamily::Square2Cnf => 1.75,
        LanguageFamily::Affine => 1.5,
        _ => 4.0,
    }
}

/// Stable predicate key used as the final deterministic tie breaker.
pub fn canonical_predicate_key(predicate: &Predicate) -> String {
    format!("{:02}:{predicate:?}", family_order(predicate.language())).replace(',', ";")
}
