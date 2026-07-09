/// Multi-objective split score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CandidateScore {
    pub predictive_gain: f64,
    pub complexity_penalty: f64,
    pub explanation_risk: f64,
    pub estimated_cost: f64,
    pub certificate_bonus: f64,
    pub final_score: f64,
}
/// Scoring policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoringPolicy {
    InformationGain,
    GiniGain,
    GainRatio,
    CertificateGuided,
    ExplanationAware,
}
/// Weights for certificate-guided scoring.
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
/// Information gain.
pub fn information_gain(parent: &[usize], left: &[usize], right: &[usize]) -> f64 {
    let n: usize = parent.iter().sum();
    let l: usize = left.iter().sum();
    let r: usize = right.iter().sum();
    entropy(parent) - l as f64 / n as f64 * entropy(left) - r as f64 / n as f64 * entropy(right)
}
/// Gini impurity.
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
/// Certificate-guided final score.
pub fn final_score(
    gain: f64,
    complexity: f64,
    risk: f64,
    cost: f64,
    cert: bool,
    w: ScoreWeights,
) -> CandidateScore {
    let cb = if cert { 1.0 } else { 0.0 };
    CandidateScore {
        predictive_gain: gain,
        complexity_penalty: complexity,
        explanation_risk: risk,
        estimated_cost: cost,
        certificate_bonus: cb,
        final_score: gain - w.lambda_size * complexity - w.lambda_axp * risk - w.lambda_time * cost
            + w.lambda_cert * cb,
    }
}
