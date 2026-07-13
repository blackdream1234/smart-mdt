//! Deterministic adaptive budget allocation across compatible certified families.

use crate::logic::{LanguageFamily, PathTheoryState};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveLanguageConfig {
    pub enabled: bool,
    pub total_candidate_budget: usize,
    pub minimum_family_quota: usize,
    pub pilot_candidates_per_family: usize,
    pub temperature: f64,
    pub diversity_top_k_per_family: usize,
}

impl Default for AdaptiveLanguageConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            total_candidate_budget: 64,
            minimum_family_quota: 2,
            pilot_candidates_per_family: 4,
            temperature: 1.0,
            diversity_top_k_per_family: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FamilyPilotMetrics {
    pub family: LanguageFamily,
    pub best_score: f64,
    pub mean_top_k_score: f64,
    pub best_gain: f64,
    pub score_per_literal: f64,
    pub generation_cost: f64,
    pub duplicate_mask_rate: f64,
    pub branch_and_bound_pruning_rate: f64,
    pub candidates_generated: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FamilyBudget {
    pub family: LanguageFamily,
    pub pilot_budget: usize,
    pub final_budget: usize,
    pub utility: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveNodeDiagnostics {
    pub depth: usize,
    pub theory_state: PathTheoryState,
    pub compatible_families: Vec<LanguageFamily>,
    pub pilots: Vec<FamilyPilotMetrics>,
    pub budgets: Vec<FamilyBudget>,
    pub candidates_generated: usize,
    pub candidates_retained: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AdaptiveLanguageDiagnostics {
    pub nodes: Vec<AdaptiveNodeDiagnostics>,
    pub selected_family_counts: BTreeMap<String, usize>,
    pub selected_family_counts_by_depth: BTreeMap<usize, BTreeMap<String, usize>>,
}

/// Allocates exactly the available budget with deterministic largest-remainder
/// rounding. Minimum quotas are honored whenever the total makes that possible.
pub fn allocate_family_budgets(
    config: &AdaptiveLanguageConfig,
    pilots: &[FamilyPilotMetrics],
) -> Vec<FamilyBudget> {
    if pilots.is_empty() {
        return Vec::new();
    }
    let total = config.total_candidate_budget;
    let floor = config
        .minimum_family_quota
        .max(config.diversity_top_k_per_family);
    let mut allocations = vec![0usize; pilots.len()];
    if total >= floor.saturating_mul(pilots.len()) {
        allocations.fill(floor);
    } else {
        for allocation in allocations.iter_mut().take(total) {
            *allocation = 1;
        }
    }
    let assigned = allocations.iter().sum::<usize>();
    let remaining = total.saturating_sub(assigned);
    let utilities = pilots.iter().map(pilot_utility).collect::<Vec<_>>();
    if remaining > 0 {
        let temperature = config.temperature.max(1e-9);
        let maximum = utilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = utilities
            .iter()
            .map(|utility| ((utility - maximum) / temperature).exp())
            .collect::<Vec<_>>();
        let weight_sum = weights.iter().sum::<f64>();
        let mut remainders = Vec::with_capacity(pilots.len());
        let mut distributed = 0usize;
        for (index, weight) in weights.iter().enumerate() {
            let exact = if weight_sum.is_finite() && weight_sum > 0.0 {
                remaining as f64 * weight / weight_sum
            } else {
                remaining as f64 / pilots.len() as f64
            };
            let whole = exact.floor() as usize;
            allocations[index] += whole;
            distributed += whole;
            remainders.push((index, exact - whole as f64));
        }
        remainders.sort_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| {
                family_rank(pilots[left.0].family).cmp(&family_rank(pilots[right.0].family))
            })
        });
        for (index, _) in remainders.into_iter().take(remaining - distributed) {
            allocations[index] += 1;
        }
    }
    pilots
        .iter()
        .zip(utilities)
        .zip(allocations)
        .map(|((pilot, utility), final_budget)| FamilyBudget {
            family: pilot.family,
            pilot_budget: config.pilot_candidates_per_family,
            final_budget,
            utility,
        })
        .collect()
}

fn pilot_utility(pilot: &FamilyPilotMetrics) -> f64 {
    pilot.best_score.max(0.0)
        + 0.25 * pilot.mean_top_k_score.max(0.0)
        + 0.10 * pilot.score_per_literal.max(0.0)
        + 0.05 * pilot.best_gain.max(0.0)
        - 0.01 * pilot.generation_cost.max(0.0)
        - 0.05 * pilot.duplicate_mask_rate.clamp(0.0, 1.0)
        + 0.01 * pilot.branch_and_bound_pruning_rate.clamp(0.0, 1.0)
}

fn family_rank(family: LanguageFamily) -> u8 {
    match family {
        LanguageFamily::Unary => 0,
        LanguageFamily::Horn => 1,
        LanguageFamily::AntiHorn => 2,
        LanguageFamily::Square2Cnf => 3,
        LanguageFamily::Affine => 4,
        _ => u8::MAX,
    }
}
