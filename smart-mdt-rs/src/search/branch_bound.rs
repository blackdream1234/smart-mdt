//! Exact bounded branch-and-bound selection for certified candidate frontiers.

use super::{compare_candidates, top_k_with_config, SplitCandidate, SplitScoreConfig};
use crate::logic::Predicate;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BranchAndBoundConfig {
    pub enabled: bool,
    pub exhaustive_fallback: bool,
    pub max_partial_states: usize,
    pub max_candidates: usize,
    pub top_k: usize,
}

impl Default for BranchAndBoundConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            exhaustive_fallback: true,
            max_partial_states: 100_000,
            max_candidates: 100_000,
            top_k: 64,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FamilyPartialState {
    Unary {
        chosen: Vec<String>,
        remaining: Vec<String>,
    },
    Horn {
        chosen: Vec<String>,
        remaining: Vec<String>,
    },
    AntiHorn {
        chosen: Vec<String>,
        remaining: Vec<String>,
    },
    Square2Cnf {
        chosen: Vec<String>,
        remaining: Vec<String>,
    },
    Affine {
        chosen: Vec<String>,
        remaining: Vec<String>,
    },
}

/// A collision-safe complete frontier state. `safe_upper_bound` is exact for
/// this completed predicate, which is conservative by equality.
#[derive(Clone, Debug)]
pub struct PartialCandidateState {
    pub family_state: FamilyPartialState,
    pub partial_mask_words: Arc<[u64]>,
    pub current_class_counts: Vec<usize>,
    pub current_score: f64,
    pub safe_upper_bound: f64,
    pub canonical_state_key: String,
    pub candidate: SplitCandidate,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BranchAndBoundDiagnostics {
    pub partial_states_created: usize,
    pub partial_states_expanded: usize,
    pub partial_states_pruned: usize,
    pub exhaustive_fallback_count: usize,
    pub complete_candidates_evaluated: usize,
    pub percentage_reduction: f64,
    pub best_bound: f64,
    pub kth_best_threshold: f64,
}

/// Selects exact top-k candidates. The frontier currently begins at completed
/// family states; this permits exact pruning during bounded selection while
/// family generators without a verified construction-time bound continue to
/// use their existing exhaustive bounded generation.
pub fn exact_branch_and_bound_top_k<F>(
    candidates: Vec<SplitCandidate>,
    config: &BranchAndBoundConfig,
    score_config: &SplitScoreConfig,
    mask_words: F,
) -> (Vec<SplitCandidate>, BranchAndBoundDiagnostics)
where
    F: Fn(&Predicate) -> Arc<[u64]>,
{
    let requested = config.top_k.min(candidates.len());
    if !config.enabled {
        return (
            top_k_with_config(candidates, requested, score_config),
            BranchAndBoundDiagnostics::default(),
        );
    }
    if candidates.len() > config.max_partial_states
        || candidates.len() > config.max_candidates
        || candidates
            .iter()
            .any(|candidate| !candidate.score.final_score.is_finite())
    {
        // Correctness takes precedence even if a caller disables the preference:
        // no unsafe truncation is performed.
        let diagnostics = BranchAndBoundDiagnostics {
            exhaustive_fallback_count: 1,
            complete_candidates_evaluated: candidates.len(),
            ..BranchAndBoundDiagnostics::default()
        };
        return (
            top_k_with_config(candidates, requested, score_config),
            diagnostics,
        );
    }

    let mut states: Vec<_> = candidates
        .into_iter()
        .map(|candidate| {
            let key = super::canonical_predicate_key(&candidate.predicate);
            let chosen = vec![key.clone()];
            let family_state = match &candidate.predicate {
                Predicate::Unary(_) => FamilyPartialState::Unary {
                    chosen,
                    remaining: Vec::new(),
                },
                Predicate::HornClause(_) => FamilyPartialState::Horn {
                    chosen,
                    remaining: Vec::new(),
                },
                Predicate::AntiHornClause(_) => FamilyPartialState::AntiHorn {
                    chosen,
                    remaining: Vec::new(),
                },
                Predicate::Square2Cnf { .. } => FamilyPartialState::Square2Cnf {
                    chosen,
                    remaining: Vec::new(),
                },
                Predicate::Affine { .. } => FamilyPartialState::Affine {
                    chosen,
                    remaining: Vec::new(),
                },
                Predicate::EmpiricalAffine { .. } => FamilyPartialState::Affine {
                    chosen,
                    remaining: Vec::new(),
                },
            };
            PartialCandidateState {
                family_state,
                partial_mask_words: mask_words(&candidate.predicate),
                current_class_counts: vec![candidate.left_count, candidate.right_count],
                current_score: candidate.score.final_score,
                safe_upper_bound: candidate.score.final_score,
                canonical_state_key: key,
                candidate,
            }
        })
        .collect();
    states.sort_by(|left, right| {
        right
            .safe_upper_bound
            .total_cmp(&left.safe_upper_bound)
            .then_with(|| compare_candidates(&left.candidate, &right.candidate, score_config))
    });

    let mut diagnostics = BranchAndBoundDiagnostics {
        partial_states_created: states.len(),
        best_bound: states.first().map_or(0.0, |state| state.safe_upper_bound),
        ..BranchAndBoundDiagnostics::default()
    };
    let total = states.len();
    let mut selected = Vec::with_capacity(requested);
    for state in states {
        if selected.len() == requested && requested > 0 {
            let threshold = selected
                .last()
                .map_or(f64::NEG_INFINITY, |candidate: &SplitCandidate| {
                    candidate.score.final_score
                });
            diagnostics.kth_best_threshold = threshold;
            if state.safe_upper_bound < threshold {
                diagnostics.partial_states_pruned +=
                    total - diagnostics.partial_states_expanded - diagnostics.partial_states_pruned;
                break;
            }
        }
        diagnostics.partial_states_expanded += 1;
        diagnostics.complete_candidates_evaluated += 1;
        selected.push(state.candidate);
        selected.sort_by(|left, right| compare_candidates(left, right, score_config));
        selected.truncate(requested);
    }
    if requested > 0 {
        diagnostics.kth_best_threshold = selected
            .last()
            .map_or(0.0, |candidate| candidate.score.final_score);
    }
    diagnostics.percentage_reduction = if total == 0 {
        0.0
    } else {
        100.0 * diagnostics.partial_states_pruned as f64 / total as f64
    };
    (selected, diagnostics)
}
