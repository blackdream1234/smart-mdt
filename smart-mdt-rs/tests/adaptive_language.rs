use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    logic::{LanguageFamily, PathTheoryState},
    tree::{
        allocate_family_budgets, learn, tree_is_certified, AdaptiveLanguageConfig,
        CandidateGenerationConfig, FamilyPilotMetrics, LanguagePolicy, LearnerConfig, NodeView,
        TrainingContext,
    },
};
use std::sync::Arc;

fn pilots() -> Vec<FamilyPilotMetrics> {
    [
        LanguageFamily::Unary,
        LanguageFamily::Horn,
        LanguageFamily::AntiHorn,
        LanguageFamily::Square2Cnf,
        LanguageFamily::Affine,
    ]
    .into_iter()
    .enumerate()
    .map(|(index, family)| FamilyPilotMetrics {
        family,
        best_score: 0.1 * index as f64,
        mean_top_k_score: 0.05 * index as f64,
        best_gain: 0.1,
        score_per_literal: 0.02,
        generation_cost: index as f64,
        duplicate_mask_rate: 0.0,
        branch_and_bound_pruning_rate: 0.0,
        candidates_generated: 4,
    })
    .collect()
}

fn dataset() -> Dataset {
    let rows: Vec<Vec<f64>> = (0..16)
        .map(|mask| (0..4).map(|bit| ((mask >> bit) & 1) as f64).collect())
        .collect();
    let labels = rows
        .iter()
        .map(|row| u32::from((row[0] == 1.0) ^ (row[1] == 1.0)))
        .collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

#[test]
fn budgets_are_exact_deterministic_and_preserve_minimum_quotas() {
    let config = AdaptiveLanguageConfig {
        enabled: true,
        total_candidate_budget: 30,
        minimum_family_quota: 3,
        pilot_candidates_per_family: 2,
        temperature: 0.5,
        diversity_top_k_per_family: 2,
    };
    let first = allocate_family_budgets(&config, &pilots());
    let second = allocate_family_budgets(&config, &pilots());
    assert_eq!(first, second);
    assert_eq!(
        first
            .iter()
            .map(|budget| budget.final_budget)
            .sum::<usize>(),
        30
    );
    assert!(first.iter().all(|budget| budget.final_budget >= 3));
    assert!(first.iter().all(|budget| budget.final_budget < 30));
}

#[test]
fn incompatible_families_receive_no_pilot_or_budget() {
    let context = TrainingContext::new(Arc::new(dataset()));
    let node = NodeView {
        rows: context.root_view().rows,
        depth: 1,
        theory_state: PathTheoryState::Horn,
    };
    let config = AdaptiveLanguageConfig {
        enabled: true,
        total_candidate_budget: 12,
        minimum_family_quota: 2,
        ..AdaptiveLanguageConfig::default()
    };
    let candidates = context
        .generate_candidates_adaptive(
            &node,
            CandidateGenerationConfig {
                policy: LanguagePolicy::SmartCertified,
                min_leaf: 1,
                beam: 8,
                score: &Default::default(),
                parallel: &Default::default(),
                adaptive: &config,
            },
        )
        .unwrap();
    assert!(candidates.iter().all(|candidate| matches!(
        candidate.predicate.language(),
        LanguageFamily::Unary | LanguageFamily::Horn
    )));
    let diagnostics = context.diagnostics();
    let node_diagnostics = diagnostics.adaptive_language.nodes.last().unwrap();
    assert_eq!(
        node_diagnostics.compatible_families,
        vec![LanguageFamily::Unary, LanguageFamily::Horn]
    );
    assert_eq!(
        node_diagnostics
            .budgets
            .iter()
            .map(|budget| budget.final_budget)
            .sum::<usize>(),
        12
    );
}

#[test]
fn disabled_adaptation_reproduces_fixed_allocation_and_enabled_tree_is_certified() {
    let data = dataset();
    let base = LearnerConfig {
        max_depth: 3,
        beam_width: 10,
        language_policy: LanguagePolicy::SmartCertified,
        ..LearnerConfig::default()
    };
    let fixed = learn(&data, &base).unwrap();
    let disabled = learn(
        &data,
        &LearnerConfig {
            adaptive_language: AdaptiveLanguageConfig {
                enabled: false,
                total_candidate_budget: 1,
                ..AdaptiveLanguageConfig::default()
            },
            ..base.clone()
        },
    )
    .unwrap();
    assert_eq!(fixed, disabled);
    let adaptive = learn(
        &data,
        &LearnerConfig {
            adaptive_language: AdaptiveLanguageConfig {
                enabled: true,
                total_candidate_budget: 24,
                minimum_family_quota: 2,
                ..AdaptiveLanguageConfig::default()
            },
            ..base
        },
    )
    .unwrap();
    assert!(tree_is_certified(&adaptive));
}
