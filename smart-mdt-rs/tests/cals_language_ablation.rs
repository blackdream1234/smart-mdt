use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    eval::{run_full_benchmark, theorem_table_filter, BenchmarkConfig},
    logic::{AllowedLanguages, LanguageFamily},
    tree::{
        learn, tree_is_certified, CacheConfig, CalsConfig, CandidateGenerationConfig,
        LanguagePolicy, TrainingContext, TreeNode,
    },
};
use std::{collections::BTreeSet, fs, path::PathBuf, sync::Arc};

const FAMILIES: [LanguageFamily; 5] = [
    LanguageFamily::Unary,
    LanguageFamily::Horn,
    LanguageFamily::AntiHorn,
    LanguageFamily::Square2Cnf,
    LanguageFamily::Affine,
];

const RESTRICTED_METHODS: [&str; 10] = [
    "cals_unary",
    "cals_horn",
    "cals_antihorn",
    "cals_square2cnf",
    "cals_affine",
    "cals_compact_explain_unary",
    "cals_compact_explain_horn",
    "cals_compact_explain_antihorn",
    "cals_compact_explain_square2cnf",
    "cals_compact_explain_affine",
];

fn boolean_dataset() -> Dataset {
    let rows = (0..16)
        .map(|mask| {
            (0..4)
                .map(|bit| ((mask >> bit) & 1) as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let labels = rows
        .iter()
        .map(|row| u32::from((row[0] == 1.0) ^ (row[1] == 1.0)))
        .collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

fn family_dataset(family: LanguageFamily) -> Dataset {
    match family {
        LanguageFamily::Unary => {
            let rows = vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
            ];
            let labels = rows.iter().map(|row| u32::from(row[0] == 1.0)).collect();
            Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
        }
        LanguageFamily::Horn => {
            Dataset::from_dl8_like("tests/fixtures/horn_separable.dl8").unwrap()
        }
        LanguageFamily::AntiHorn => {
            Dataset::from_dl8_like("tests/fixtures/antihorn_separable.dl8").unwrap()
        }
        LanguageFamily::Square2Cnf => {
            Dataset::from_dl8_like("tests/fixtures/square2cnf_separable.dl8").unwrap()
        }
        LanguageFamily::Affine => Dataset::from_dl8_like("tests/fixtures/affine_xor8.dl8").unwrap(),
        other => panic!("unsupported test family {other:?}"),
    }
}

fn internal_families(tree: &TreeNode, output: &mut Vec<LanguageFamily>) {
    if let TreeNode::Internal {
        predicate,
        left,
        right,
        ..
    } = tree
    {
        output.push(predicate.language());
        internal_families(left, output);
        internal_families(right, output);
    }
}

fn assert_profile_restricted_to(profile: CalsConfig, family: LanguageFamily) {
    let config = profile
        .with_allowed_languages(AllowedLanguages::only(family))
        .learner_config(3, 42);
    let tree = learn(&family_dataset(family), &config).unwrap();
    let mut selected = Vec::new();
    internal_families(&tree, &mut selected);
    assert!(selected.iter().all(|selected| *selected == family));
    assert!(tree_is_certified(&tree));
}

#[test]
fn every_single_language_cals_and_compact_tree_uses_only_its_allowed_family() {
    for family in FAMILIES {
        assert_profile_restricted_to(CalsConfig::thesis(), family);
        assert_profile_restricted_to(CalsConfig::compact_explain(), family);
    }
}

#[test]
fn family_filter_runs_before_candidate_ranking() {
    let data = boolean_dataset();
    let context = TrainingContext::new(Arc::new(data));
    for family in FAMILIES {
        let candidates = context
            .generate_candidates_with_allowed(
                &context.root_view(),
                LanguagePolicy::SmartCertified,
                AllowedLanguages::only(family),
                1,
                16,
                &Default::default(),
            )
            .unwrap();
        assert!(!candidates.is_empty(), "{family:?} candidate pool is empty");
        assert!(candidates
            .iter()
            .all(|candidate| candidate.predicate.language() == family));
    }
}

#[test]
fn unrestricted_profiles_retain_multi_family_candidate_search() {
    let data = Arc::new(boolean_dataset());
    for profile in [CalsConfig::thesis(), CalsConfig::compact_explain()] {
        let config = profile.learner_config(3, 42);
        assert_eq!(config.allowed_languages, AllowedLanguages::all());
        let context = TrainingContext::new(data.clone());
        let candidates = context
            .generate_candidates_adaptive(
                &context.root_view(),
                CandidateGenerationConfig {
                    policy: config.language_policy,
                    allowed_languages: config.allowed_languages,
                    min_leaf: config.min_samples_leaf,
                    beam: config.beam_width,
                    score: &config.split_score,
                    parallel: &config.parallel,
                    adaptive: &config.adaptive_language,
                },
            )
            .unwrap();
        let families = candidates
            .iter()
            .map(|candidate| format!("{:?}", candidate.predicate.language()))
            .collect::<BTreeSet<_>>();
        assert!(families.len() > 1, "unrestricted candidates: {families:?}");
    }
}

#[test]
fn explicit_all_mask_preserves_existing_unrestricted_behavior() {
    let data = boolean_dataset();
    for profile in [CalsConfig::thesis(), CalsConfig::compact_explain()] {
        let implicit = learn(&data, &profile.learner_config(3, 42)).unwrap();
        let explicit = learn(
            &data,
            &profile
                .with_allowed_languages(AllowedLanguages::all())
                .learner_config(3, 42),
        )
        .unwrap();
        assert_eq!(explicit, implicit);
    }
}

#[test]
fn restricted_cache_on_off_preserves_deterministic_result() {
    let data = family_dataset(LanguageFamily::Horn);
    for profile in [CalsConfig::thesis(), CalsConfig::compact_explain()] {
        let cached = profile
            .clone()
            .with_allowed_languages(AllowedLanguages::only(LanguageFamily::Horn))
            .learner_config(3, 42);
        let mut uncached = cached.clone();
        uncached.cache = CacheConfig::disabled();
        assert_eq!(
            learn(&data, &cached).unwrap(),
            learn(&data, &uncached).unwrap()
        );
    }
}

#[test]
fn restricted_methods_report_configuration_and_pass_full_certification() {
    let base = std::env::temp_dir().join(format!(
        "smart-mdt-language-ablation-{}",
        std::process::id()
    ));
    let data_dir = base.join("data");
    let output = base.join("out");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&data_dir).unwrap();
    fs::copy(
        PathBuf::from("tests/fixtures/affine_xor8.dl8"),
        data_dir.join("affine_xor8.dl8"),
    )
    .unwrap();
    let rows = run_full_benchmark(&BenchmarkConfig {
        data_dir,
        depths: vec![2],
        runs: 1,
        methods: RESTRICTED_METHODS.iter().map(ToString::to_string).collect(),
        datasets: vec!["affine_xor8".into()],
        output: output.clone(),
        seed: 42,
        strict_data_checks: false,
        cals: CalsConfig::thesis(),
        compact_explain: CalsConfig::compact_explain(),
    })
    .unwrap();
    assert_eq!(rows.len(), RESTRICTED_METHODS.len());
    for row in &rows {
        assert_eq!(row.ablation_variant, "single_language");
        assert!(!row.allowed_language_set.contains('|'));
        assert!(matches!(
            row.optimizer_profile.as_str(),
            "cals" | "compact_explain"
        ));
        assert!(row.config.contains("allowed_languages: AllowedLanguages"));
        assert!(row.path_certified);
        assert!(row.theorem_certified);
        assert_eq!(row.final_axp_rows, row.test_rows);
        assert!(theorem_table_filter(row), "rejected row: {row:#?}");
    }
    let csv = fs::read_to_string(output.join("full_results.csv")).unwrap();
    let header = csv.lines().next().unwrap();
    for column in [
        "optimizer_profile",
        "allowed_language_set",
        "ablation_variant",
        "predicate_literals",
        "root_language",
    ] {
        assert!(header.split(',').any(|value| value == column));
    }
    let _ = fs::remove_dir_all(base);
}
