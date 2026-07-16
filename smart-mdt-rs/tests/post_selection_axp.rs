use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    explain::extract_final_tree_axps,
    tree::{learn_with_diagnostics, LanguagePolicy, LearnerConfig, PruningConfig},
};

fn dataset() -> Dataset {
    let rows = (0..32)
        .map(|mask| {
            (0..5)
                .map(|bit| ((mask >> bit) & 1) as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let labels = rows
        .iter()
        .map(|row| u32::from(row[0] == 1.0 || row[1] == 1.0))
        .collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

#[test]
fn default_workflow_extracts_axps_only_after_final_tree_selection() {
    let data = dataset();
    let config = LearnerConfig {
        max_depth: 3,
        language_policy: LanguagePolicy::SmartCertified,
        pruning: PruningConfig {
            enabled: true,
            ..PruningConfig::default()
        },
        ..LearnerConfig::default()
    };
    assert!(!config.axp_rerank.enabled);
    let (final_tree, diagnostics) = learn_with_diagnostics(&data, &config).unwrap();
    assert_eq!(diagnostics.axp_rerank.candidates_evaluated, 0);

    let first = extract_final_tree_axps(&final_tree, &data.features, 8, true);
    let second = extract_final_tree_axps(&final_tree, &data.features, 8, true);
    assert_eq!(first.results.len(), 8);
    assert_eq!(first.mean_length, second.mean_length);
    assert_eq!(first.max_length, second.max_length);
    assert!(first.theorem_certified);
    assert_eq!(
        first
            .results
            .iter()
            .map(|result| (&result.features, &result.metadata))
            .collect::<Vec<_>>(),
        second
            .results
            .iter()
            .map(|result| (&result.features, &result.metadata))
            .collect::<Vec<_>>()
    );
}
