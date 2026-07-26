use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    tree::{
        learn, predict_all,
        serialize::{from_json, to_json},
        tree_is_certified, LanguagePolicy, LearnerConfig,
    },
};

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn random_dataset(seed: &mut u64, case: usize) -> Dataset {
    let rows = 1 + next(seed) as usize % 12;
    let features = 1 + next(seed) as usize % 5;
    let mut matrix = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut values = Vec::with_capacity(features);
        for feature in 0..features {
            let value = match case % 8 {
                0 => 0.0,
                1 => 1.0,
                2 => (row & 1) as f64,
                3 if feature > 0 => values[0],
                4 => ((row + feature) % 3) as f64,
                5 => (next(seed) & 1) as f64,
                _ => (next(seed) % 4) as f64 - 1.0,
            };
            values.push(value);
        }
        matrix.push(values);
    }
    let labels = (0..rows)
        .map(|row| match case % 7 {
            0 => 0,
            1 => u32::from(row + 1 == rows),
            2 => u32::from(row & 1 == 1),
            _ => (next(seed) & 1) as u32,
        })
        .collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&matrix).unwrap(), labels).unwrap()
}

#[test]
fn randomized_corner_datasets_never_panic_and_preserve_tree_invariants() {
    let policies = [
        LanguagePolicy::UnaryOnly,
        LanguagePolicy::HornOnly,
        LanguagePolicy::AntiHornOnly,
        LanguagePolicy::Square2CnfOnly,
        LanguagePolicy::AffineOnly,
        LanguagePolicy::SmartCertified,
    ];
    let mut seed = 0xd36e_4a91_72bc_058fu64;
    for case in 0..512 {
        let dataset = random_dataset(&mut seed, case);
        let max_depth = case % 8;
        let config = LearnerConfig {
            max_depth,
            max_candidates_per_node: 16,
            beam_width: 8,
            language_policy: policies[case % policies.len()],
            theorem_mode: true,
            random_seed: case as u64,
            ..LearnerConfig::default()
        };
        let tree = learn(&dataset, &config).unwrap();
        assert!(tree.depth() <= max_depth);
        assert_eq!(tree.nodes() % 2, 1);
        assert!(tree_is_certified(&tree));
        let predictions = predict_all(&tree, &dataset.features);
        assert_eq!(predictions.len(), dataset.labels.len());

        let json = to_json(&tree).unwrap();
        let restored = from_json(&json).unwrap();
        assert_eq!(restored, tree);
        assert_eq!(predict_all(&restored, &dataset.features), predictions);

        if case < 32 {
            assert_eq!(learn(&dataset, &config).unwrap(), tree);
        }
    }
}

#[test]
fn empty_zero_feature_and_very_deep_requests_are_safe() {
    let empty = Dataset::new(ColumnMajorMatrix::from_rows(&[]).unwrap(), vec![]).unwrap();
    let empty_tree = learn(&empty, &LearnerConfig::default()).unwrap();
    assert_eq!(empty_tree.nodes(), 1);
    assert!(predict_all(&empty_tree, &empty.features).is_empty());

    let no_features = Dataset::new(
        ColumnMajorMatrix::from_rows(&[vec![], vec![]]).unwrap(),
        vec![0, 1],
    )
    .unwrap();
    let deep = learn(
        &no_features,
        &LearnerConfig {
            max_depth: 10_000,
            language_policy: LanguagePolicy::SmartCertified,
            ..LearnerConfig::default()
        },
    )
    .unwrap();
    assert_eq!(deep.nodes(), 1);
    assert!(tree_is_certified(&deep));
}
