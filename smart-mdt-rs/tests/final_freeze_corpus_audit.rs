use smart_mdt_rs::data::{binarize_labels_python, load_dl8_with_metadata};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

fn discover(root: &Path) -> Vec<PathBuf> {
    fn visit(dir: &Path, files: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().and_then(|value| value.to_str()) == Some("dl8") {
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    visit(root, &mut files);
    files.sort();
    files
}

fn raw_labels(path: &Path) -> Vec<i32> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            (!line.is_empty() && !line.starts_with('#')).then(|| {
                line.split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<i32>()
                    .unwrap()
            })
        })
        .collect()
}

#[test]
fn final_freeze_corpus_has_46_strict_deterministic_leakage_free_datasets() {
    let root = Path::new("../data");
    let files = discover(root);
    assert_eq!(
        files,
        discover(root),
        "dataset discovery is not deterministic"
    );
    assert_eq!(files.len(), 46, "unexpected dataset count");

    let stems = files
        .iter()
        .map(|path| path.file_stem().unwrap().to_string_lossy().into_owned())
        .collect::<BTreeSet<_>>();
    assert_eq!(stems.len(), files.len(), "duplicate dataset stems");

    for path in files {
        let loaded = load_dl8_with_metadata(&path).unwrap();
        let metadata = &loaded.metadata;
        assert!(!metadata.skipped, "{} was skipped", metadata.dataset);
        assert!(metadata.skip_reason.is_empty(), "{}", metadata.dataset);
        assert_eq!(metadata.label_column_used, 0, "{}", metadata.dataset);
        assert!(
            metadata.label_excluded_from_features,
            "{}",
            metadata.dataset
        );
        assert_eq!(
            metadata.feature_equal_to_label_count, 0,
            "{} leaks through retained feature(s) {}",
            metadata.dataset, metadata.feature_equal_to_label_indices
        );
        assert!(
            !metadata.suspicious_feature_label_leakage,
            "{}",
            metadata.dataset
        );
        assert!(metadata.n_samples > 0, "{}", metadata.dataset);
        assert!(metadata.raw_label_unique_count >= 2, "{}", metadata.dataset);
        assert!(
            metadata.n_features_after_constant_removal > 0,
            "{}",
            metadata.dataset
        );
        assert_eq!(
            metadata.n_features_original,
            metadata.n_features_after_constant_removal + metadata.removed_constant_columns_count,
            "{}",
            metadata.dataset
        );

        let dataset = loaded.dataset.unwrap();
        assert_eq!(dataset.features.rows(), metadata.n_samples);
        assert_eq!(
            dataset.features.cols(),
            metadata.n_features_after_constant_removal
        );
        assert_eq!(dataset.labels.len(), metadata.n_samples);
        assert_eq!(
            dataset.labels.iter().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 1]),
            "{}",
            metadata.dataset
        );

        let raw = raw_labels(&path);
        let independently_binarized = binarize_labels_python(&raw)
            .into_iter()
            .map(|label| label as u32)
            .collect::<Vec<_>>();
        assert_eq!(
            dataset.labels, independently_binarized,
            "{} binarization mismatch",
            metadata.dataset
        );

        let actual_binary = (0..dataset.features.cols() as u32).all(|feature| {
            dataset
                .features
                .column(feature)
                .iter()
                .all(|value| *value == 0.0 || *value == 1.0)
        });
        assert_eq!(
            metadata.is_binary_features, actual_binary,
            "{} binary-feature metadata mismatch",
            metadata.dataset
        );

        for feature in 0..dataset.features.cols() as u32 {
            let column = dataset.features.column(feature);
            assert!(
                column.windows(2).any(|pair| pair[0] != pair[1]),
                "{} retained a constant feature {feature}",
                metadata.dataset
            );
            assert!(
                !column
                    .iter()
                    .zip(&dataset.labels)
                    .all(|(value, label)| *value == f64::from(*label)),
                "{} retained a binarized-label copy at {feature}",
                metadata.dataset
            );
            assert!(
                !column
                    .iter()
                    .zip(&dataset.labels)
                    .all(|(value, label)| *value == f64::from(1 - *label)),
                "{} retained a binarized-label complement at {feature}",
                metadata.dataset
            );
            assert!(
                !column
                    .iter()
                    .zip(&raw)
                    .all(|(value, label)| *value == f64::from(*label)),
                "{} retained a raw-label copy at {feature}",
                metadata.dataset
            );
        }
    }
}
