use smart_mdt_rs::data::{binarize_labels_python, load_dl8_with_metadata};
use std::fs;

fn write_fixture(name: &str, body: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("smart-mdt-{name}-{}.dl8", std::process::id()));
    fs::write(&path, body).unwrap();
    path
}

#[test]
fn python_binarization_equivalence_cases() {
    assert_eq!(binarize_labels_python(&[0, 1]), vec![0, 1]);
    assert_eq!(binarize_labels_python(&[1, 2]), vec![0, 1]);
    assert_eq!(binarize_labels_python(&[0, 1, 2, 2]), vec![0, 0, 1, 1]);
    assert_eq!(binarize_labels_python(&[7, 7]), vec![0, 0]);
}

#[test]
fn first_column_excluded_constants_removed_and_metadata_correct() {
    let path = write_fixture("metadata", "0 0 5 1\n1 1 5 0\n0 0 5 1\n1 1 5 0\n");
    let loaded = load_dl8_with_metadata(&path).unwrap();
    let ds = loaded.dataset.unwrap();
    assert_eq!(ds.labels, vec![0, 1, 0, 1]);
    assert_eq!(ds.features.cols(), 2); // constant middle feature removed
    assert_eq!(loaded.metadata.label_column_used, 0);
    assert!(loaded.metadata.label_excluded_from_features);
    assert_eq!(loaded.metadata.n_features_original, 3);
    assert_eq!(loaded.metadata.n_features_after_constant_removal, 2);
    assert_eq!(loaded.metadata.removed_constant_columns_count, 1);
    assert_eq!(loaded.metadata.raw_label_counts, "0:2;1:2");
    assert_eq!(loaded.metadata.binarized_label_counts, "0:2;1:2");
    let _ = fs::remove_file(path);
}

#[test]
fn leakage_detection_records_feature_equal_to_label() {
    let path = write_fixture("leak", "0 0 1\n1 1 0\n0 0 1\n1 1 0\n");
    let loaded = load_dl8_with_metadata(&path).unwrap();
    assert_eq!(loaded.metadata.feature_equal_to_label_count, 1);
    assert_eq!(loaded.metadata.feature_equal_to_label_indices, "0");
    assert!(loaded.metadata.suspicious_feature_label_leakage);
    let _ = fs::remove_file(path);
}

#[test]
fn one_class_target_is_skipped_after_binarization() {
    let path = write_fixture("oneclass", "5 0\n5 1\n5 0\n");
    let loaded = load_dl8_with_metadata(&path).unwrap();
    assert!(loaded.dataset.is_none());
    assert!(loaded.metadata.skipped);
    assert_eq!(
        loaded.metadata.skip_reason,
        "target_one_class_after_binarization"
    );
    let _ = fs::remove_file(path);
}
