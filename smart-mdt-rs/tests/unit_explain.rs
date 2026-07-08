use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    explain::extract_axp_deletion,
    tree::{learn, LanguagePolicy, LearnerConfig},
};
#[test]
fn axp_deletion_runs() {
    let ds = Dataset::new(
        ColumnMajorMatrix::from_rows(&[vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap(),
        vec![0, 0, 1, 1],
    )
    .unwrap();
    let cfg = LearnerConfig {
        language_policy: LanguagePolicy::UnaryOnly,
        ..LearnerConfig::default()
    };
    let t = learn(&ds, &cfg).unwrap();
    let axp = extract_axp_deletion(&t, &ds.features, 0, true);
    assert!(axp.metadata.theorem_certified);
}
