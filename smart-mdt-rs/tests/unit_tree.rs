use smart_mdt_rs::{
    data::{ColumnMajorMatrix, Dataset},
    tree::{learn, predict_all, LanguagePolicy, LearnerConfig},
};
#[test]
fn tree_prediction_deterministic() {
    let ds = Dataset::new(
        ColumnMajorMatrix::from_rows(&[vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap(),
        vec![0, 0, 1, 1],
    )
    .unwrap();
    let cfg = LearnerConfig {
        language_policy: LanguagePolicy::UnaryOnly,
        ..LearnerConfig::default()
    };
    let t1 = learn(&ds, &cfg).unwrap();
    let t2 = learn(&ds, &cfg).unwrap();
    assert_eq!(t1, t2);
    assert_eq!(predict_all(&t1, &ds.features), vec![0, 0, 1, 1]);
}
