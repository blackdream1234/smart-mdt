use smart_mdt_rs::data::Dataset;
use std::fs;

#[test]
fn dl8_loader_uses_first_column_as_label_to_avoid_label_leakage() {
    let path = std::env::temp_dir().join(format!("smart-mdt-loader-{}.dl8", std::process::id()));
    fs::write(&path, "1 0 0 1\n0 1 1 0\n").unwrap();
    let ds = Dataset::from_dl8_like(&path).unwrap();
    assert_eq!(ds.labels, vec![1, 0]);
    assert_eq!(ds.features.cols(), 3);
    assert_eq!(ds.features.get(0, 0), 0.0);
    assert_eq!(ds.features.get(0, 2), 1.0);
    let _ = fs::remove_file(path);
}
