use smart_mdt_rs::eval::{run_full_benchmark, BenchmarkConfig};
use std::{path::PathBuf, process::Command};

#[test]
fn train_rejects_an_unknown_method_instead_of_changing_policy() {
    let output = Command::new(env!("CARGO_BIN_EXE_smart-mdt-rs"))
        .args([
            "train",
            "--data",
            "tests/fixtures/horn_separable.dl8",
            "--method",
            "unray",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown method unray"));
}

#[test]
fn benchmark_rejects_an_unknown_method_instead_of_silently_skipping_rows() {
    let output =
        std::env::temp_dir().join(format!("smart-mdt-invalid-method-{}", std::process::id()));
    let config = BenchmarkConfig {
        data_dir: PathBuf::from("tests/fixtures"),
        depths: vec![1],
        runs: 1,
        methods: vec!["unray".into()],
        output,
        seed: 42,
        strict_data_checks: false,
        cals: Default::default(),
        compact_explain: smart_mdt_rs::tree::CalsConfig::compact_explain(),
    };
    let error = run_full_benchmark(&config).unwrap_err();
    assert!(format!("{error}").contains("unknown benchmark method unray"));
}

#[test]
fn malformed_numeric_and_enum_arguments_are_rejected() {
    for args in [
        vec![
            "train",
            "--data",
            "tests/fixtures/horn_separable.dl8",
            "--max-depth",
            "five",
        ],
        vec![
            "benchmark",
            "--data",
            "tests/fixtures",
            "--depths",
            "5,nope",
        ],
        vec!["benchmark", "--quick", "--tree-search", "beamish"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_smart-mdt-rs"))
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("InvalidInput"));
    }
}
