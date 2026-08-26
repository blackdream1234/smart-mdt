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
        datasets: Vec::new(),
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

#[test]
fn missing_string_flag_value_cannot_consume_another_flag() {
    let output = Command::new(env!("CARGO_BIN_EXE_smart-mdt-rs"))
        .args(["benchmark", "--quick", "--output", "--strict-data-checks"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--output requires a value"));
}

#[test]
fn unknown_cli_flag_is_rejected_instead_of_being_ignored() {
    let output = Command::new(env!("CARGO_BIN_EXE_smart-mdt-rs"))
        .args(["benchmark", "--quick", "--rums", "1"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown option --rums"));
}

#[test]
fn unknown_subcommand_returns_an_explicit_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_smart-mdt-rs"))
        .arg("benchmrk")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown command benchmrk"));
}

#[test]
fn quick_benchmark_rejects_ignored_experiment_selection_flags() {
    let output = Command::new(env!("CARGO_BIN_EXE_smart-mdt-rs"))
        .args(["benchmark", "--quick", "--methods", "unary"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("--methods is not supported with benchmark --quick"));
}

#[test]
fn known_options_are_rejected_on_the_wrong_subcommand() {
    for args in [
        vec![
            "train",
            "--data",
            "tests/fixtures/horn_separable.dl8",
            "--depths",
            "5",
        ],
        vec!["debug-candidates", "--strict-data-checks"],
        vec![
            "explain",
            "--data",
            "tests/fixtures/horn_separable.dl8",
            "--runs",
            "1",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_smart-mdt-rs"))
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("is not supported with"));
    }
}

#[test]
fn mutually_exclusive_boolean_pairs_are_rejected() {
    for (enabled, disabled) in [
        ("--branch-and-bound", "--no-branch-and-bound"),
        ("--cache", "--no-cache"),
        ("--parallel", "--no-parallel"),
        ("--adaptive-language", "--no-adaptive-language"),
        ("--prune", "--no-prune"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_smart-mdt-rs"))
            .args([
                "train",
                "--data",
                "tests/fixtures/horn_separable.dl8",
                "--method",
                "cals",
                enabled,
                disabled,
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("are mutually exclusive"));
    }
}

#[test]
fn cals_tuning_options_require_a_cals_method() {
    for args in [
        vec![
            "train",
            "--data",
            "tests/fixtures/horn_separable.dl8",
            "--method",
            "horn",
            "--tree-search",
            "beam",
        ],
        vec![
            "explain",
            "--data",
            "tests/fixtures/horn_separable.dl8",
            "--method",
            "horn",
            "--prune",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_smart-mdt-rs"))
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr)
            .contains("requires --method cals or cals_compact_explain"));
    }
}

#[test]
fn full_benchmark_rejects_unused_cals_configuration() {
    for (args, expected) in [
        (
            vec![
                "benchmark",
                "--data",
                "tests/fixtures",
                "--methods",
                "unary",
                "--tree-search",
                "beam",
            ],
            "requires cals or cals_compact_explain in --methods",
        ),
        (
            vec![
                "benchmark",
                "--data",
                "tests/fixtures",
                "--methods",
                "cals",
                "--cals-profile",
                "thesis",
            ],
            "--cals-profile is not consumed by the benchmark command",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_smart-mdt-rs"))
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains(expected));
    }
}
