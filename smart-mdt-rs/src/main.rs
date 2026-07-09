use smart_mdt_rs::{
    data::Dataset,
    eval::{
        run_debug_candidates, run_full_benchmark, run_quick, BenchmarkConfig, DebugCandidateConfig,
    },
    tree::serialize::to_json,
    tree::{learn, LanguagePolicy, LearnerConfig},
    Result,
};
use std::path::PathBuf;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn policy(s: &str) -> LanguagePolicy {
    match s {
        "unary" => LanguagePolicy::UnaryOnly,
        "horn" => LanguagePolicy::HornOnly,
        "antihorn" => LanguagePolicy::AntiHornOnly,
        "square2cnf" => LanguagePolicy::Square2CnfOnly,
        "affine" => LanguagePolicy::AffineOnly,
        _ => LanguagePolicy::BestCertifiedPerNode,
    }
}

fn parse_usize_list(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
}

fn parse_method_list(s: &str) -> Vec<String> {
    s.split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("train") => {
            let data = arg(&args, "--data").ok_or_else(|| {
                smart_mdt_rs::SmartMdtError::InvalidInput("--data required".into())
            })?;
            let method = arg(&args, "--method").unwrap_or_else(|| "horn".into());
            let max_depth = arg(&args, "--max-depth")
                .and_then(|s| s.parse().ok())
                .unwrap_or(5);
            let ds = Dataset::from_dl8_like(data)?;
            let cfg = LearnerConfig {
                max_depth,
                language_policy: policy(&method),
                ..LearnerConfig::default()
            };
            let tree = learn(&ds, &cfg)?;
            println!("{}", to_json(&tree)?);
        }
        Some("benchmark") => {
            let output = PathBuf::from(
                arg(&args, "--output")
                    .unwrap_or_else(|| "experiment_artifacts/smart_mdt_rs/".into()),
            );
            if has_flag(&args, "--quick") || arg(&args, "--data").is_none() {
                let rows = run_quick(output)?;
                println!("wrote {} quick benchmark rows", rows.len());
            } else {
                let data_dir = PathBuf::from(arg(&args, "--data").unwrap_or_default());
                let depths = arg(&args, "--depths")
                    .map(|s| parse_usize_list(&s))
                    .filter(|v| !v.is_empty())
                    .unwrap_or_else(|| vec![5, 7]);
                let runs = arg(&args, "--runs")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10);
                let methods = arg(&args, "--methods")
                    .map(|s| parse_method_list(&s))
                    .filter(|v| !v.is_empty())
                    .unwrap_or_else(|| {
                        vec![
                            "unary".into(),
                            "horn".into(),
                            "antihorn".into(),
                            "square2cnf".into(),
                        ]
                    });
                let seed = arg(&args, "--seed")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(42);
                let cfg = BenchmarkConfig {
                    data_dir,
                    depths,
                    runs,
                    methods,
                    output,
                    seed,
                    strict_data_checks: has_flag(&args, "--strict-data-checks"),
                };
                let rows = run_full_benchmark(&cfg)?;
                println!("wrote {} dataset benchmark rows", rows.len());
            }
        }

        Some("debug-candidates") => {
            let data_dir = PathBuf::from(arg(&args, "--data").ok_or_else(|| {
                smart_mdt_rs::SmartMdtError::InvalidInput("--data required".into())
            })?);
            let dataset = arg(&args, "--dataset").ok_or_else(|| {
                smart_mdt_rs::SmartMdtError::InvalidInput("--dataset required".into())
            })?;
            let method = arg(&args, "--method").unwrap_or_else(|| "unary".into());
            let depth = arg(&args, "--depth")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let node_path = arg(&args, "--node-path").unwrap_or_else(|| "root".into());
            let top_k = arg(&args, "--top-k")
                .and_then(|s| s.parse().ok())
                .unwrap_or(20);
            let output =
                PathBuf::from(arg(&args, "--output").unwrap_or_else(|| "debug_candidates".into()));
            let seed = arg(&args, "--seed")
                .and_then(|s| s.parse().ok())
                .unwrap_or(42);
            let max_candidates_per_node = arg(&args, "--max-candidates-per-node")
                .and_then(|s| s.parse().ok())
                .unwrap_or(128);
            let beam_width = arg(&args, "--beam-width")
                .and_then(|s| s.parse().ok())
                .unwrap_or(32);
            let cfg = DebugCandidateConfig {
                data_dir,
                dataset,
                method,
                depth,
                node_path,
                top_k,
                output,
                seed,
                max_candidates_per_node,
                beam_width,
            };
            let rows = run_debug_candidates(&cfg)?;
            println!("wrote {} debug candidates", rows.len());
        }
        Some("explain") => {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(
                "explain requires serialized JSON phase; train/benchmark are available".into(),
            ));
        }
        _ => println!("usage: smart-mdt-rs train|benchmark|debug-candidates|explain"),
    }
    Ok(())
}
