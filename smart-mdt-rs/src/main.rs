use smart_mdt_rs::{
    data::Dataset,
    eval::run_quick,
    tree::serialize::to_json,
    tree::{learn, LanguagePolicy, LearnerConfig},
    Result,
};
fn arg(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}
fn policy(s: &str) -> LanguagePolicy {
    match s {
        "unary" => LanguagePolicy::UnaryOnly,
        "horn" => LanguagePolicy::HornOnly,
        "antihorn" => LanguagePolicy::AntiHornOnly,
        "square2cnf" => LanguagePolicy::Square2CnfOnly,
        _ => LanguagePolicy::BestCertifiedPerNode,
    }
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
            let output = arg(&args, "--output")
                .unwrap_or_else(|| "experiment_artifacts/smart_mdt_rs/".into());
            let rows = run_quick(output)?;
            println!("wrote {} benchmark rows", rows.len());
        }
        Some("explain") => {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(
                "explain requires serialized JSON phase; train/benchmark are available".into(),
            ));
        }
        _ => println!("usage: smart-mdt-rs train|benchmark|explain"),
    }
    Ok(())
}
