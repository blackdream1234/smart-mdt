use smart_mdt_rs::{
    data::Dataset,
    eval::{
        run_debug_candidates, run_full_benchmark, run_quick, BenchmarkConfig, DebugCandidateConfig,
    },
    explain::{
        compile_verified_explanation, render_human_explanation, verified_explanation_to_json,
        ExplanationAudience,
    },
    search::{SplitScoreConfig, SplitScoreProfile},
    tree::serialize::to_json,
    tree::{learn, CalsConfig, LanguagePolicy, LearnerConfig, TreeSearchStrategy},
    Result,
};
use std::{collections::BTreeSet, fs, path::PathBuf};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == name && !window[1].starts_with("--"))
        .map(|window| window[1].clone())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn option_takes_value(option: &str) -> bool {
    matches!(
        option,
        "--tree-search"
            | "--score-profile"
            | "--cals-profile"
            | "--audience"
            | "--depths"
            | "--method"
            | "--methods"
            | "--output"
            | "--data"
            | "--dataset"
            | "--node-path"
            | "--tree-beam-width"
            | "--candidate-beam-width"
            | "--lookahead-depth"
            | "--node-budget"
            | "--branch-and-bound-threshold"
            | "--cache-max-entries"
            | "--threads"
            | "--axp-shortlist"
            | "--max-depth"
            | "--runs"
            | "--depth"
            | "--top-k"
            | "--max-candidates-per-node"
            | "--beam-width"
            | "--row"
            | "--time-budget-ms"
            | "--seed"
            | "--balanced-accuracy-epsilon"
            | "--minimum-minority-recall"
            | "--root-collapse-majority-threshold"
            | "--prune-validation-fraction"
            | "--prune-alpha-nodes"
            | "--prune-alpha-leaves"
            | "--prune-alpha-literals"
            | "--accuracy-epsilon"
    )
}

fn is_boolean_option(option: &str) -> bool {
    matches!(
        option,
        "--branch-and-bound"
            | "--no-branch-and-bound"
            | "--cache"
            | "--no-cache"
            | "--parallel"
            | "--no-parallel"
            | "--adaptive-language"
            | "--no-adaptive-language"
            | "--prune"
            | "--no-prune"
            | "--class-aware-pruning"
            | "--axp-rerank"
            | "--quick"
            | "--strict-data-checks"
    )
}

fn is_cals_tuning_option(option: &str) -> bool {
    matches!(
        option,
        "--tree-search"
            | "--tree-beam-width"
            | "--candidate-beam-width"
            | "--lookahead-depth"
            | "--node-budget"
            | "--time-budget-ms"
            | "--score-profile"
            | "--branch-and-bound"
            | "--no-branch-and-bound"
            | "--branch-and-bound-threshold"
            | "--cache"
            | "--no-cache"
            | "--cache-max-entries"
            | "--parallel"
            | "--no-parallel"
            | "--threads"
            | "--adaptive-language"
            | "--no-adaptive-language"
            | "--prune"
            | "--no-prune"
            | "--class-aware-pruning"
            | "--balanced-accuracy-epsilon"
            | "--minimum-minority-recall"
            | "--root-collapse-majority-threshold"
            | "--prune-validation-fraction"
            | "--prune-alpha-nodes"
            | "--prune-alpha-leaves"
            | "--prune-alpha-literals"
            | "--accuracy-epsilon"
            | "--axp-rerank"
            | "--axp-shortlist"
            | "--cals-profile"
    )
}

fn validate_option_tokens(args: &[String]) -> Result<()> {
    let mut seen = BTreeSet::new();
    let mut index = 2;
    while let Some(option) = args.get(index) {
        if !option.starts_with("--") {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "unexpected positional argument {option}"
            )));
        }
        if !option_takes_value(option) && !is_boolean_option(option) {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "unknown option {option}"
            )));
        }
        if !seen.insert(option.as_str()) {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "duplicate option {option}"
            )));
        }
        if option_takes_value(option) {
            let Some(value) = args.get(index + 1) else {
                return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                    "{option} requires a value"
                )));
            };
            if value.starts_with("--") {
                return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                    "{option} requires a value"
                )));
            }
            index += 2;
        } else {
            index += 1;
        }
    }
    Ok(())
}

fn reject_mutually_exclusive_options(args: &[String]) -> Result<()> {
    for (enabled, disabled) in [
        ("--branch-and-bound", "--no-branch-and-bound"),
        ("--cache", "--no-cache"),
        ("--parallel", "--no-parallel"),
        ("--adaptive-language", "--no-adaptive-language"),
        ("--prune", "--no-prune"),
    ] {
        if has_flag(args, enabled) && has_flag(args, disabled) {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "{enabled} and {disabled} are mutually exclusive"
            )));
        }
    }
    Ok(())
}

fn validate_command_options(args: &[String]) -> Result<()> {
    let Some(command) = args.get(1).map(String::as_str) else {
        return Ok(());
    };
    let benchmark_is_quick =
        command == "benchmark" && (has_flag(args, "--quick") || arg(args, "--data").is_none());
    for option in args.iter().skip(2).filter(|value| value.starts_with("--")) {
        let allowed = match command {
            "train" => {
                matches!(option.as_str(), "--data" | "--method" | "--max-depth")
                    || is_cals_tuning_option(option)
            }
            "benchmark" if benchmark_is_quick => {
                matches!(option.as_str(), "--quick" | "--output")
            }
            "benchmark" => {
                matches!(
                    option.as_str(),
                    "--data"
                        | "--depths"
                        | "--runs"
                        | "--methods"
                        | "--output"
                        | "--seed"
                        | "--strict-data-checks"
                ) || is_cals_tuning_option(option)
            }
            "debug-candidates" => matches!(
                option.as_str(),
                "--data"
                    | "--dataset"
                    | "--method"
                    | "--depth"
                    | "--node-path"
                    | "--top-k"
                    | "--output"
                    | "--seed"
                    | "--max-candidates-per-node"
                    | "--beam-width"
            ),
            "explain" => {
                matches!(
                    option.as_str(),
                    "--data" | "--method" | "--max-depth" | "--row" | "--audience" | "--output"
                ) || is_cals_tuning_option(option)
            }
            _ => false,
        };
        if !allowed {
            let context = if benchmark_is_quick {
                "benchmark --quick"
            } else {
                command
            };
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "{option} is not supported with {context}"
            )));
        }
    }

    if matches!(command, "train" | "explain") {
        let default_method = if command == "train" {
            "horn"
        } else {
            "cals_compact_explain"
        };
        let method = arg(args, "--method").unwrap_or_else(|| default_method.into());
        for option in args
            .iter()
            .skip(2)
            .filter(|value| is_cals_tuning_option(value))
        {
            let accepted = method == "cals"
                || (method == "cals_compact_explain" && option != "--cals-profile");
            if !accepted {
                return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                    "{option} requires --method cals or cals_compact_explain"
                )));
            }
        }
    }

    if command == "benchmark" && !benchmark_is_quick {
        let methods = arg(args, "--methods").unwrap_or_default();
        let has_cals_method = methods
            .split(',')
            .map(str::trim)
            .any(|method| matches!(method, "cals" | "cals_compact_explain"));
        for option in args
            .iter()
            .skip(2)
            .filter(|value| is_cals_tuning_option(value))
        {
            if option == "--cals-profile" {
                return Err(smart_mdt_rs::SmartMdtError::InvalidInput(
                    "--cals-profile is not consumed by the benchmark command".into(),
                ));
            }
            if !has_cals_method {
                return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                    "{option} requires cals or cals_compact_explain in --methods"
                )));
            }
        }
    }
    Ok(())
}

fn validate_cli_args(args: &[String]) -> Result<()> {
    if let Some(command) = args.get(1) {
        if !matches!(
            command.as_str(),
            "train" | "benchmark" | "debug-candidates" | "explain"
        ) {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "unknown command {command}"
            )));
        }
    }
    validate_option_tokens(args)?;
    validate_command_options(args)?;
    reject_mutually_exclusive_options(args)?;
    for name in [
        "--tree-search",
        "--score-profile",
        "--cals-profile",
        "--audience",
        "--depths",
        "--method",
        "--methods",
        "--output",
        "--data",
        "--dataset",
        "--node-path",
    ] {
        if has_flag(args, name) && arg(args, name).is_none() {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "{name} requires a value"
            )));
        }
    }
    let usize_args = [
        "--tree-beam-width",
        "--candidate-beam-width",
        "--lookahead-depth",
        "--node-budget",
        "--branch-and-bound-threshold",
        "--cache-max-entries",
        "--threads",
        "--axp-shortlist",
        "--max-depth",
        "--runs",
        "--depth",
        "--top-k",
        "--max-candidates-per-node",
        "--beam-width",
        "--row",
    ];
    for name in usize_args {
        if has_flag(args, name) {
            let value = arg(args, name).ok_or_else(|| {
                smart_mdt_rs::SmartMdtError::InvalidInput(format!("{name} requires a value"))
            })?;
            value.parse::<usize>().map_err(|_| {
                smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                    "{name} requires a non-negative integer, got {value}"
                ))
            })?;
        }
    }
    for name in ["--time-budget-ms", "--seed"] {
        if has_flag(args, name) {
            let value = arg(args, name).ok_or_else(|| {
                smart_mdt_rs::SmartMdtError::InvalidInput(format!("{name} requires a value"))
            })?;
            value.parse::<u64>().map_err(|_| {
                smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                    "{name} requires a non-negative integer, got {value}"
                ))
            })?;
        }
    }
    for name in [
        "--balanced-accuracy-epsilon",
        "--minimum-minority-recall",
        "--root-collapse-majority-threshold",
        "--prune-validation-fraction",
        "--prune-alpha-nodes",
        "--prune-alpha-leaves",
        "--prune-alpha-literals",
        "--accuracy-epsilon",
    ] {
        if has_flag(args, name) {
            let value = arg(args, name).ok_or_else(|| {
                smart_mdt_rs::SmartMdtError::InvalidInput(format!("{name} requires a value"))
            })?;
            let parsed = value.parse::<f64>().map_err(|_| {
                smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                    "{name} requires a finite number, got {value}"
                ))
            })?;
            if !parsed.is_finite() {
                return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                    "{name} requires a finite number, got {value}"
                )));
            }
        }
    }
    if let Some(depths) = arg(args, "--depths") {
        if depths.is_empty()
            || depths
                .split(',')
                .any(|depth| depth.trim().parse::<usize>().is_err())
        {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "--depths requires a comma-separated integer list, got {depths}"
            )));
        }
    }
    if let Some(methods) = arg(args, "--methods") {
        if methods.split(',').any(|method| method.trim().is_empty()) {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "--methods requires a comma-separated method list, got {methods}"
            )));
        }
    }
    if let Some(value) = arg(args, "--tree-search") {
        if !matches!(
            value.as_str(),
            "greedy"
                | "beam"
                | "lookahead"
                | "sparse"
                | "sparse-lookahead"
                | "selective"
                | "selective-lookahead"
        ) {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "unknown tree-search strategy {value}"
            )));
        }
    }
    if let Some(value) = arg(args, "--score-profile") {
        if !matches!(
            value.as_str(),
            "information-gain" | "gain-ratio" | "sparse-certified" | "lookahead-ready"
        ) {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "unknown score profile {value}"
            )));
        }
    }
    if let Some(value) = arg(args, "--cals-profile") {
        if !matches!(
            value.as_str(),
            "thesis" | "compact-explain" | "compact_explain"
        ) {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "unknown CALS profile {value}"
            )));
        }
    }
    if let Some(value) = arg(args, "--audience") {
        if !matches!(
            value.as_str(),
            "general" | "clinical" | "engineering" | "management" | "audit" | "technical"
        ) {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "unknown explanation audience {value}"
            )));
        }
    }
    Ok(())
}

fn policy(s: &str) -> Result<LanguagePolicy> {
    match s {
        "unary" => Ok(LanguagePolicy::UnaryOnly),
        "horn" => Ok(LanguagePolicy::HornOnly),
        "antihorn" => Ok(LanguagePolicy::AntiHornOnly),
        "square2cnf" => Ok(LanguagePolicy::Square2CnfOnly),
        "affine" => Ok(LanguagePolicy::AffineOnly),
        "smart_certified" => Ok(LanguagePolicy::SmartCertified),
        "cals" | "cals_compact_explain" => Ok(LanguagePolicy::SmartCertified),
        "best-certified" => Ok(LanguagePolicy::BestCertifiedPerNode),
        _ => Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
            "unknown method {s}"
        ))),
    }
}

fn apply_cals_args(args: &[String], mut config: CalsConfig) -> CalsConfig {
    if let Some(strategy) = arg(args, "--tree-search") {
        config.tree_search.strategy = match strategy.as_str() {
            "greedy" => TreeSearchStrategy::Greedy,
            "beam" => TreeSearchStrategy::GlobalBeam,
            "selective" | "selective-lookahead" => TreeSearchStrategy::SelectiveLookahead,
            _ => TreeSearchStrategy::SparseLookahead,
        };
    }
    if let Some(width) = arg(args, "--tree-beam-width").and_then(|value| value.parse().ok()) {
        config.tree_search.tree_beam_width = width;
        config.tree_search.selective.tree_beam_width = width;
    }
    if let Some(width) = arg(args, "--candidate-beam-width").and_then(|value| value.parse().ok()) {
        config.tree_search.candidate_beam_width = width;
        config.tree_search.selective.candidate_beam_width = width;
    }
    if let Some(depth) = arg(args, "--lookahead-depth").and_then(|value| value.parse().ok()) {
        config.tree_search.lookahead_depth = depth;
        config.tree_search.selective.maximum_depth = depth;
    }
    if let Some(budget) = arg(args, "--node-budget").and_then(|value| value.parse().ok()) {
        config.tree_search.node_budget = budget;
    }
    if let Some(budget) = arg(args, "--time-budget-ms").and_then(|value| value.parse().ok()) {
        config.tree_search.time_budget_ms = Some(budget);
    }
    if let Some(profile) = arg(args, "--score-profile") {
        config.scoring = match profile.as_str() {
            "information-gain" => SplitScoreConfig::information_gain(),
            "gain-ratio" => SplitScoreConfig::gain_ratio(),
            "lookahead-ready" => SplitScoreConfig::lookahead_ready(),
            _ => SplitScoreConfig::sparse_certified(),
        };
        debug_assert!(matches!(
            config.scoring.profile,
            SplitScoreProfile::InformationGain
                | SplitScoreProfile::GainRatio
                | SplitScoreProfile::SparseCertified
                | SplitScoreProfile::LookaheadReady
        ));
    }
    if has_flag(args, "--branch-and-bound") {
        config.branch_and_bound.enabled = true;
    }
    if has_flag(args, "--no-branch-and-bound") {
        config.branch_and_bound.enabled = false;
    }
    if let Some(threshold) =
        arg(args, "--branch-and-bound-threshold").and_then(|value| value.parse().ok())
    {
        config.conditional_search.enabled = true;
        config
            .conditional_search
            .branch_and_bound_candidate_threshold = threshold;
    }
    if has_flag(args, "--cache") {
        config.cache.enabled = true;
    }
    if has_flag(args, "--no-cache") {
        config.cache.enabled = false;
    }
    if let Some(entries) = arg(args, "--cache-max-entries").and_then(|value| value.parse().ok()) {
        config.cache.max_entries = entries;
    }
    if has_flag(args, "--parallel") {
        config.parallel.enabled = true;
    }
    if has_flag(args, "--no-parallel") {
        config.parallel.enabled = false;
    }
    if let Some(threads) = arg(args, "--threads").and_then(|value| value.parse().ok()) {
        config.parallel.enabled = true;
        config.parallel.threads = Some(threads);
    }
    if has_flag(args, "--adaptive-language") {
        config.adaptive_language.enabled = true;
    }
    if has_flag(args, "--no-adaptive-language") {
        config.adaptive_language.enabled = false;
    }
    if has_flag(args, "--prune") {
        config.pruning.enabled = true;
    }
    if has_flag(args, "--no-prune") {
        config.pruning.enabled = false;
    }
    if has_flag(args, "--class-aware-pruning") {
        config.pruning.class_aware.enabled = true;
    }
    if let Some(value) =
        arg(args, "--balanced-accuracy-epsilon").and_then(|value| value.parse().ok())
    {
        config.pruning.class_aware.balanced_accuracy_epsilon = value;
    }
    if let Some(value) = arg(args, "--minimum-minority-recall").and_then(|value| value.parse().ok())
    {
        config.pruning.class_aware.minimum_minority_recall = Some(value);
    }
    if let Some(value) =
        arg(args, "--root-collapse-majority-threshold").and_then(|value| value.parse().ok())
    {
        config.pruning.class_aware.root_collapse_majority_threshold = value;
    }
    if let Some(value) = arg(args, "--prune-validation-fraction").and_then(|v| v.parse().ok()) {
        config.pruning.validation_fraction = value;
    }
    if let Some(value) = arg(args, "--prune-alpha-nodes").and_then(|v| v.parse().ok()) {
        config.pruning.alpha_nodes = value;
    }
    if let Some(value) = arg(args, "--prune-alpha-leaves").and_then(|v| v.parse().ok()) {
        config.pruning.alpha_leaves = value;
    }
    if let Some(value) = arg(args, "--prune-alpha-literals").and_then(|v| v.parse().ok()) {
        config.pruning.alpha_literals = value;
    }
    if let Some(value) = arg(args, "--accuracy-epsilon").and_then(|v| v.parse().ok()) {
        config.pruning.accuracy_epsilon = value;
    }
    if has_flag(args, "--axp-rerank") {
        config.axp_rerank.enabled = true;
    }
    if let Some(size) = arg(args, "--axp-shortlist").and_then(|value| value.parse().ok()) {
        config.axp_rerank.shortlist_size = size;
    }
    config
}

fn cals_config(args: &[String]) -> CalsConfig {
    let base = match arg(args, "--cals-profile").as_deref() {
        Some("compact-explain") | Some("compact_explain") => CalsConfig::compact_explain(),
        _ => CalsConfig::thesis(),
    };
    apply_cals_args(args, base)
}

fn thesis_config(args: &[String]) -> CalsConfig {
    apply_cals_args(args, CalsConfig::thesis())
}

fn compact_explain_config(args: &[String]) -> CalsConfig {
    apply_cals_args(args, CalsConfig::compact_explain())
}

fn audience(value: &str) -> ExplanationAudience {
    match value {
        "general" => ExplanationAudience::General,
        "clinical" => ExplanationAudience::Clinical,
        "engineering" => ExplanationAudience::Engineering,
        "management" => ExplanationAudience::Management,
        "audit" => ExplanationAudience::Audit,
        _ => ExplanationAudience::Technical,
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
    validate_cli_args(&args)?;
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
            let cfg = if method == "cals_compact_explain" {
                compact_explain_config(&args).learner_config(max_depth, 42)
            } else if method == "cals" {
                cals_config(&args).learner_config(max_depth, 42)
            } else {
                LearnerConfig {
                    max_depth,
                    language_policy: policy(&method)?,
                    theorem_mode: method != "best-certified",
                    ..LearnerConfig::default()
                }
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
                    cals: thesis_config(&args),
                    compact_explain: compact_explain_config(&args),
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
            let data = arg(&args, "--data").ok_or_else(|| {
                smart_mdt_rs::SmartMdtError::InvalidInput("--data required".into())
            })?;
            let method = arg(&args, "--method").unwrap_or_else(|| "cals_compact_explain".into());
            let max_depth = arg(&args, "--max-depth")
                .and_then(|value| value.parse().ok())
                .unwrap_or(5);
            let row = arg(&args, "--row")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            let selected_audience =
                audience(&arg(&args, "--audience").unwrap_or_else(|| "technical".into()));
            let output = PathBuf::from(
                arg(&args, "--output").unwrap_or_else(|| "verified_explanation".into()),
            );
            let dataset = Dataset::from_dl8_like(data)?;
            let config = if method == "cals_compact_explain" {
                compact_explain_config(&args).learner_config(max_depth, 42)
            } else if method == "cals" {
                cals_config(&args).learner_config(max_depth, 42)
            } else {
                LearnerConfig {
                    max_depth,
                    language_policy: policy(&method)?,
                    theorem_mode: method != "best-certified",
                    ..LearnerConfig::default()
                }
            };
            let tree = learn(&dataset, &config)?;
            let explanation =
                compile_verified_explanation(&tree, &dataset, row, selected_audience)?;
            let json = verified_explanation_to_json(&explanation)?;
            let text = render_human_explanation(&json)?;
            fs::create_dir_all(&output)?;
            fs::write(output.join("verified_explanation.json"), json)?;
            fs::write(output.join("human_explanation.txt"), text)?;
            println!("wrote verified_explanation.json and human_explanation.txt");
        }
        Some(command) => {
            return Err(smart_mdt_rs::SmartMdtError::InvalidInput(format!(
                "unknown command {command}"
            )))
        }
        None => println!("usage: smart-mdt-rs train|benchmark|debug-candidates|explain"),
    }
    Ok(())
}
