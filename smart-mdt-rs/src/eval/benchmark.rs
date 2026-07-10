use super::{accuracy, theorem_table_filter, BenchmarkWarning, ResultRow};
use crate::{
    data::{load_dl8_with_metadata, ColumnMajorMatrix, Dataset, DatasetMetadata},
    explain::extract_axp_deletion,
    logic::{Backend, LanguageFamily},
    tree::{learn, predict_all, tree_is_certified, LanguagePolicy, LearnerConfig},
    Result, SmartMdtError,
};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

/// Full benchmark configuration.
#[derive(Clone, Debug)]
pub struct BenchmarkConfig {
    /// Directory containing `.dl8` datasets.
    pub data_dir: PathBuf,
    /// Depth values to evaluate.
    pub depths: Vec<usize>,
    /// Number of deterministic train/test split repetitions.
    pub runs: usize,
    /// Method names to evaluate.
    pub methods: Vec<String>,
    /// Output directory.
    pub output: PathBuf,
    /// Base random seed.
    pub seed: u64,
    /// Fail on data leakage or invalid dataset metadata instead of warning/skipping.
    pub strict_data_checks: bool,
}

/// Runs a quick deterministic synthetic benchmark and writes CSV artifacts.
pub fn run_quick(output: impl AsRef<Path>) -> Result<Vec<ResultRow>> {
    let rows = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y = vec![0, 0, 1, 1];
    let ds = Dataset::new(ColumnMajorMatrix::from_rows(&rows)?, y)?;
    let methods = default_methods();
    run_dataset_methods(DatasetRunSpec {
        dataset_name: "synthetic_quick",
        ds: &ds,
        runs: &[0],
        depths: &[3],
        methods: &methods,
        output,
        seed: 42,
        measure_times: false,
    })
}

/// Runs the full recursive `.dl8` dataset benchmark protocol.
pub fn run_full_benchmark(cfg: &BenchmarkConfig) -> Result<Vec<ResultRow>> {
    let files = discover_dl8_files(&cfg.data_dir)?;
    if files.is_empty() {
        return Err(SmartMdtError::InvalidInput(format!(
            "no .dl8 files found under {}",
            cfg.data_dir.display()
        )));
    }
    let mut all_rows = Vec::new();
    let mut metadata = Vec::new();
    let mut warnings = Vec::new();
    for file in files {
        let loaded = load_dl8_with_metadata(&file)?;
        let meta = loaded.metadata.clone();
        collect_metadata_warnings(&meta, &mut warnings);
        validate_metadata(&meta, cfg.strict_data_checks)?;
        metadata.push(meta.clone());
        let Some(ds) = loaded.dataset else {
            continue;
        };
        let runs: Vec<usize> = (0..cfg.runs).collect();
        let mut rows = run_dataset_methods(DatasetRunSpec {
            dataset_name: &meta.dataset,
            ds: &ds,
            runs: &runs,
            depths: &cfg.depths,
            methods: &cfg.methods,
            output: &cfg.output,
            seed: cfg.seed,
            measure_times: true,
        })?;
        collect_result_warnings(&rows, &mut warnings);
        all_rows.append(&mut rows);
    }
    write_all_outputs(&cfg.output, &all_rows, &metadata, &warnings)?;
    Ok(all_rows)
}

fn default_methods() -> Vec<String> {
    vec![
        "unary".into(),
        "horn".into(),
        "antihorn".into(),
        "square2cnf".into(),
        "best-certified".into(),
    ]
}

fn discover_dl8_files(root: &Path) -> Result<Vec<PathBuf>> {
    fn visit(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit(&path, out)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("dl8") {
                out.push(path);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    visit(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn method_policy(method: &str) -> Option<(LanguagePolicy, LanguageFamily, Backend)> {
    match method {
        "unary" => Some((
            LanguagePolicy::UnaryOnly,
            LanguageFamily::Unary,
            Backend::StructuralHorn,
        )),
        "horn" => Some((
            LanguagePolicy::HornOnly,
            LanguageFamily::Horn,
            Backend::StructuralHorn,
        )),
        "antihorn" => Some((
            LanguagePolicy::AntiHornOnly,
            LanguageFamily::AntiHorn,
            Backend::StructuralAntiHorn,
        )),
        "square2cnf" => Some((
            LanguagePolicy::Square2CnfOnly,
            LanguageFamily::Square2Cnf,
            Backend::TwoSat,
        )),
        "best-certified" => Some((
            LanguagePolicy::BestCertifiedPerNode,
            LanguageFamily::Unary,
            Backend::StructuralHorn,
        )),
        _ => None,
    }
}

struct DatasetRunSpec<'a, P: AsRef<Path>> {
    dataset_name: &'a str,
    ds: &'a Dataset,
    runs: &'a [usize],
    depths: &'a [usize],
    methods: &'a [String],
    output: P,
    seed: u64,
    measure_times: bool,
}

fn run_dataset_methods<P: AsRef<Path>>(spec: DatasetRunSpec<'_, P>) -> Result<Vec<ResultRow>> {
    let DatasetRunSpec {
        dataset_name,
        ds,
        runs,
        depths,
        methods,
        output,
        seed,
        measure_times,
    } = spec;
    fs::create_dir_all(&output)?;
    let git_sha = git_sha();
    let mut rows = Vec::new();
    for &run in runs {
        let (train, test) = split_train_test(ds, seed.wrapping_add(run as u64))?;
        for &depth in depths {
            for method in methods {
                let Some((policy, declared_family, declared_backend)) = method_policy(method)
                else {
                    continue;
                };
                let cfg = LearnerConfig {
                    max_depth: depth,
                    language_policy: policy,
                    random_seed: seed.wrapping_add(run as u64),
                    ..LearnerConfig::default()
                };
                let train_start = Instant::now();
                let tree = learn(&train, &cfg)?;
                let train_time = if measure_times {
                    train_start.elapsed().as_secs_f64()
                } else {
                    0.0
                };

                let predict_start = Instant::now();
                let pred = predict_all(&tree, &test.features);
                let predict_time = if measure_times {
                    predict_start.elapsed().as_secs_f64()
                } else {
                    0.0
                };

                let axp_start = Instant::now();
                let (mean_axp_length, theorem_certified) = if test.features.rows() == 0 {
                    (0.0, tree_is_certified(&tree))
                } else {
                    let limit = test.features.rows().min(8);
                    let mut total = 0usize;
                    let mut certified = tree_is_certified(&tree);
                    for row in 0..limit {
                        let axp = extract_axp_deletion(&tree, &test.features, row, true);
                        total += axp.features.len();
                        certified &= axp.metadata.theorem_certified;
                    }
                    (total as f64 / limit as f64, certified)
                };
                let axp_time = if measure_times {
                    axp_start.elapsed().as_secs_f64()
                } else {
                    0.0
                };

                rows.push(ResultRow {
                    dataset: dataset_name.to_string(),
                    run,
                    depth,
                    method: method.clone(),
                    accuracy: accuracy(&test.labels, &pred),
                    train_time,
                    predict_time,
                    tree_nodes: tree.nodes(),
                    leaves: tree.leaves(),
                    max_depth_reached: tree.depth(),
                    mean_axp_length,
                    axp_time,
                    theorem_certified,
                    language_family: declared_family,
                    backend: declared_backend,
                    git_sha: git_sha.clone(),
                    config: format!("{:?}", &cfg),
                    random_state: seed.wrapping_add(run as u64),
                    n_runs: runs.len(),
                    train_test_split_protocol: "deterministic_hash_70_30_first_label".into(),
                });
            }
        }
    }
    write_all_outputs(output, &rows, &[], &[])?;
    Ok(rows)
}

fn split_train_test(ds: &Dataset, seed: u64) -> Result<(Dataset, Dataset)> {
    let n = ds.labels.len();
    let mut keyed: Vec<(u64, usize)> = (0..n).map(|i| (mix(seed ^ i as u64), i)).collect();
    keyed.sort_by_key(|x| x.0);
    let train_len = ((n as f64) * 0.7).round() as usize;
    let train_len = train_len.clamp(1, n.saturating_sub(1).max(1));
    let train_idx: Vec<_> = keyed.iter().take(train_len).map(|x| x.1).collect();
    let test_idx: Vec<_> = keyed.iter().skip(train_len).map(|x| x.1).collect();
    Ok((subset(ds, &train_idx)?, subset(ds, &test_idx)?))
}

fn subset(ds: &Dataset, rows: &[usize]) -> Result<Dataset> {
    let matrix_rows: Vec<Vec<f64>> = rows
        .iter()
        .map(|&i| {
            (0..ds.features.cols())
                .map(|j| ds.features.get(i, j as u32))
                .collect()
        })
        .collect();
    let labels = rows.iter().map(|&i| ds.labels[i]).collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&matrix_rows)?, labels)
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into())
        .trim()
        .to_string()
}

fn validate_metadata(meta: &DatasetMetadata, strict: bool) -> Result<()> {
    if !strict {
        return Ok(());
    }
    if meta.skipped {
        return Err(SmartMdtError::InvalidInput(format!(
            "strict data checks rejected {}: {}",
            meta.dataset, meta.skip_reason
        )));
    }
    if meta.n_features_after_constant_removal == 0 {
        return Err(SmartMdtError::InvalidInput(format!(
            "strict data checks rejected {}: no non-constant features",
            meta.dataset
        )));
    }
    if meta.raw_label_unique_count == 0 || meta.binarized_label_counts == "0:0" {
        return Err(SmartMdtError::InvalidInput(format!(
            "strict data checks rejected {}: impossible label metadata",
            meta.dataset
        )));
    }
    if meta.feature_equal_to_label_count > 0 {
        return Err(SmartMdtError::InvalidInput(format!(
            "strict data checks rejected {}: feature-label leakage at {}",
            meta.dataset, meta.feature_equal_to_label_indices
        )));
    }
    if !meta.label_excluded_from_features || meta.label_column_used != 0 {
        return Err(SmartMdtError::InvalidInput(format!(
            "strict data checks rejected {}: label column metadata invalid",
            meta.dataset
        )));
    }
    Ok(())
}

fn collect_metadata_warnings(meta: &DatasetMetadata, warnings: &mut Vec<BenchmarkWarning>) {
    if meta.majority_class_rate >= 0.99 {
        warnings.push(BenchmarkWarning {
            dataset: meta.dataset.clone(),
            run: String::new(),
            depth: String::new(),
            method: String::new(),
            warning_type: "suspicious_majority_rate".into(),
            message: "majority_class_rate >= 0.99".into(),
            value: meta.majority_class_rate.to_string(),
        });
    }
    if meta.feature_equal_to_label_count > 0 {
        warnings.push(BenchmarkWarning {
            dataset: meta.dataset.clone(),
            run: String::new(),
            depth: String::new(),
            method: String::new(),
            warning_type: "feature_label_leakage".into(),
            message: format!(
                "feature columns equal binarized label: {}",
                meta.feature_equal_to_label_indices
            ),
            value: meta.feature_equal_to_label_count.to_string(),
        });
    }
    if meta.skipped {
        warnings.push(BenchmarkWarning {
            dataset: meta.dataset.clone(),
            run: String::new(),
            depth: String::new(),
            method: String::new(),
            warning_type: "dataset_skipped".into(),
            message: meta.skip_reason.clone(),
            value: "1".into(),
        });
    }
}

fn collect_result_warnings(rows: &[ResultRow], warnings: &mut Vec<BenchmarkWarning>) {
    for r in rows {
        if r.accuracy >= 0.99 && r.tree_nodes <= 3 {
            warnings.push(BenchmarkWarning {
                dataset: r.dataset.clone(),
                run: r.run.to_string(),
                depth: r.depth.to_string(),
                method: r.method.clone(),
                warning_type: "high_accuracy_tiny_tree".into(),
                message: "accuracy >= 0.99 and tree_nodes <= 3".into(),
                value: r.accuracy.to_string(),
            });
        }
    }
    let mut methods: Vec<_> = rows.iter().map(|r| r.method.clone()).collect();
    methods.sort();
    methods.dedup();
    for method in methods {
        let selected: Vec<_> = rows.iter().filter(|r| r.method == method).collect();
        if !selected.is_empty() && selected.iter().all(|r| r.tree_nodes == 1) {
            warnings.push(BenchmarkWarning {
                dataset: String::new(),
                run: String::new(),
                depth: String::new(),
                method: method.clone(),
                warning_type: "method_all_constant_trees".into(),
                message: "method has tree_nodes == 1 for all rows in this benchmark slice".into(),
                value: selected.len().to_string(),
            });
        }
        if !selected.is_empty() && selected.iter().all(|r| r.mean_axp_length == 0.0) {
            warnings.push(BenchmarkWarning {
                dataset: String::new(),
                run: String::new(),
                depth: String::new(),
                method,
                warning_type: "method_all_zero_axp".into(),
                message: "method has mean_axp_length == 0 for all rows in this benchmark slice"
                    .into(),
                value: selected.len().to_string(),
            });
        }
    }
}

fn write_metadata_csv(path: impl AsRef<Path>, metadata: &[DatasetMetadata]) -> Result<()> {
    let mut out = String::from("dataset,path,n_samples,n_columns_original,n_features_original,n_features_after_constant_removal,raw_label_unique_count,raw_label_counts,binarized_label_counts,positive_rate,majority_class_rate,removed_constant_columns_count,is_binary_features,skipped,skip_reason,label_column_used,label_excluded_from_features,feature_equal_to_label_count,feature_equal_to_label_indices,suspicious_majority_rate,suspicious_feature_label_leakage\n");
    for m in metadata {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            csv_escape(&m.dataset),
            csv_escape(&m.path),
            m.n_samples,
            m.n_columns_original,
            m.n_features_original,
            m.n_features_after_constant_removal,
            m.raw_label_unique_count,
            csv_escape(&m.raw_label_counts),
            csv_escape(&m.binarized_label_counts),
            m.positive_rate,
            m.majority_class_rate,
            m.removed_constant_columns_count,
            m.is_binary_features,
            m.skipped,
            csv_escape(&m.skip_reason),
            m.label_column_used,
            m.label_excluded_from_features,
            m.feature_equal_to_label_count,
            csv_escape(&m.feature_equal_to_label_indices),
            m.suspicious_majority_rate,
            m.suspicious_feature_label_leakage
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn write_warnings_csv(path: impl AsRef<Path>, warnings: &[BenchmarkWarning]) -> Result<()> {
    for w in warnings {
        eprintln!(
            "benchmark warning [{}] dataset={} run={} depth={} method={} value={}: {}",
            w.warning_type, w.dataset, w.run, w.depth, w.method, w.value, w.message
        );
    }
    let mut out = String::from("dataset,run,depth,method,warning_type,message,value\n");
    for w in warnings {
        out.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            csv_escape(&w.dataset),
            w.run,
            w.depth,
            w.method,
            w.warning_type,
            csv_escape(&w.message),
            csv_escape(&w.value)
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn write_all_outputs(
    output: impl AsRef<Path>,
    rows: &[ResultRow],
    metadata: &[DatasetMetadata],
    warnings: &[BenchmarkWarning],
) -> Result<()> {
    fs::create_dir_all(&output)?;
    write_csv(output.as_ref().join("full_results.csv"), rows)?;
    write_metadata_csv(output.as_ref().join("dataset_metadata.csv"), metadata)?;
    write_warnings_csv(output.as_ref().join("benchmark_warnings.csv"), warnings)?;
    write_summary(output.as_ref().join("summary_by_method.csv"), rows)?;
    let cert: Vec<_> = rows
        .iter()
        .filter(|r| theorem_table_filter(r))
        .cloned()
        .collect();
    write_csv(output.as_ref().join("theorem_certified_results.csv"), &cert)?;
    let emp: Vec<ResultRow> = rows
        .iter()
        .filter(|r| !theorem_table_filter(r))
        .cloned()
        .collect();
    write_csv(output.as_ref().join("empirical_results.csv"), &emp)?;
    write_csv(output.as_ref().join("axp_metadata.csv"), rows)?;
    write_csv(output.as_ref().join("tuning_diagnostics.csv"), &emp)?;
    fs::write(
        output.as_ref().join("README_RESULTS.md"),
        format!(
            "# CGS-MDT benchmark results\n\nRows: {}\n\nThe theorem table is filtered to Unary, Horn, AntiHorn and Square2CNF methods with certified backends only.\n",
            rows.len()
        ),
    )?;
    Ok(())
}

fn write_summary(path: impl AsRef<Path>, rows: &[ResultRow]) -> Result<()> {
    let mut out = String::from("method,rows,accuracy_mean,tree_nodes_mean,mean_axp_length_mean\n");
    let mut methods: Vec<String> = rows.iter().map(|r| r.method.clone()).collect();
    methods.sort();
    methods.dedup();
    for method in methods {
        let selected: Vec<_> = rows.iter().filter(|r| r.method == method).collect();
        let n = selected.len() as f64;
        let acc = selected.iter().map(|r| r.accuracy).sum::<f64>() / n;
        let nodes = selected.iter().map(|r| r.tree_nodes as f64).sum::<f64>() / n;
        let axp = selected.iter().map(|r| r.mean_axp_length).sum::<f64>() / n;
        out.push_str(&format!(
            "{},{},{acc},{nodes},{axp}\n",
            method,
            selected.len()
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn csv_escape(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "'"))
}

fn method_label(method: &str) -> &str {
    match method {
        "unary" => "Unary",
        "horn" => "Horn",
        "antihorn" => "AntiHorn",
        "square2cnf" => "Square2CNF",
        "best-certified" => "Best certified per node",
        other => other,
    }
}

fn path_certificate(backend: Backend) -> &'static str {
    match backend {
        Backend::StructuralHorn => "HornCnf",
        Backend::StructuralAntiHorn => "AntiHornCnf",
        Backend::TwoSat => "TwoCnf",
        Backend::Gf2Gaussian => "Gf2System",
        Backend::Affine => "Empirical",
        _ => "Unsupported",
    }
}

fn write_csv(path: impl AsRef<Path>, rows: &[ResultRow]) -> Result<()> {
    let mut out = String::from("dataset,run,depth,method,accuracy,train_time,predict_time,tree_nodes,leaves,max_depth_reached,mean_axp_length,axp_time,theorem_certified,language_family,backend,git_sha,config,method_key,method_label,category,acc,acc_std,size,expl,axp_valid_rate,axp_minimal_rate,n_success,n_fail,axp_backend,path_certificate,rejected_reason,theorem_mode_used,random_state,n_runs,train_test_split_protocol\n");
    for r in rows {
        let category = if theorem_table_filter(r) {
            "certified"
        } else {
            "empirical_or_adaptive"
        };
        let axp_rate = if r.theorem_certified { 1.0 } else { 0.0 };
        let n_success = usize::from(r.theorem_certified);
        let n_fail = usize::from(!r.theorem_certified);
        let rejected_reason = if r.theorem_certified {
            ""
        } else {
            "not theorem-table eligible"
        };
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{:?},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{},{:?},{},{},{},{},{},{}\n",
            csv_escape(&r.dataset),
            r.run,
            r.depth,
            r.method,
            r.accuracy,
            r.train_time,
            r.predict_time,
            r.tree_nodes,
            r.leaves,
            r.max_depth_reached,
            r.mean_axp_length,
            r.axp_time,
            r.theorem_certified,
            r.language_family,
            r.backend,
            r.git_sha,
            csv_escape(&r.config),
            r.method,
            csv_escape(method_label(&r.method)),
            category,
            r.accuracy,
            0.0,
            r.tree_nodes,
            r.mean_axp_length,
            axp_rate,
            axp_rate,
            n_success,
            n_fail,
            r.backend,
            path_certificate(r.backend),
            csv_escape(rejected_reason),
            true,
            r.random_state,
            r.n_runs,
            r.train_test_split_protocol
        ));
    }
    fs::write(path, out)?;
    Ok(())
}
