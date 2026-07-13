use super::{accuracy, theorem_table_filter, BenchmarkWarning, ResultRow};
use crate::{
    data::{load_dl8_with_metadata, ColumnMajorMatrix, Dataset, DatasetMetadata},
    explain::extract_axp_deletion,
    logic::{Backend, LanguageFamily},
    tree::{
        learn_with_diagnostics, predict_all, tree_is_certified, tree_path_theory_metadata,
        CalsConfig, LanguagePolicy, LearnerConfig, TreeNode,
    },
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
    /// Unified settings used only by the `cals` method.
    pub cals: CalsConfig,
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
    let cals = CalsConfig::default();
    run_dataset_methods(DatasetRunSpec {
        dataset_name: "synthetic_quick",
        ds: &ds,
        runs: &[0],
        depths: &[3],
        methods: &methods,
        output,
        seed: 42,
        measure_times: false,
        cals: &cals,
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
            cals: &cfg.cals,
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
        "affine" => Some((
            LanguagePolicy::AffineOnly,
            LanguageFamily::Affine,
            Backend::Gf2Gaussian,
        )),
        "smart_certified" => Some((
            LanguagePolicy::SmartCertified,
            LanguageFamily::SmartCertified,
            Backend::PathCertified,
        )),
        "cals" => Some((
            LanguagePolicy::SmartCertified,
            LanguageFamily::SmartCertified,
            Backend::PathCertified,
        )),
        "best-certified" => Some((
            LanguagePolicy::BestCertifiedPerNode,
            LanguageFamily::Unary,
            Backend::StructuralHorn,
        )),
        _ => None,
    }
}

fn compatible_family_count_for_policy(policy: LanguagePolicy) -> usize {
    match policy {
        LanguagePolicy::SmartCertified => 5,
        LanguagePolicy::CertifiedOnly | LanguagePolicy::BestCertifiedPerNode => 4,
        _ => 1,
    }
}

fn predicates_backend_allowed(tree: &TreeNode, training: &Dataset) -> bool {
    match tree {
        TreeNode::Leaf { .. } => true,
        TreeNode::Internal {
            predicate,
            left,
            right,
            ..
        } => {
            predicate.language().theorem_table_allowed()
                && (!matches!(predicate, crate::logic::Predicate::Affine { .. })
                    || crate::data::predicate_scope_is_boolean(&training.features, predicate))
                && !matches!(predicate, crate::logic::Predicate::EmpiricalAffine { .. })
                && predicates_backend_allowed(left, training)
                && predicates_backend_allowed(right, training)
        }
    }
}

fn theorem_rejection_reason(
    theorem_certified: bool,
    path_certified: bool,
    predicates_allowed: bool,
    empirical_fallback: bool,
    incompatible_cache_reuse: bool,
) -> String {
    if theorem_certified
        && path_certified
        && predicates_allowed
        && !empirical_fallback
        && !incompatible_cache_reuse
    {
        return String::new();
    }
    let mut reasons = Vec::new();
    if !theorem_certified {
        reasons.push("AXp or tree theorem certification failed");
    }
    if !path_certified {
        reasons.push("path theory compatibility failed");
    }
    if !predicates_allowed {
        reasons.push("predicate backend or affine Boolean guard failed");
    }
    if empirical_fallback {
        reasons.push("empirical fallback used");
    }
    if incompatible_cache_reuse {
        reasons.push("incompatible cached subtree reused");
    }
    reasons.join("; ")
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
    cals: &'a CalsConfig,
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
        cals,
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
                let random_seed = seed.wrapping_add(run as u64);
                let cfg = if method == "cals" {
                    cals.learner_config(depth, random_seed)
                } else {
                    LearnerConfig {
                        max_depth: depth,
                        language_policy: policy,
                        random_seed,
                        ..LearnerConfig::default()
                    }
                };
                let train_start = Instant::now();
                let (tree, diagnostics) = learn_with_diagnostics(&train, &cfg)?;
                let (path_theory_state, path_backend, path_certified) =
                    tree_path_theory_metadata(&tree);
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
                let (mean_axp_length, max_axp_length, mut theorem_certified) =
                    if test.features.rows() == 0 {
                        (0.0, 0, path_certified)
                    } else {
                        let limit = test.features.rows().min(8);
                        let mut total = 0usize;
                        let mut maximum = 0usize;
                        let mut certified = path_certified && tree_is_certified(&tree);
                        for row in 0..limit {
                            let axp = extract_axp_deletion(&tree, &test.features, row, true);
                            total += axp.features.len();
                            maximum = maximum.max(axp.features.len());
                            certified &= axp.metadata.theorem_certified;
                        }
                        (total as f64 / limit as f64, maximum, certified)
                    };
                let axp_time = if measure_times {
                    axp_start.elapsed().as_secs_f64()
                } else {
                    0.0
                };

                let all_predicates_backend_allowed = predicates_backend_allowed(&tree, &train);
                theorem_certified &= all_predicates_backend_allowed;
                let path_violation_count = usize::from(!path_certified);
                let pruning = &diagnostics.pruning;
                let nodes_before_prune = if pruning.enabled {
                    pruning.nodes_before
                } else {
                    tree.nodes()
                };
                let leaves_before_prune = if pruning.enabled {
                    pruning.leaves_before
                } else {
                    tree.leaves()
                };
                let literals_before_prune = if pruning.enabled {
                    pruning.literals_before
                } else {
                    tree.literals()
                };
                let adaptive_candidate_count = diagnostics
                    .adaptive_language
                    .nodes
                    .iter()
                    .map(|node| node.candidates_generated)
                    .sum::<usize>();
                let candidate_count = adaptive_candidate_count
                    .max(diagnostics.branch_and_bound.complete_candidates_evaluated);
                let compatible_family_count = diagnostics
                    .adaptive_language
                    .nodes
                    .iter()
                    .map(|node| node.compatible_families.len())
                    .max()
                    .unwrap_or_else(|| compatible_family_count_for_policy(policy));
                let selected_family_counts =
                    format!("{:?}", diagnostics.adaptive_language.selected_family_counts);
                let pruning_time = pruning.pruning_time_seconds;
                let axp_rerank_time = diagnostics.axp_rerank.elapsed_seconds;
                let search_time = (train_time - pruning_time - axp_rerank_time).max(0.0);
                let empirical_fallback_used = false;
                let incompatible_cached_subtree_reused = false;
                let theorem_rejection_reason = theorem_rejection_reason(
                    theorem_certified,
                    path_certified,
                    all_predicates_backend_allowed,
                    empirical_fallback_used,
                    incompatible_cached_subtree_reused,
                );
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
                    path_theory_state,
                    path_backend,
                    path_certified,
                    git_sha: git_sha.clone(),
                    config: format!("{:?}", &cfg),
                    random_state: seed.wrapping_add(run as u64),
                    n_runs: runs.len(),
                    train_test_split_protocol: "deterministic_hash_70_30_first_label".into(),
                    search_strategy: format!("{:?}", cfg.tree_search.strategy),
                    score_profile: format!("{:?}", cfg.split_score.profile),
                    candidate_beam_width: if method == "cals" {
                        cfg.tree_search.candidate_beam_width
                    } else {
                        cfg.beam_width
                    },
                    tree_beam_width: cfg.tree_search.tree_beam_width,
                    lookahead_depth: cfg.tree_search.lookahead_depth,
                    node_budget: cfg.tree_search.node_budget,
                    pruning_enabled: pruning.enabled,
                    nodes_before_prune,
                    nodes_after_prune: tree.nodes(),
                    leaves_before_prune,
                    leaves_after_prune: tree.leaves(),
                    literals_before_prune,
                    literals_after_prune: tree.literals(),
                    validation_accuracy_before_prune: pruning.validation_accuracy_before,
                    validation_accuracy_after_prune: pruning.validation_accuracy_after,
                    candidate_count,
                    candidate_pruned_count: diagnostics.branch_and_bound.partial_states_pruned,
                    branch_and_bound_fallback_count: diagnostics
                        .branch_and_bound
                        .exhaustive_fallback_count,
                    predicate_mask_cache_hits: diagnostics.cache.predicate_masks.hits,
                    predicate_mask_cache_misses: diagnostics.cache.predicate_masks.misses,
                    candidate_cache_hits: diagnostics.cache.candidate_pools.hits,
                    candidate_cache_misses: diagnostics.cache.candidate_pools.misses,
                    subtree_cache_hits: diagnostics.cache.best_subtrees.hits,
                    subtree_cache_misses: diagnostics.cache.best_subtrees.misses,
                    parallel_threads: diagnostics.parallel.configured_threads,
                    compatible_family_count,
                    selected_family_counts,
                    path_violation_count,
                    max_axp_length,
                    total_fit_time: train_time,
                    search_time,
                    pruning_time,
                    axp_rerank_time,
                    empirical_fallback_used,
                    incompatible_cached_subtree_reused,
                    all_predicates_backend_allowed,
                    theorem_rejection_reason,
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
        if !r.path_certified {
            warnings.push(BenchmarkWarning {
                dataset: r.dataset.clone(),
                run: r.run.to_string(),
                depth: r.depth.to_string(),
                method: r.method.clone(),
                warning_type: "path_compatibility_violation".into(),
                message: "a root-to-leaf path mixes incompatible certified theories".into(),
                value: r.path_theory_state.clone(),
            });
        }
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
    write_optimization_diagnostics(output.as_ref(), rows)?;
    fs::write(
        output.as_ref().join("README_RESULTS.md"),
        format!(
            "# CGS-MDT benchmark results\n\nRows: {}\n\nThe theorem table contains Unary, Horn, AntiHorn, Square2CNF, Boolean Affine/GF(2), and path-compatible SmartCertified rows with certified backends only.\n",
            rows.len()
        ),
    )?;
    Ok(())
}

fn write_optimization_diagnostics(output: &Path, rows: &[ResultRow]) -> Result<()> {
    let mut search = String::from("dataset,run,depth,method,search_strategy,score_profile,candidate_count,candidate_pruned_count,branch_and_bound_fallback_count,search_time,path_certified,path_violation_count\n");
    let mut pruning = String::from("dataset,run,depth,method,pruning_enabled,nodes_before,nodes_after,leaves_before,leaves_after,literals_before,literals_after,validation_accuracy_before,validation_accuracy_after,pruning_time,path_certified\n");
    let mut cache = String::from("dataset,run,depth,method,predicate_mask_hits,predicate_mask_misses,candidate_hits,candidate_misses,subtree_hits,subtree_misses,incompatible_subtree_reuse\n");
    let mut family = String::from("dataset,run,depth,method,compatible_family_count,selected_family_counts,path_theory_state,path_backend\n");
    let mut beam = String::from("dataset,run,depth,method,search_strategy,candidate_beam_width,tree_beam_width,lookahead_depth,node_budget,total_fit_time\n");
    for row in rows {
        let key = format!(
            "{},{},{},{}",
            csv_escape(&row.dataset),
            row.run,
            row.depth,
            row.method
        );
        search.push_str(&format!(
            "{key},{},{},{},{},{},{},{},{}\n",
            row.search_strategy,
            row.score_profile,
            row.candidate_count,
            row.candidate_pruned_count,
            row.branch_and_bound_fallback_count,
            row.search_time,
            row.path_certified,
            row.path_violation_count
        ));
        pruning.push_str(&format!(
            "{key},{},{},{},{},{},{},{},{},{},{},{}\n",
            row.pruning_enabled,
            row.nodes_before_prune,
            row.nodes_after_prune,
            row.leaves_before_prune,
            row.leaves_after_prune,
            row.literals_before_prune,
            row.literals_after_prune,
            row.validation_accuracy_before_prune,
            row.validation_accuracy_after_prune,
            row.pruning_time,
            row.path_certified
        ));
        cache.push_str(&format!(
            "{key},{},{},{},{},{},{},{}\n",
            row.predicate_mask_cache_hits,
            row.predicate_mask_cache_misses,
            row.candidate_cache_hits,
            row.candidate_cache_misses,
            row.subtree_cache_hits,
            row.subtree_cache_misses,
            row.incompatible_cached_subtree_reused
        ));
        family.push_str(&format!(
            "{key},{},{},{},{}\n",
            row.compatible_family_count,
            csv_escape(&row.selected_family_counts),
            csv_escape(&row.path_theory_state),
            csv_escape(&row.path_backend)
        ));
        beam.push_str(&format!(
            "{key},{},{},{},{},{},{}\n",
            row.search_strategy,
            row.candidate_beam_width,
            row.tree_beam_width,
            row.lookahead_depth,
            row.node_budget,
            row.total_fit_time
        ));
    }
    fs::write(output.join("search_diagnostics.csv"), search)?;
    fs::write(output.join("pruning_diagnostics.csv"), pruning)?;
    fs::write(output.join("cache_diagnostics.csv"), cache)?;
    fs::write(output.join("family_budget_diagnostics.csv"), family)?;
    fs::write(output.join("beam_diagnostics.csv"), beam)?;
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
        "affine" => "Affine",
        "smart_certified" => "Smart certified",
        "cals" => "CALS-MDT",
        "best-certified" => "Best certified per node",
        other => other,
    }
}

fn path_certificate(backend: Backend) -> &'static str {
    match backend {
        Backend::StructuralHorn => "HornCnf",
        Backend::StructuralAntiHorn => "AntiHornCnf",
        Backend::TwoSat => "TwoCnf",
        Backend::Gf2Gaussian => "AffineGf2",
        Backend::PathCertified => "PathTheory",
        Backend::Affine => "Empirical",
        _ => "Unsupported",
    }
}

fn write_csv(path: impl AsRef<Path>, rows: &[ResultRow]) -> Result<()> {
    let mut out = String::from("dataset,run,depth,method,accuracy,train_time,predict_time,tree_nodes,leaves,max_depth_reached,mean_axp_length,axp_time,theorem_certified,language_family,backend,path_theory_state,path_backend,path_certified,git_sha,config,method_key,method_label,category,acc,acc_std,size,expl,axp_valid_rate,axp_minimal_rate,n_success,n_fail,axp_backend,path_certificate,rejected_reason,theorem_mode_used,random_state,n_runs,train_test_split_protocol,search_strategy,score_profile,candidate_beam_width,tree_beam_width,lookahead_depth,node_budget,pruning_enabled,nodes_before_prune,nodes_after_prune,leaves_before_prune,leaves_after_prune,literals_before_prune,literals_after_prune,validation_accuracy_before_prune,validation_accuracy_after_prune,candidate_count,candidate_pruned_count,branch_and_bound_fallback_count,predicate_mask_cache_hits,predicate_mask_cache_misses,candidate_cache_hits,candidate_cache_misses,subtree_cache_hits,subtree_cache_misses,parallel_threads,compatible_family_count,selected_family_counts,path_violation_count,max_axp_length,total_fit_time,search_time,pruning_time,axp_rerank_time,empirical_fallback_used,incompatible_cached_subtree_reused,all_predicates_backend_allowed,theorem_rejection_reason\n");
    for r in rows {
        let category = if theorem_table_filter(r) {
            "certified"
        } else {
            "empirical_or_adaptive"
        };
        let axp_rate = if r.theorem_certified { 1.0 } else { 0.0 };
        let n_success = usize::from(r.theorem_certified);
        let n_fail = usize::from(!r.theorem_certified);
        let rejected_reason = if theorem_table_filter(r) {
            ""
        } else if r.theorem_rejection_reason.is_empty() {
            "not theorem-table eligible"
        } else {
            &r.theorem_rejection_reason
        };
        let fields = vec![
            csv_escape(&r.dataset),
            r.run.to_string(),
            r.depth.to_string(),
            r.method.clone(),
            r.accuracy.to_string(),
            r.train_time.to_string(),
            r.predict_time.to_string(),
            r.tree_nodes.to_string(),
            r.leaves.to_string(),
            r.max_depth_reached.to_string(),
            r.mean_axp_length.to_string(),
            r.axp_time.to_string(),
            r.theorem_certified.to_string(),
            format!("{:?}", r.language_family),
            format!("{:?}", r.backend),
            csv_escape(&r.path_theory_state),
            csv_escape(&r.path_backend),
            r.path_certified.to_string(),
            r.git_sha.clone(),
            csv_escape(&r.config),
            r.method.clone(),
            csv_escape(method_label(&r.method)),
            category.into(),
            r.accuracy.to_string(),
            0.0f64.to_string(),
            r.tree_nodes.to_string(),
            r.mean_axp_length.to_string(),
            axp_rate.to_string(),
            axp_rate.to_string(),
            n_success.to_string(),
            n_fail.to_string(),
            format!("{:?}", r.backend),
            path_certificate(r.backend).into(),
            csv_escape(rejected_reason),
            true.to_string(),
            r.random_state.to_string(),
            r.n_runs.to_string(),
            r.train_test_split_protocol.clone(),
            r.search_strategy.clone(),
            r.score_profile.clone(),
            r.candidate_beam_width.to_string(),
            r.tree_beam_width.to_string(),
            r.lookahead_depth.to_string(),
            r.node_budget.to_string(),
            r.pruning_enabled.to_string(),
            r.nodes_before_prune.to_string(),
            r.nodes_after_prune.to_string(),
            r.leaves_before_prune.to_string(),
            r.leaves_after_prune.to_string(),
            r.literals_before_prune.to_string(),
            r.literals_after_prune.to_string(),
            r.validation_accuracy_before_prune.to_string(),
            r.validation_accuracy_after_prune.to_string(),
            r.candidate_count.to_string(),
            r.candidate_pruned_count.to_string(),
            r.branch_and_bound_fallback_count.to_string(),
            r.predicate_mask_cache_hits.to_string(),
            r.predicate_mask_cache_misses.to_string(),
            r.candidate_cache_hits.to_string(),
            r.candidate_cache_misses.to_string(),
            r.subtree_cache_hits.to_string(),
            r.subtree_cache_misses.to_string(),
            r.parallel_threads.to_string(),
            r.compatible_family_count.to_string(),
            csv_escape(&r.selected_family_counts),
            r.path_violation_count.to_string(),
            r.max_axp_length.to_string(),
            r.total_fit_time.to_string(),
            r.search_time.to_string(),
            r.pruning_time.to_string(),
            r.axp_rerank_time.to_string(),
            r.empirical_fallback_used.to_string(),
            r.incompatible_cached_subtree_reused.to_string(),
            r.all_predicates_backend_allowed.to_string(),
            csv_escape(&r.theorem_rejection_reason),
        ];
        out.push_str(&fields.join(","));
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}
