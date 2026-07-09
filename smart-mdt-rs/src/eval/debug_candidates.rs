//! Candidate-generation diagnostics for Python-parity debugging.
use crate::{
    data::{
        class_counts, load_dl8_with_metadata, predicate_mask, predicate_scope_is_boolean, Dataset,
    },
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    search::affine::AffineConfig,
    search::antihorn::generate_antihorn,
    search::horn::generate_horn,
    search::scoring::{final_score, gini, information_gain, CandidateScore, ScoreWeights},
    search::square2cnf::generate_square2cnf,
    FeatureId, Result, SmartMdtError,
};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Configuration for candidate debugging.
#[derive(Clone, Debug)]
pub struct DebugCandidateConfig {
    pub data_dir: PathBuf,
    pub dataset: String,
    pub method: String,
    pub depth: usize,
    pub node_path: String,
    pub top_k: usize,
    pub output: PathBuf,
    pub seed: u64,
    pub max_candidates_per_node: usize,
    pub beam_width: usize,
}

#[derive(Clone, Debug)]
struct CandidateDiagnostic {
    predicate: Predicate,
    score: CandidateScore,
    true_count: usize,
    false_count: usize,
    true_counts: Vec<usize>,
    false_counts: Vec<usize>,
    impurity_parent: f64,
    impurity_true: f64,
    impurity_false: f64,
    balance: f64,
    boolean_scope: bool,
    theorem_certified: bool,
    rejected: bool,
    rejected_reason: String,
}

/// Runs root candidate diagnostics and writes `debug_candidates.csv` plus masks for top 5.
pub fn run_debug_candidates(cfg: &DebugCandidateConfig) -> Result<Vec<String>> {
    fs::create_dir_all(&cfg.output)?;
    let path = find_dataset_path(&cfg.data_dir, &cfg.dataset)?;
    let loaded = load_dl8_with_metadata(&path)?;
    let ds = loaded.dataset.ok_or_else(|| {
        SmartMdtError::InvalidInput(format!("dataset skipped: {}", loaded.metadata.skip_reason))
    })?;
    let mut candidates = generate_diagnostics(
        &ds,
        &cfg.method,
        cfg.beam_width,
        cfg.max_candidates_per_node,
    );
    candidates.sort_by(|a, b| b.score.final_score.total_cmp(&a.score.final_score));
    candidates.truncate(cfg.top_k);
    write_candidates_csv(cfg, &ds, &candidates)?;
    write_masks_csv(cfg, &ds, &candidates)?;
    Ok(candidates
        .iter()
        .map(|c| predicate_debug(&c.predicate))
        .collect())
}

fn find_dataset_path(root: &Path, wanted: &str) -> Result<PathBuf> {
    fn visit(dir: &Path, wanted: &str, out: &mut Option<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit(&path, wanted, out)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("dl8")
                && path.file_stem().and_then(|s| s.to_str()) == Some(wanted)
            {
                *out = Some(path);
                return Ok(());
            }
        }
        Ok(())
    }
    let mut out = None;
    visit(root, wanted, &mut out)?;
    out.ok_or_else(|| SmartMdtError::InvalidInput(format!("dataset {wanted} not found")))
}

fn generate_diagnostics(
    ds: &Dataset,
    method: &str,
    beam_width: usize,
    cap: usize,
) -> Vec<CandidateDiagnostic> {
    let literals = ranked_literals(ds);
    let selected: Vec<_> = literals.into_iter().take(beam_width.max(2)).collect();
    let mut predicates = Vec::new();
    match method {
        "unary" => predicates.extend(selected.iter().map(|l| Predicate::Unary(*l))),
        "horn" => {
            return generate_horn(ds, 1, beam_width)
                .into_iter()
                .take(cap)
                .map(|c| score_predicate(ds, c.predicate))
                .collect();
        }
        "antihorn" => {
            return generate_antihorn(ds, 1, beam_width)
                .into_iter()
                .take(cap)
                .map(|c| score_predicate(ds, c.predicate))
                .collect();
        }
        "square2cnf" => {
            return generate_square2cnf(ds, 1, beam_width)
                .into_iter()
                .take(cap)
                .map(|c| score_predicate(ds, c.predicate))
                .collect();
        }
        "affine" => {
            return affine_debug_predicates(ds, beam_width, AffineConfig::default())
                .into_iter()
                .take(cap)
                .map(|p| score_predicate(ds, p))
                .collect();
        }
        _ => {}
    }
    predicates
        .into_iter()
        .take(cap)
        .map(|p| score_predicate(ds, p))
        .collect()
}

/// Builds affine debug predicates over all features (including non-Boolean ones,
/// so that Boolean-scope rejections are visible in the diagnostics), ranking
/// features by unary gain and enumerating XOR combinations up to `max_arity`.
fn affine_debug_predicates(ds: &Dataset, beam: usize, cfg: AffineConfig) -> Vec<Predicate> {
    let features = ranked_features_for_affine(ds);
    let pool: Vec<FeatureId> = features
        .into_iter()
        .take(beam.max(cfg.max_arity).max(2))
        .collect();
    let max_arity = cfg.max_arity.clamp(2, 4);
    let mut out = Vec::new();
    for k in 2..=max_arity {
        for combo in affine_combinations(pool.len(), k) {
            let literals: Vec<Literal> = combo.iter().map(|&i| affine_literal(pool[i])).collect();
            for rhs in [false, true] {
                out.push(Predicate::Affine {
                    literals: literals.clone(),
                    rhs,
                });
            }
        }
    }
    out
}

fn affine_literal(feature: FeatureId) -> Literal {
    Literal {
        atom: ThresholdAtom {
            feature,
            threshold_id: 0,
            threshold: 0.5,
            op: ThresholdOp::GreaterEqual,
        },
        positive: true,
    }
}

fn ranked_features_for_affine(ds: &Dataset) -> Vec<FeatureId> {
    let mut features: Vec<FeatureId> = (0..ds.features.cols() as FeatureId).collect();
    features.sort_by(|&a, &b| {
        let ga = literal_gain(ds, &affine_literal(b));
        let gb = literal_gain(ds, &affine_literal(a));
        ga.total_cmp(&gb).then(a.cmp(&b))
    });
    features
}

fn affine_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        let mut i = k - 1;
        while idx[i] == i + n - k {
            if i == 0 {
                return out;
            }
            i -= 1;
        }
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

fn ranked_literals(ds: &Dataset) -> Vec<Literal> {
    let mut lits = Vec::new();
    for f in 0..ds.features.cols() {
        let mut vals = ds.features.column(f as u32).to_vec();
        vals.sort_by(f64::total_cmp);
        vals.dedup();
        for w in vals.windows(2) {
            let t = (w[0] + w[1]) / 2.0;
            let atom = ThresholdAtom {
                feature: f as u32,
                threshold_id: 0,
                threshold: t,
                op: ThresholdOp::GreaterEqual,
            };
            lits.push(Literal {
                atom,
                positive: true,
            });
            lits.push(Literal {
                atom,
                positive: false,
            });
        }
    }
    lits.sort_by(|a, b| literal_gain(ds, b).total_cmp(&literal_gain(ds, a)));
    lits
}

fn literal_gain(ds: &Dataset, lit: &Literal) -> f64 {
    score_predicate(ds, Predicate::Unary(*lit))
        .score
        .predictive_gain
}

fn score_predicate(ds: &Dataset, predicate: Predicate) -> CandidateDiagnostic {
    let classes = ds.class_count().max(2);
    let mut parent = vec![0usize; classes];
    for &y in &ds.labels {
        parent[y as usize] += 1;
    }
    let mask = predicate_mask(&ds.features, &predicate);
    let true_count = mask.count_ones();
    let false_count = ds.labels.len().saturating_sub(true_count);
    let true_counts = class_counts(&ds.labels, &mask, classes);
    let false_counts: Vec<_> = parent
        .iter()
        .zip(&true_counts)
        .map(|(a, b)| a - b)
        .collect();
    let gain = information_gain(&parent, &true_counts, &false_counts);
    // The Boolean-domain guard: affine may only be theorem-certified when every
    // feature in its scope is Boolean over the loaded domain.
    let is_affine = matches!(predicate, Predicate::Affine { .. });
    let boolean_scope = predicate_scope_is_boolean(&ds.features, &predicate);
    let base_cert = predicate.certificate(true).theorem_certified;
    let theorem_certified = base_cert && (!is_affine || boolean_scope);
    let degenerate = true_count == 0 || false_count == 0;
    let guard_rejected = is_affine && !boolean_scope;
    let rejected = degenerate || guard_rejected;
    let rejected_reason = if true_count == 0 {
        "empty_true_child"
    } else if false_count == 0 {
        "empty_false_child"
    } else if guard_rejected {
        "non_boolean_scope"
    } else {
        ""
    }
    .to_string();
    let score = final_score(
        gain,
        predicate.arity() as f64,
        0.0,
        0.0,
        theorem_certified,
        ScoreWeights::default(),
    );
    CandidateDiagnostic {
        predicate,
        score,
        true_count,
        false_count,
        true_counts,
        false_counts,
        impurity_parent: gini(&parent),
        impurity_true: gini(&class_counts(&ds.labels, &mask, classes)),
        impurity_false: gini(
            &parent
                .iter()
                .zip(class_counts(&ds.labels, &mask, classes))
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        ),
        balance: true_count.min(false_count) as f64 / ds.labels.len().max(1) as f64,
        boolean_scope,
        theorem_certified,
        rejected,
        rejected_reason,
    }
}

fn write_candidates_csv(
    cfg: &DebugCandidateConfig,
    ds: &Dataset,
    candidates: &[CandidateDiagnostic],
) -> Result<()> {
    let mut out = String::from("dataset,method,depth,node_path,n_node_samples,node_class_counts,candidate_rank,candidate_id,predicate_debug,language_family,backend,theorem_certified,true_count,false_count,true_class_counts,false_class_counts,impurity_parent,impurity_true,impurity_false,impurity_gain,balance,complexity,raw_score,certificate_bonus,final_score,rejected,rejected_reason,arity,rhs,boolean_scope\n");
    let node_counts = counts_string(&ds.labels.iter().map(|&x| x as usize).collect::<Vec<_>>());
    for (rank, c) in candidates.iter().enumerate() {
        let candidate_id = format!("cand_{}", rank + 1);
        out.push_str(&format!("{},{},{},{},{},{},{},{},{},{:?},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            csv(&cfg.dataset), cfg.method, cfg.depth, csv(&cfg.node_path), ds.labels.len(), csv(&node_counts), rank + 1,
            candidate_id, csv(&predicate_debug(&c.predicate)), c.predicate.language(), c.predicate.backend(), c.theorem_certified,
            c.true_count, c.false_count, csv(&usize_counts(&c.true_counts)), csv(&usize_counts(&c.false_counts)), c.impurity_parent,
            c.impurity_true, c.impurity_false, c.score.predictive_gain, c.balance, c.predicate.arity(), c.score.predictive_gain,
            c.score.certificate_bonus, c.score.final_score, c.rejected, csv(&c.rejected_reason),
            c.predicate.arity(), affine_rhs_str(&c.predicate), c.boolean_scope));
    }
    fs::write(cfg.output.join("debug_candidates.csv"), out)?;
    Ok(())
}

fn write_masks_csv(
    cfg: &DebugCandidateConfig,
    ds: &Dataset,
    candidates: &[CandidateDiagnostic],
) -> Result<()> {
    let mut out = String::from("dataset,method,candidate_rank,sample_index,y,predicate_value\n");
    for (rank, c) in candidates.iter().take(5).enumerate() {
        let mask = predicate_mask(&ds.features, &c.predicate);
        for (i, y) in ds.labels.iter().enumerate() {
            out.push_str(&format!(
                "{},{},{},{},{},{}\n",
                csv(&cfg.dataset),
                cfg.method,
                rank + 1,
                i,
                y,
                mask.get(i)
            ));
        }
    }
    fs::write(cfg.output.join("debug_candidate_masks.csv"), out)?;
    Ok(())
}

fn predicate_debug(p: &Predicate) -> String {
    format!("{:?}", p).replace(',', ";")
}
fn affine_rhs_str(p: &Predicate) -> String {
    match p {
        Predicate::Affine { rhs, .. } => rhs.to_string(),
        _ => String::new(),
    }
}
fn csv(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "'"))
}
fn usize_counts(xs: &[usize]) -> String {
    xs.iter()
        .enumerate()
        .map(|(i, c)| format!("{}:{}", i, c))
        .collect::<Vec<_>>()
        .join(";")
}
fn counts_string(xs: &[usize]) -> String {
    let mut counts = std::collections::BTreeMap::new();
    for &x in xs {
        *counts.entry(x).or_insert(0usize) += 1;
    }
    counts
        .into_iter()
        .map(|(k, v)| format!("{}:{}", k, v))
        .collect::<Vec<_>>()
        .join(";")
}
