use super::{accuracy, theorem_table_filter, ResultRow};
use crate::{
    data::{ColumnMajorMatrix, Dataset},
    tree::{learn, predict_all, LanguagePolicy, LearnerConfig},
    Result,
};
use std::{fs, path::Path};
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
    run_methods(&ds, output)
}
/// Runs certified methods and writes benchmark output tables.
pub fn run_methods(ds: &Dataset, output: impl AsRef<Path>) -> Result<Vec<ResultRow>> {
    fs::create_dir_all(&output)?;
    let methods = [
        ("unary", LanguagePolicy::UnaryOnly),
        ("horn", LanguagePolicy::HornOnly),
        ("antihorn", LanguagePolicy::AntiHornOnly),
        ("square2cnf", LanguagePolicy::Square2CnfOnly),
        ("best-certified", LanguagePolicy::BestCertifiedPerNode),
    ];
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".into())
        .trim()
        .to_string();
    let mut rows = Vec::new();
    for (name, pol) in methods {
        let cfg = LearnerConfig {
            language_policy: pol,
            ..LearnerConfig::default()
        };
        let tree = learn(ds, &cfg)?;
        let pred = predict_all(&tree, &ds.features);
        let fam = match pol {
            LanguagePolicy::AntiHornOnly => crate::logic::LanguageFamily::AntiHorn,
            LanguagePolicy::Square2CnfOnly => crate::logic::LanguageFamily::Square2Cnf,
            LanguagePolicy::HornOnly => crate::logic::LanguageFamily::Horn,
            _ => crate::logic::LanguageFamily::Unary,
        };
        let backend = match fam {
            crate::logic::LanguageFamily::AntiHorn => crate::logic::Backend::StructuralAntiHorn,
            crate::logic::LanguageFamily::Square2Cnf => crate::logic::Backend::TwoSat,
            _ => crate::logic::Backend::StructuralHorn,
        };
        rows.push(ResultRow {
            method: name.into(),
            accuracy: accuracy(&ds.labels, &pred),
            tree_nodes: tree.nodes(),
            leaves: tree.leaves(),
            max_depth_reached: tree.depth(),
            theorem_certified: true,
            language_family: fam,
            backend,
            git_sha: git_sha.clone(),
            config: format!("{:?}", &cfg),
        });
    }
    write_csv(output.as_ref().join("full_results.csv"), &rows)?;
    write_csv(output.as_ref().join("summary_by_method.csv"), &rows)?;
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
    write_csv(output.as_ref().join("axp_metadata.csv"), &rows)?;
    write_csv(output.as_ref().join("tuning_diagnostics.csv"), &emp)?;
    fs::write(output.as_ref().join("README_RESULTS.md"),"# CGS-MDT benchmark results\n\nThe theorem table is filtered to Unary, Horn, AntiHorn and Square2CNF with certified backends only.\n")?;
    Ok(rows)
}
fn write_csv(path: impl AsRef<Path>, rows: &[ResultRow]) -> Result<()> {
    let mut out = String::from("method,accuracy,tree_nodes,leaves,max_depth_reached,theorem_certified,language_family,backend,git_sha,config\n");
    for r in rows {
        out.push_str(&format!(
            "{},{},{},{},{},{},{:?},{:?},{},\"{}\"\n",
            r.method,
            r.accuracy,
            r.tree_nodes,
            r.leaves,
            r.max_depth_reached,
            r.theorem_certified,
            r.language_family,
            r.backend,
            r.git_sha,
            r.config.replace('\"', "'")
        ));
    }
    std::fs::write(path, out)?;
    Ok(())
}
