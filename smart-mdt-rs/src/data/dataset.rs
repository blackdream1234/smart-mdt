use super::ColumnMajorMatrix;
use crate::{ClassId, Result, SmartMdtError};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};
/// In-memory supervised dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct Dataset {
    pub features: ColumnMajorMatrix,
    pub labels: Vec<ClassId>,
}

/// Dataset metadata emitted by the Python-equivalent `.dl8` parser.
#[derive(Clone, Debug, PartialEq)]
pub struct DatasetMetadata {
    pub dataset: String,
    pub path: String,
    pub n_samples: usize,
    pub n_columns_original: usize,
    pub n_features_original: usize,
    pub n_features_after_constant_removal: usize,
    pub raw_label_unique_count: usize,
    pub raw_label_counts: String,
    pub binarized_label_counts: String,
    pub positive_rate: f64,
    pub majority_class_rate: f64,
    pub removed_constant_columns_count: usize,
    pub is_binary_features: bool,
    pub skipped: bool,
    pub skip_reason: String,
    pub label_column_used: usize,
    pub label_excluded_from_features: bool,
    pub feature_equal_to_label_count: usize,
    pub feature_equal_to_label_indices: String,
    pub suspicious_majority_rate: bool,
    pub suspicious_feature_label_leakage: bool,
}

/// Result of parsing a `.dl8` file with metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Dl8LoadResult {
    pub dataset: Option<Dataset>,
    pub metadata: DatasetMetadata,
}

impl Dataset {
    /// Creates a dataset.
    pub fn new(features: ColumnMajorMatrix, labels: Vec<ClassId>) -> Result<Self> {
        if features.rows() != labels.len() {
            return Err(SmartMdtError::Dimension("label count".into()));
        }
        Ok(Self { features, labels })
    }

    /// Loads a `.dl8` file using the Python baseline convention:
    /// `label f1 f2 ... fn`, label binarization, and constant-column removal.
    pub fn from_dl8_like(path: impl AsRef<Path>) -> Result<Self> {
        let loaded = load_dl8_with_metadata(path)?;
        loaded.dataset.ok_or_else(|| {
            SmartMdtError::InvalidInput(format!("dataset skipped: {}", loaded.metadata.skip_reason))
        })
    }

    /// Number of classes by max label + 1.
    pub fn class_count(&self) -> usize {
        self.labels
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m as usize + 1)
    }
}

/// Loads a `.dl8` file and returns both the processed dataset and metadata.
pub fn load_dl8_with_metadata(path: impl AsRef<Path>) -> Result<Dl8LoadResult> {
    let path = path.as_ref();
    let f = File::open(path)?;
    let mut raw_rows: Vec<Vec<i32>> = Vec::new();
    let mut width = None;
    for (lineno, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let toks: Vec<_> = t.split_whitespace().collect();
        if toks.len() < 2 {
            return Err(SmartMdtError::InvalidInput(format!(
                "{}:{} has fewer than 2 columns",
                path.display(),
                lineno + 1
            )));
        }
        if let Some(w) = width {
            if toks.len() != w {
                return Err(SmartMdtError::Dimension(format!(
                    "{}:{} inconsistent row width: expected {}, got {}",
                    path.display(),
                    lineno + 1,
                    w,
                    toks.len()
                )));
            }
        } else {
            width = Some(toks.len());
        }
        let mut row = Vec::with_capacity(toks.len());
        for tok in toks {
            row.push(tok.parse::<i32>().map_err(|_| {
                SmartMdtError::InvalidInput(format!(
                    "{}:{} invalid integer token {}",
                    path.display(),
                    lineno + 1,
                    tok
                ))
            })?);
        }
        raw_rows.push(row);
    }
    let dataset_name = dataset_stem(path);
    let n_samples = raw_rows.len();
    let n_columns_original = width.unwrap_or(0);
    let n_features_original = n_columns_original.saturating_sub(1);
    if n_samples == 0 || n_columns_original < 2 {
        let meta = base_metadata(
            &dataset_name,
            path,
            n_samples,
            n_columns_original,
            n_features_original,
            "empty_or_too_few_columns",
        );
        return Ok(Dl8LoadResult {
            dataset: None,
            metadata: meta,
        });
    }

    let raw_labels: Vec<i32> = raw_rows.iter().map(|r| r[0]).collect();
    let raw_label_counts = counts_string(&raw_labels);
    let raw_label_unique_count = unique_sorted(&raw_labels).len();
    let bin_labels_i32 = binarize_labels_python(&raw_labels);
    let binarized_label_counts = counts_string(&bin_labels_i32);
    let positives = bin_labels_i32.iter().filter(|&&y| y == 1).count();
    let positive_rate = positives as f64 / n_samples as f64;
    let majority_class_rate = positive_rate.max(1.0 - positive_rate);

    let feature_rows: Vec<Vec<f64>> = raw_rows
        .iter()
        .map(|r| r[1..].iter().map(|&v| f64::from(v)).collect())
        .collect();
    let keep = non_constant_columns(&feature_rows, 1e-12);
    let removed_constant_columns_count = n_features_original.saturating_sub(keep.len());
    let processed_rows: Vec<Vec<f64>> = feature_rows
        .iter()
        .map(|r| keep.iter().map(|&j| r[j]).collect())
        .collect();
    let is_binary_features = processed_rows
        .iter()
        .flatten()
        .all(|v| *v == 0.0 || *v == 1.0);
    let feature_equal_to_label_indices = feature_equal_to_label(&processed_rows, &bin_labels_i32);
    let feature_equal_to_label_count = feature_equal_to_label_indices.len();

    let mut skipped = false;
    let mut skip_reason = String::new();
    if unique_sorted(&bin_labels_i32).len() < 2 {
        skipped = true;
        skip_reason = "target_one_class_after_binarization".into();
    } else if keep.is_empty() {
        skipped = true;
        skip_reason = "no_non_constant_features".into();
    }

    let metadata = DatasetMetadata {
        dataset: dataset_name,
        path: path.to_string_lossy().to_string(),
        n_samples,
        n_columns_original,
        n_features_original,
        n_features_after_constant_removal: keep.len(),
        raw_label_unique_count,
        raw_label_counts,
        binarized_label_counts,
        positive_rate,
        majority_class_rate,
        removed_constant_columns_count,
        is_binary_features,
        skipped,
        skip_reason: skip_reason.clone(),
        label_column_used: 0,
        label_excluded_from_features: true,
        feature_equal_to_label_count,
        feature_equal_to_label_indices: indices_string(&feature_equal_to_label_indices),
        suspicious_majority_rate: majority_class_rate >= 0.99,
        suspicious_feature_label_leakage: feature_equal_to_label_count > 0,
    };
    let dataset = if skipped {
        None
    } else {
        let labels = bin_labels_i32.iter().map(|&y| y as ClassId).collect();
        Some(Dataset::new(
            ColumnMajorMatrix::from_rows(&processed_rows)?,
            labels,
        )?)
    };
    Ok(Dl8LoadResult { dataset, metadata })
}

/// Python-equivalent `binarize_labels(y)`.
pub fn binarize_labels_python(y: &[i32]) -> Vec<i32> {
    let labels = unique_sorted(y);
    if labels.len() < 2 {
        return vec![0; y.len()];
    }
    if labels.len() > 2 {
        let max_label = y.iter().copied().filter(|v| *v >= 0).max().unwrap_or(0) as usize;
        let mut counts = vec![0usize; max_label + 1];
        for &v in y {
            if v >= 0 {
                counts[v as usize] += 1;
            }
        }
        let majority = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| **c)
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
        return y.iter().map(|&v| i32::from(v == majority)).collect();
    }
    let positive_label = labels[1];
    y.iter().map(|&v| i32::from(v == positive_label)).collect()
}

fn unique_sorted(y: &[i32]) -> Vec<i32> {
    let mut labels = y.to_vec();
    labels.sort_unstable();
    labels.dedup();
    labels
}

fn counts_string(y: &[i32]) -> String {
    let mut m = BTreeMap::new();
    for &v in y {
        *m.entry(v).or_insert(0usize) += 1;
    }
    m.into_iter()
        .map(|(k, v)| format!("{}:{}", k, v))
        .collect::<Vec<_>>()
        .join(";")
}

fn non_constant_columns(rows: &[Vec<f64>], eps: f64) -> Vec<usize> {
    let Some(first) = rows.first() else {
        return Vec::new();
    };
    (0..first.len())
        .filter(|&j| {
            let mean = rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64;
            let var = rows
                .iter()
                .map(|r| {
                    let d = r[j] - mean;
                    d * d
                })
                .sum::<f64>()
                / rows.len() as f64;
            var > eps
        })
        .collect()
}

fn feature_equal_to_label(rows: &[Vec<f64>], y: &[i32]) -> Vec<usize> {
    let Some(first) = rows.first() else {
        return Vec::new();
    };
    (0..first.len())
        .filter(|&j| {
            rows.iter()
                .zip(y)
                .all(|(r, &label)| r[j] == f64::from(label))
        })
        .collect()
}

fn indices_string(indices: &[usize]) -> String {
    indices
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(";")
}

fn dataset_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn base_metadata(
    dataset: &str,
    path: &Path,
    n_samples: usize,
    n_columns_original: usize,
    n_features_original: usize,
    reason: &str,
) -> DatasetMetadata {
    DatasetMetadata {
        dataset: dataset.to_string(),
        path: path.to_string_lossy().to_string(),
        n_samples,
        n_columns_original,
        n_features_original,
        n_features_after_constant_removal: 0,
        raw_label_unique_count: 0,
        raw_label_counts: String::new(),
        binarized_label_counts: String::new(),
        positive_rate: 0.0,
        majority_class_rate: 0.0,
        removed_constant_columns_count: 0,
        is_binary_features: false,
        skipped: true,
        skip_reason: reason.into(),
        label_column_used: 0,
        label_excluded_from_features: true,
        feature_equal_to_label_count: 0,
        feature_equal_to_label_indices: String::new(),
        suspicious_majority_rate: false,
        suspicious_feature_label_leakage: false,
    }
}
