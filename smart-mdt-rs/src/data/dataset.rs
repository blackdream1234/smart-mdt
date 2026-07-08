use super::ColumnMajorMatrix;
use crate::{ClassId, Result, SmartMdtError};
use std::{
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
impl Dataset {
    /// Creates a dataset.
    pub fn new(features: ColumnMajorMatrix, labels: Vec<ClassId>) -> Result<Self> {
        if features.rows() != labels.len() {
            return Err(SmartMdtError::Dimension("label count".into()));
        }
        Ok(Self { features, labels })
    }
    /// Loads a simple whitespace/comma separated file where the last column is the label.
    pub fn from_dl8_like(path: impl AsRef<Path>) -> Result<Self> {
        let f = File::open(path)?;
        let mut rows = Vec::new();
        let mut labels = Vec::new();
        for line in BufReader::new(f).lines() {
            let line = line?;
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            let toks: Vec<_> = t
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .collect();
            if toks.len() < 2 {
                continue;
            }
            let vals: Vec<f64> = toks[..toks.len() - 1]
                .iter()
                .map(|s| s.parse().unwrap_or(0.0))
                .collect();
            let lab = toks[toks.len() - 1].parse().unwrap_or(0);
            rows.push(vals);
            labels.push(lab);
        }
        Self::new(ColumnMajorMatrix::from_rows(&rows)?, labels)
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
