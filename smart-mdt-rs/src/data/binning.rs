use super::{BinMatrix, ColumnMajorMatrix};
use crate::Result;
/// Deterministic feature binning metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Binning {
    pub edges: Vec<Vec<f64>>,
    pub exact_unique_limit: usize,
}
/// Binning output.
#[derive(Clone, Debug, PartialEq)]
pub struct BinnedData {
    pub bins: BinMatrix,
    pub binning: Binning,
}
/// Deterministically bins each column using exact unique thresholds or quantiles.
pub fn deterministic_binning(
    x: &ColumnMajorMatrix,
    max_bins: usize,
    exact_unique_limit: usize,
) -> Result<BinnedData> {
    let mut all_edges = Vec::new();
    let mut data = vec![0u16; x.rows() * x.cols()];
    for j in 0..x.cols() {
        let mut vals: Vec<f64> = x
            .column(j as u32)
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .collect();
        vals.sort_by(f64::total_cmp);
        vals.dedup_by(|a, b| a.total_cmp(b).is_eq());
        let edges = if vals.len() <= exact_unique_limit {
            vals.windows(2)
                .map(|w| (w[0] + w[1]) / 2.0)
                .collect::<Vec<_>>()
        } else {
            (1..max_bins)
                .filter_map(|k| vals.get(k * vals.len() / max_bins).copied())
                .collect::<Vec<_>>()
        };
        for i in 0..x.rows() {
            let v = x.get(i, j as u32);
            data[j * x.rows() + i] = edges.partition_point(|e| v >= *e) as u16;
        }
        all_edges.push(edges);
    }
    Ok(BinnedData {
        bins: BinMatrix::new(x.rows(), x.cols(), data)?,
        binning: Binning {
            edges: all_edges,
            exact_unique_limit,
        },
    })
}
        