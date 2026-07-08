use crate::{FeatureId, Result, SmartMdtError};

/// Dense numeric matrix in column-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnMajorMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl ColumnMajorMatrix {
    /// Builds a matrix from row-major rows.
    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Self> {
        let r = rows.len();
        let c = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|x| x.len() != c) {
            return Err(SmartMdtError::Dimension("ragged rows".into()));
        }
        let mut data = vec![0.0; r * c];
        for (i, row) in rows.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                data[j * r + i] = *v;
            }
        }
        Ok(Self {
            rows: r,
            cols: c,
            data,
        })
    }
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Returns value at row and feature.
    pub fn get(&self, row: usize, feature: FeatureId) -> f64 {
        self.data[feature as usize * self.rows + row]
    }
    /// Returns a column slice.
    pub fn column(&self, feature: FeatureId) -> &[f64] {
        let s = feature as usize * self.rows;
        &self.data[s..s + self.rows]
    }
}

/// Quantized bin matrix in column-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct BinMatrix {
    rows: usize,
    cols: usize,
    data: Vec<u16>,
}
impl BinMatrix {
    /// Creates a bin matrix.
    pub fn new(rows: usize, cols: usize, data: Vec<u16>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(SmartMdtError::Dimension("bin data length".into()));
        }
        Ok(Self { rows, cols, data })
    }
    /// Row count.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Column count.
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Returns bin value.
    pub fn get(&self, row: usize, feature: FeatureId) -> u16 {
        self.data[feature as usize * self.rows + row]
    }
}

/// Compact bitset backed by u64 blocks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BitSet {
    len: usize,
    blocks: Vec<u64>,
}
impl BitSet {
    /// Creates an empty bitset.
    pub fn zeros(len: usize) -> Self {
        Self {
            len,
            blocks: vec![0; len.div_ceil(64)],
        }
    }
    /// Creates a full bitset.
    pub fn ones(len: usize) -> Self {
        let mut b = Self {
            len,
            blocks: vec![!0; len.div_ceil(64)],
        };
        b.clear_tail();
        b
    }
    /// Number of represented bits.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether no bits are represented.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Sets a bit.
    pub fn set(&mut self, idx: usize, val: bool) {
        if val {
            self.blocks[idx / 64] |= 1u64 << (idx % 64);
        } else {
            self.blocks[idx / 64] &= !(1u64 << (idx % 64));
        }
    }
    /// Gets a bit.
    pub fn get(&self, idx: usize) -> bool {
        ((self.blocks[idx / 64] >> (idx % 64)) & 1) == 1
    }
    /// Population count.
    pub fn count_ones(&self) -> usize {
        self.blocks.iter().map(|b| b.count_ones() as usize).sum()
    }
    /// Bitwise and.
    pub fn and(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a & b)
    }
    /// Bitwise or.
    pub fn or(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a | b)
    }
    /// Bitwise xor.
    pub fn xor(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a ^ b)
    }
    /// Bitwise complement within length.
    pub fn not(&self) -> Self {
        let mut x = Self {
            len: self.len,
            blocks: self.blocks.iter().map(|b| !b).collect(),
        };
        x.clear_tail();
        x
    }
    fn zip<F: Fn(u64, u64) -> u64>(&self, other: &Self, f: F) -> Self {
        assert_eq!(self.len, other.len);
        Self {
            len: self.len,
            blocks: self
                .blocks
                .iter()
                .zip(&other.blocks)
                .map(|(a, b)| f(*a, *b))
                .collect(),
        }
    }
    fn clear_tail(&mut self) {
        if !self.len.is_multiple_of(64) {
            let keep = (1u64 << (self.len % 64)) - 1;
            if let Some(last) = self.blocks.last_mut() {
                *last &= keep;
            }
        }
    }
}
