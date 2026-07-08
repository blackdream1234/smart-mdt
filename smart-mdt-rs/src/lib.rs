//! Certificate-Guided Smart MDT (CGS-MDT): a research crate for certificate-first
//! multivariate decision trees.

pub mod data;
pub mod eval;
pub mod explain;
pub mod ffi;
pub mod logic;
pub mod sat;
pub mod search;
pub mod tree;
/// Feature identifier.
pub type FeatureId = u32;
/// Threshold identifier.
pub type ThresholdId = u32;
/// Class identifier.
pub type ClassId = u32;
/// Sample identifier.
pub type SampleId = u32;
/// Crate result type.
pub type Result<T> = std::result::Result<T, SmartMdtError>;
/// Recoverable errors for CGS-MDT.
#[derive(Debug)]
pub enum SmartMdtError {
    /// Input dimensions are inconsistent.
    Dimension(String),
    /// The requested operation is not supported in theorem mode.
    TheoremRejected(String),
    /// IO error.
    Io(std::io::Error),
    /// CSV-like write error.
    Csv(String),
    /// JSON-like error.
    Json(String),
    /// Invalid CLI or data input.
    InvalidInput(String),
}
impl std::fmt::Display for SmartMdtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for SmartMdtError {}
impl From<std::io::Error> for SmartMdtError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
