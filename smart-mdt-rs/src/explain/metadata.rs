use crate::{logic::CertificateMetadata, FeatureId};
use std::time::Duration;
/// Result of a weak AXp check.
#[derive(Clone, Debug)]
pub struct WeakAxpResult {
    pub is_weak_axp: bool,
    pub metadata: CertificateMetadata,
    pub opposite_paths_checked: usize,
}
/// Result of deletion-based AXp extraction.
#[derive(Clone, Debug)]
pub struct AxpResult {
    pub features: Vec<FeatureId>,
    pub weak_checks: usize,
    pub opposite_paths_checked: usize,
    pub metadata: CertificateMetadata,
    pub elapsed_micros: u128,
}
impl AxpResult {
    /// Builds an AXp result.
    pub fn new(
        features: Vec<FeatureId>,
        weak_checks: usize,
        paths: usize,
        metadata: CertificateMetadata,
        elapsed: Duration,
    ) -> Self {
        Self {
            features,
            weak_checks,
            opposite_paths_checked: paths,
            metadata,
            elapsed_micros: elapsed.as_micros(),
        }
    }
}
