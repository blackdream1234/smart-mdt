use super::LanguageFamily;
/// Explanation backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    StructuralHorn,
    StructuralAntiHorn,
    TwoSat,
    /// Certified path satisfiability by Gaussian elimination over GF(2).
    Gf2Gaussian,
    IntervalDfsFallback,
    PrototypeCaseSplit,
    Affine,
    EmpiricalMixed,
    None,
}
/// Path certificate type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PathCertificate {
    HornCnf,
    AntiHornCnf,
    TwoCnf,
    /// GF(2) linear system solved by Gaussian elimination.
    AffineGf2,
    Empirical,
    Unsupported,
}
/// Certificate metadata emitted by explanation checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertificateMetadata {
    pub theorem_mode: bool,
    pub theorem_certified: bool,
    pub language_family: LanguageFamily,
    pub backend: Backend,
    pub path_certificate: PathCertificate,
    pub rejected_reason: Option<String>,
}
impl CertificateMetadata {
    /// Builds metadata and computes certified flag from backend/family.
    pub fn new(
        theorem_mode: bool,
        language_family: LanguageFamily,
        backend: Backend,
        path_certificate: PathCertificate,
    ) -> Self {
        let theorem_certified = language_family.theorem_table_allowed()
            && matches!(
                backend,
                Backend::StructuralHorn
                    | Backend::StructuralAntiHorn
                    | Backend::TwoSat
                    | Backend::Gf2Gaussian
            );
        Self {
            theorem_mode,
            theorem_certified,
            language_family,
            backend,
            path_certificate,
            rejected_reason: None,
        }
    }
    /// Rejection metadata.
    pub fn rejected(
        theorem_mode: bool,
        language_family: LanguageFamily,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            theorem_mode,
            theorem_certified: false,
            language_family,
            backend: Backend::None,
            path_certificate: PathCertificate::Unsupported,
            rejected_reason: Some(reason.into()),
        }
    }
}
