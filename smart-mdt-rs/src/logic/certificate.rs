use super::LanguageFamily;

/// Domain on which the relation was actually certified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DomainRegime {
    Boolean,
    OrderedFiniteUi,
    Unsupported,
}

/// Normative result used for the certificate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TheoremSource {
    Theorem3,
    Theorem4,
    Theorem5,
    Theorem6,
    Proposition5,
    Proposition6,
    Theorem7,
    Theorem8,
    UnaryBaseline,
    Proposition1,
}

/// Structural proof attached to a relation or a path-certified tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StructuralCheck {
    UnaryRelation,
    StarNestedHorn,
    StarNestedAntiHorn,
    SingleGf2Equation,
    Square2CnfEmpty,
    Square2CnfComplete,
    Square2CnfFormI,
    Square2CnfFormII,
    Square2CnfFormIII,
    PathCompatibleExactRelations,
    Unsupported,
}

/// Evidence that logical negation stays in the claimed node language.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComplementCheck {
    UnaryNegation,
    StarNestedConstruction,
    Gf2RhsFlip,
    Square2CnfDualForm,
    PerNodeVerified,
    Unsupported,
}

/// Exact CSP super-language check used for opposite-class paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PathCheck {
    HornCnfValidated,
    AntiHornCnfValidated,
    TwoCnfValidated,
    Gf2SystemValidated,
    PerPathTheoryValidated,
    Unsupported,
}

/// Full theorem certificate.  No field is inferred from a method name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TheoremCertificate {
    pub domain_regime: DomainRegime,
    pub language_family: LanguageFamily,
    pub theorem_id: TheoremSource,
    pub structural_check: StructuralCheck,
    pub complement_check: ComplementCheck,
    pub backend: Backend,
    pub assumptions_supported: bool,
    pub path_check: PathCheck,
}

impl TheoremCertificate {
    /// Fail-closed validation of the complete theorem tuple.
    pub fn is_valid(&self) -> bool {
        if !self.assumptions_supported || self.domain_regime == DomainRegime::Unsupported {
            return false;
        }
        match self.domain_regime {
            DomainRegime::Boolean => matches!(
                (
                    self.language_family,
                    self.theorem_id,
                    self.structural_check,
                    self.complement_check,
                    self.backend,
                    self.path_check,
                ),
                (
                    LanguageFamily::Unary,
                    TheoremSource::UnaryBaseline,
                    StructuralCheck::UnaryRelation,
                    ComplementCheck::UnaryNegation,
                    Backend::StructuralHorn,
                    PathCheck::HornCnfValidated,
                ) | (
                    LanguageFamily::Horn,
                    TheoremSource::Theorem3,
                    StructuralCheck::StarNestedHorn,
                    ComplementCheck::StarNestedConstruction,
                    Backend::StructuralHorn,
                    PathCheck::HornCnfValidated,
                ) | (
                    LanguageFamily::AntiHorn,
                    TheoremSource::Theorem4,
                    StructuralCheck::StarNestedAntiHorn,
                    ComplementCheck::StarNestedConstruction,
                    Backend::StructuralAntiHorn,
                    PathCheck::AntiHornCnfValidated,
                ) | (
                    LanguageFamily::Affine,
                    TheoremSource::Theorem5,
                    StructuralCheck::SingleGf2Equation,
                    ComplementCheck::Gf2RhsFlip,
                    Backend::Gf2Gaussian,
                    PathCheck::Gf2SystemValidated,
                ) | (
                    LanguageFamily::Square2Cnf,
                    TheoremSource::Theorem6,
                    StructuralCheck::Square2CnfEmpty
                        | StructuralCheck::Square2CnfComplete
                        | StructuralCheck::Square2CnfFormI
                        | StructuralCheck::Square2CnfFormII
                        | StructuralCheck::Square2CnfFormIII,
                    ComplementCheck::Square2CnfDualForm,
                    Backend::TwoSat,
                    PathCheck::TwoCnfValidated,
                ) | (
                    LanguageFamily::SmartCertified,
                    TheoremSource::Proposition1,
                    StructuralCheck::PathCompatibleExactRelations,
                    ComplementCheck::PerNodeVerified,
                    Backend::PathCertified,
                    PathCheck::PerPathTheoryValidated,
                )
            ),
            // Ordered finite-domain certificates are intentionally unavailable
            // in this Boolean-only repository.  The enum exists so that output
            // metadata cannot silently conflate the two regimes.
            DomainRegime::OrderedFiniteUi | DomainRegime::Unsupported => false,
        }
    }
}

/// Explanation backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    StructuralHorn,
    StructuralAntiHorn,
    TwoSat,
    /// Certified path satisfiability by Gaussian elimination over GF(2).
    Gf2Gaussian,
    /// Meta-backend indicating that every path has its own certified backend.
    PathCertified,
    IntervalDfsFallback,
    PrototypeCaseSplit,
    Affine,
    EmpiricalMixed,
    None,
}

/// Backward-compatible coarse path label used in benchmark CSVs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PathCertificate {
    HornCnf,
    AntiHornCnf,
    TwoCnf,
    AffineGf2,
    PathTheory,
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
    pub theorem_certificate: Option<TheoremCertificate>,
    pub rejected_reason: Option<String>,
}

impl CertificateMetadata {
    /// Legacy constructor retained for API compatibility.  A backend/family
    /// tuple alone is not a structural theorem proof and therefore fails closed.
    pub fn new(
        theorem_mode: bool,
        language_family: LanguageFamily,
        backend: Backend,
        path_certificate: PathCertificate,
    ) -> Self {
        Self {
            theorem_mode,
            theorem_certified: false,
            language_family,
            backend,
            path_certificate,
            theorem_certificate: None,
            rejected_reason: theorem_mode.then(|| {
                "structured theorem certificate required; backend names are insufficient".into()
            }),
        }
    }

    pub fn from_theorem(theorem_mode: bool, certificate: TheoremCertificate) -> Self {
        let path_certificate = match certificate.path_check {
            PathCheck::HornCnfValidated => PathCertificate::HornCnf,
            PathCheck::AntiHornCnfValidated => PathCertificate::AntiHornCnf,
            PathCheck::TwoCnfValidated => PathCertificate::TwoCnf,
            PathCheck::Gf2SystemValidated => PathCertificate::AffineGf2,
            PathCheck::PerPathTheoryValidated => PathCertificate::PathTheory,
            PathCheck::Unsupported => PathCertificate::Unsupported,
        };
        let theorem_certified = theorem_mode && certificate.is_valid();
        let rejected_reason = (theorem_mode && !theorem_certified)
            .then(|| "structured theorem certificate failed validation".into());
        Self {
            theorem_mode,
            theorem_certified,
            language_family: certificate.language_family,
            backend: certificate.backend,
            path_certificate,
            theorem_certificate: Some(certificate),
            rejected_reason,
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
            theorem_certificate: None,
            rejected_reason: Some(reason.into()),
        }
    }
}
