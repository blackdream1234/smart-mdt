use super::{Backend, CertificateMetadata, LanguageFamily, Literal, PathCertificate};
use crate::data::ColumnMajorMatrix;
/// Split predicate with certificate-first metadata.
#[derive(Clone, Debug, PartialEq)]
pub enum Predicate {
    Unary(Literal),
    HornClause(Vec<Literal>),
    AntiHornClause(Vec<Literal>),
    Square2Cnf {
        a: Literal,
        b: Literal,
        c: Literal,
        d: Literal,
    },
    EmpiricalAffine {
        literals: Vec<Literal>,
        parity: bool,
    },
}
impl Predicate {
    /// Evaluates predicate on a row.
    pub fn eval(&self, x: &ColumnMajorMatrix, row: usize) -> bool {
        match self {
            Self::Unary(l) => l.eval_value(x.get(row, l.atom.feature)),
            Self::HornClause(ls) | Self::AntiHornClause(ls) => {
                ls.iter().any(|l| l.eval_value(x.get(row, l.atom.feature)))
            }
            Self::Square2Cnf { a, b, c, d } => {
                (a.eval_value(x.get(row, a.atom.feature))
                    || b.eval_value(x.get(row, b.atom.feature)))
                    && (c.eval_value(x.get(row, c.atom.feature))
                        || d.eval_value(x.get(row, d.atom.feature)))
            }
            Self::EmpiricalAffine { literals, parity } => {
                literals.iter().fold(false, |acc, l| {
                    acc ^ l.eval_value(x.get(row, l.atom.feature))
                }) == *parity
            }
        }
    }
    /// Language family.
    pub fn language(&self) -> LanguageFamily {
        match self {
            Self::Unary(_) => LanguageFamily::Unary,
            Self::HornClause(_) => LanguageFamily::Horn,
            Self::AntiHornClause(_) => LanguageFamily::AntiHorn,
            Self::Square2Cnf { .. } => LanguageFamily::Square2Cnf,
            Self::EmpiricalAffine { .. } => LanguageFamily::EmpiricalAffine,
        }
    }
    /// Supported backend.
    pub fn backend(&self) -> Backend {
        match self {
            Self::Unary(_) | Self::HornClause(_) => Backend::StructuralHorn,
            Self::AntiHornClause(_) => Backend::StructuralAntiHorn,
            Self::Square2Cnf { .. } => Backend::TwoSat,
            Self::EmpiricalAffine { .. } => Backend::Affine,
        }
    }
    /// Certificate metadata.
    pub fn certificate(&self, theorem_mode: bool) -> CertificateMetadata {
        let pc = match self.backend() {
            Backend::StructuralHorn => PathCertificate::HornCnf,
            Backend::StructuralAntiHorn => PathCertificate::AntiHornCnf,
            Backend::TwoSat => PathCertificate::TwoCnf,
            Backend::Affine => PathCertificate::Empirical,
            Backend::Gf2Gaussian => PathCertificate::Gf2System,
            _ => PathCertificate::Unsupported,
        };
        CertificateMetadata::new(theorem_mode, self.language(), self.backend(), pc)
    }
    /// Predicate complexity as literal count.
    pub fn arity(&self) -> usize {
        match self {
            Self::Unary(_) => 1,
            Self::HornClause(v) | Self::AntiHornClause(v) => v.len(),
            Self::Square2Cnf { .. } => 4,
            Self::EmpiricalAffine { literals, .. } => literals.len(),
        }
    }
}
