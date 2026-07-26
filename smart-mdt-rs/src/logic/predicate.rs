use super::{Backend, CertificateMetadata, LanguageFamily, Literal, PathCertificate};
use crate::{data::ColumnMajorMatrix, FeatureId};
use std::collections::BTreeMap;
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
    /// Certified Boolean affine predicate: a single GF(2) equation
    /// `x_i1 ⊕ x_i2 ⊕ ... ⊕ x_ik = rhs` over Boolean literals in canonical feature order.
    Affine {
        literals: Vec<Literal>,
        rhs: bool,
    },
    EmpiricalAffine {
        literals: Vec<Literal>,
        parity: bool,
    },
}
impl Predicate {
    /// Whether the variant's contents satisfy the syntactic assumptions of
    /// its advertised certified backend.
    pub(crate) fn certificate_shape_is_valid(&self) -> bool {
        match self {
            Self::Unary(_) | Self::Square2Cnf { .. } => true,
            Self::HornClause(literals) => normalized_boolean_clause_polarities(literals)
                .is_none_or(|polarities| {
                    polarities.into_iter().filter(|positive| *positive).count() <= 1
                }),
            Self::AntiHornClause(literals) => normalized_boolean_clause_polarities(literals)
                .is_none_or(|polarities| {
                    polarities.into_iter().filter(|positive| !*positive).count() <= 1
                }),
            Self::Affine { literals, .. } => {
                literals.len() <= 128
                    && literals.iter().all(|literal| {
                        literal.positive
                            && literal.atom.op == super::ThresholdOp::GreaterEqual
                            && literal.atom.threshold == 0.5
                    })
                    && literals
                        .windows(2)
                        .all(|pair| pair[0].atom.feature < pair[1].atom.feature)
            }
            Self::EmpiricalAffine { .. } => false,
        }
    }

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
            Self::Affine { literals, rhs } => {
                literals.iter().fold(false, |acc, l| {
                    acc ^ l.eval_value(x.get(row, l.atom.feature))
                }) == *rhs
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
            Self::Affine { .. } => LanguageFamily::Affine,
            Self::EmpiricalAffine { .. } => LanguageFamily::EmpiricalAffine,
        }
    }
    /// Supported backend.
    pub fn backend(&self) -> Backend {
        match self {
            Self::Unary(_) | Self::HornClause(_) => Backend::StructuralHorn,
            Self::AntiHornClause(_) => Backend::StructuralAntiHorn,
            Self::Square2Cnf { .. } => Backend::TwoSat,
            Self::Affine { .. } => Backend::Gf2Gaussian,
            Self::EmpiricalAffine { .. } => Backend::Affine,
        }
    }
    /// Certificate metadata.
    pub fn certificate(&self, theorem_mode: bool) -> CertificateMetadata {
        if !self.certificate_shape_is_valid() && !matches!(self, Self::EmpiricalAffine { .. }) {
            return CertificateMetadata::rejected(
                theorem_mode,
                self.language(),
                "predicate does not satisfy its advertised certificate shape",
            );
        }
        let pc = match self.backend() {
            Backend::StructuralHorn => PathCertificate::HornCnf,
            Backend::StructuralAntiHorn => PathCertificate::AntiHornCnf,
            Backend::TwoSat => PathCertificate::TwoCnf,
            Backend::Gf2Gaussian => PathCertificate::AffineGf2,
            Backend::Affine => PathCertificate::Empirical,
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
            Self::Affine { literals, .. } | Self::EmpiricalAffine { literals, .. } => {
                literals.len()
            }
        }
    }
    /// Distinct feature indices in the predicate scope, in canonical sorted order.
    pub fn scope_features(&self) -> Vec<FeatureId> {
        let mut fs: Vec<FeatureId> = match self {
            Self::Unary(l) => vec![l.atom.feature],
            Self::HornClause(v) | Self::AntiHornClause(v) => {
                v.iter().map(|l| l.atom.feature).collect()
            }
            Self::Square2Cnf { a, b, c, d } => {
                vec![
                    a.atom.feature,
                    b.atom.feature,
                    c.atom.feature,
                    d.atom.feature,
                ]
            }
            Self::Affine { literals, .. } | Self::EmpiricalAffine { literals, .. } => {
                literals.iter().map(|l| l.atom.feature).collect()
            }
        };
        fs.sort_unstable();
        fs.dedup();
        fs
    }
    /// Complement of an affine predicate is the same equation with the right-hand
    /// side flipped; the literal scope and coefficients are unchanged. Returns
    /// `None` for non-affine predicates, whose complement is expressed as CNF.
    pub fn affine_complement(&self) -> Option<Predicate> {
        match self {
            Self::Affine { literals, rhs } => Some(Self::Affine {
                literals: literals.clone(),
                rhs: !rhs,
            }),
            _ => None,
        }
    }
}

/// Effective propositional polarities after evaluating threshold literals on
/// the certified Boolean domain. `None` denotes a tautological clause.
fn normalized_boolean_clause_polarities(literals: &[Literal]) -> Option<Vec<bool>> {
    let mut normalized = BTreeMap::<FeatureId, bool>::new();
    for literal in literals {
        let at_zero = literal.eval_value(0.0);
        let at_one = literal.eval_value(1.0);
        match (at_zero, at_one) {
            (true, true) => return None,
            (false, false) => {}
            (false, true) | (true, false) => {
                let positive = at_one;
                if normalized
                    .get(&literal.atom.feature)
                    .is_some_and(|existing| *existing != positive)
                {
                    return None;
                }
                normalized.insert(literal.atom.feature, positive);
            }
        }
    }
    Some(normalized.into_values().collect())
}
