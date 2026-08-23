use super::{
    classify_square_2cnf, classify_star_nested_antihorn, classify_star_nested_horn, Backend,
    BooleanFormula, BooleanLiteral, CertificateMetadata, ComplementCheck, DomainRegime,
    LanguageFamily, Literal, PathCertificate, PathCheck, SingleGf2Equation, Square2CnfForm,
    StructuralCheck, TheoremCertificate, TheoremSource,
};
use crate::{data::ColumnMajorMatrix, FeatureId};
use std::collections::BTreeSet;
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
        self.theorem_certificate().is_some()
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
        if matches!(self, Self::EmpiricalAffine { .. }) {
            return CertificateMetadata::new(
                theorem_mode,
                LanguageFamily::EmpiricalAffine,
                Backend::Affine,
                PathCertificate::Empirical,
            );
        }
        match self.theorem_certificate() {
            Some(certificate) => CertificateMetadata::from_theorem(theorem_mode, certificate),
            None => CertificateMetadata::rejected(
                theorem_mode,
                self.language(),
                "predicate structure or complement is outside the exact theorem class",
            ),
        }
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

    /// Builds a relation-level certificate for the explicitly Boolean
    /// interpretation.  Callers applying a tree to data must additionally
    /// prove that the actual feature domain is Boolean.
    pub fn theorem_certificate(&self) -> Option<TheoremCertificate> {
        let (theorem_id, structural_check, complement_check, path_check) = match self {
            Self::Unary(literal) => {
                boolean_formula_from_clauses([std::slice::from_ref(literal)]);
                (
                    TheoremSource::UnaryBaseline,
                    StructuralCheck::UnaryRelation,
                    ComplementCheck::UnaryNegation,
                    PathCheck::HornCnfValidated,
                )
            }
            Self::HornClause(literals) => {
                let formula = boolean_formula_from_clauses([literals.as_slice()]);
                let witness = classify_star_nested_horn(&formula)?;
                witness.complement()?;
                (
                    TheoremSource::Theorem3,
                    StructuralCheck::StarNestedHorn,
                    ComplementCheck::StarNestedConstruction,
                    PathCheck::HornCnfValidated,
                )
            }
            Self::AntiHornClause(literals) => {
                let formula = boolean_formula_from_clauses([literals.as_slice()]);
                let witness = classify_star_nested_antihorn(&formula)?;
                witness.complement()?;
                (
                    TheoremSource::Theorem4,
                    StructuralCheck::StarNestedAntiHorn,
                    ComplementCheck::StarNestedConstruction,
                    PathCheck::AntiHornCnfValidated,
                )
            }
            Self::Square2Cnf { a, b, c, d } => {
                let formula = boolean_formula_from_clauses([[*a, *b], [*c, *d]]);
                let witness = classify_square_2cnf(&formula)?;
                witness.complement()?;
                let structural_check = match witness.form {
                    Square2CnfForm::Empty => StructuralCheck::Square2CnfEmpty,
                    Square2CnfForm::Complete => StructuralCheck::Square2CnfComplete,
                    Square2CnfForm::FormI => StructuralCheck::Square2CnfFormI,
                    Square2CnfForm::FormII => StructuralCheck::Square2CnfFormII,
                    Square2CnfForm::FormIII => StructuralCheck::Square2CnfFormIII,
                };
                (
                    TheoremSource::Theorem6,
                    structural_check,
                    ComplementCheck::Square2CnfDualForm,
                    PathCheck::TwoCnfValidated,
                )
            }
            Self::Affine { literals, rhs } => {
                let equation = normalized_affine_equation(literals, *rhs);
                if equation.variables.len() > 128 {
                    return None;
                }
                let complement = equation.complement();
                if equation.variables != complement.variables || equation.rhs == complement.rhs {
                    return None;
                }
                (
                    TheoremSource::Theorem5,
                    StructuralCheck::SingleGf2Equation,
                    ComplementCheck::Gf2RhsFlip,
                    PathCheck::Gf2SystemValidated,
                )
            }
            Self::EmpiricalAffine { .. } => return None,
        };
        let certificate = TheoremCertificate {
            domain_regime: DomainRegime::Boolean,
            language_family: self.language(),
            theorem_id,
            structural_check,
            complement_check,
            backend: self.backend(),
            assumptions_supported: true,
            path_check,
        };
        certificate.is_valid().then_some(certificate)
    }
}

fn boolean_formula_from_clauses<I, C>(clauses: I) -> BooleanFormula
where
    I: IntoIterator<Item = C>,
    C: AsRef<[Literal]>,
{
    let mut normalized = Vec::new();
    for clause in clauses {
        let mut literals = Vec::new();
        let mut tautology = false;
        for literal in clause.as_ref() {
            match (literal.eval_value(0.0), literal.eval_value(1.0)) {
                (true, true) => {
                    tautology = true;
                    break;
                }
                (false, false) => {}
                (false, true) => {
                    literals.push(BooleanLiteral::new(literal.atom.feature, true));
                }
                (true, false) => {
                    literals.push(BooleanLiteral::new(literal.atom.feature, false));
                }
            }
        }
        if !tautology {
            normalized.push(literals);
        }
    }
    BooleanFormula::new(normalized)
}

/// Canonicalizes an affine literal list over the Boolean domain.  Repeated
/// variables cancel modulo two and negative/constant literals adjust the RHS.
pub fn normalized_affine_equation(literals: &[Literal], mut rhs: bool) -> SingleGf2Equation {
    let mut variables = BTreeSet::new();
    for literal in literals {
        let toggle_variable = match (literal.eval_value(0.0), literal.eval_value(1.0)) {
            (false, false) => false,
            (true, true) => {
                rhs = !rhs;
                false
            }
            (false, true) => true,
            (true, false) => {
                rhs = !rhs;
                true
            }
        };
        if toggle_variable && !variables.insert(literal.atom.feature) {
            variables.remove(&literal.atom.feature);
        }
    }
    SingleGf2Equation {
        variables: variables.into_iter().collect(),
        rhs,
    }
}
