//! Exact Boolean relation classes from Carbonnel--Cooper et al. (2025).
//!
//! The types in this module are deliberately independent of solver names.  A
//! relation receives a theorem label only after its normalized structure and
//! its theorem-preserving complement have both been checked.

use crate::FeatureId;
use std::collections::{BTreeMap, BTreeSet};

/// A propositional literal in canonical feature order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BooleanLiteral {
    pub feature: FeatureId,
    pub positive: bool,
}

impl BooleanLiteral {
    pub const fn new(feature: FeatureId, positive: bool) -> Self {
        Self { feature, positive }
    }

    pub const fn negated(self) -> Self {
        Self {
            feature: self.feature,
            positive: !self.positive,
        }
    }

    pub fn eval(self, assignment: &[bool]) -> bool {
        assignment
            .get(self.feature as usize)
            .is_some_and(|value| *value == self.positive)
    }
}

/// Canonical CNF over Boolean variables.
///
/// Literals and clauses are sorted, duplicate literals/clauses are removed,
/// tautological clauses are discarded, and subsumed clauses are removed.  An
/// empty clause denotes false; no clauses denotes true.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BooleanFormula {
    pub clauses: Vec<Vec<BooleanLiteral>>,
}

impl BooleanFormula {
    pub fn new<I, C>(clauses: I) -> Self
    where
        I: IntoIterator<Item = C>,
        C: IntoIterator<Item = BooleanLiteral>,
    {
        let mut normalized = Vec::<Vec<BooleanLiteral>>::new();
        for clause in clauses {
            let mut by_feature = BTreeMap::<FeatureId, bool>::new();
            let mut tautology = false;
            for literal in clause {
                if by_feature
                    .get(&literal.feature)
                    .is_some_and(|positive| *positive != literal.positive)
                {
                    tautology = true;
                    break;
                }
                by_feature.insert(literal.feature, literal.positive);
            }
            if tautology {
                continue;
            }
            let mut clause = by_feature
                .into_iter()
                .map(|(feature, positive)| BooleanLiteral { feature, positive })
                .collect::<Vec<_>>();
            clause.sort_unstable();
            if clause.is_empty() {
                return Self {
                    clauses: vec![Vec::new()],
                };
            }
            normalized.push(clause);
        }
        normalized.sort_by(|left, right| left.len().cmp(&right.len()).then(left.cmp(right)));
        normalized.dedup();

        let mut irredundant = Vec::<Vec<BooleanLiteral>>::new();
        for clause in normalized {
            if irredundant.iter().any(|kept| sorted_subset(kept, &clause)) {
                continue;
            }
            irredundant.retain(|kept| !sorted_subset(&clause, kept));
            irredundant.push(clause);
        }
        irredundant.sort_by(|left, right| left.len().cmp(&right.len()).then(left.cmp(right)));
        Self {
            clauses: irredundant,
        }
    }

    pub fn complete() -> Self {
        Self { clauses: vec![] }
    }

    pub fn empty() -> Self {
        Self {
            clauses: vec![vec![]],
        }
    }

    pub fn is_complete(&self) -> bool {
        self.clauses.is_empty()
    }

    pub fn is_empty(&self) -> bool {
        self.clauses.iter().any(Vec::is_empty)
    }

    pub fn variables(&self) -> Vec<FeatureId> {
        self.clauses
            .iter()
            .flatten()
            .map(|literal| literal.feature)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn eval(&self, assignment: &[bool]) -> bool {
        self.clauses.iter().all(|clause| {
            clause
                .iter()
                .copied()
                .any(|literal| literal.eval(assignment))
        })
    }

    pub fn polarity_dual(&self) -> Self {
        Self::new(
            self.clauses
                .iter()
                .map(|clause| clause.iter().copied().map(BooleanLiteral::negated)),
        )
    }
}

/// Canonical witness that a formula is star-nested Horn (Definition 4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarNestedHorn {
    pub clauses: Vec<Vec<BooleanLiteral>>,
    /// The chain `S_0, ..., S_q`, represented by feature identifiers for the
    /// negative literals `not x_i`. `S_0` is always empty.
    pub negative_sets: Vec<Vec<FeatureId>>,
    pub positive_literals: Vec<FeatureId>,
}

impl StarNestedHorn {
    pub fn formula(&self) -> BooleanFormula {
        BooleanFormula::new(self.clauses.clone())
    }

    /// Proposition 2's recursive, theorem-preserving complement construction.
    pub fn complement(&self) -> Option<Self> {
        let formula = complement_star_nested_horn(&self.formula())?;
        classify_star_nested_horn(&formula)
    }
}

pub fn is_star_nested_horn(formula: &BooleanFormula) -> bool {
    classify_star_nested_horn(formula).is_some()
}

pub fn classify_star_nested_horn(formula: &BooleanFormula) -> Option<StarNestedHorn> {
    let formula = BooleanFormula::new(formula.clauses.clone());
    let mut negative_sets = BTreeSet::<Vec<FeatureId>>::from([Vec::new()]);
    let mut positive_literals = BTreeSet::<FeatureId>::new();
    for clause in &formula.clauses {
        let positives = clause
            .iter()
            .filter(|literal| literal.positive)
            .collect::<Vec<_>>();
        if positives.len() > 1 {
            return None;
        }
        if let Some(positive) = positives.first() {
            positive_literals.insert(positive.feature);
        }
        let negatives = clause
            .iter()
            .filter(|literal| !literal.positive)
            .map(|literal| literal.feature)
            .collect::<Vec<_>>();
        negative_sets.insert(negatives);
    }
    let mut negative_sets = negative_sets.into_iter().collect::<Vec<_>>();
    negative_sets.sort_by(|left, right| left.len().cmp(&right.len()).then(left.cmp(right)));
    if negative_sets
        .windows(2)
        .any(|sets| !sorted_subset(&sets[0], &sets[1]))
    {
        return None;
    }
    Some(StarNestedHorn {
        clauses: formula.clauses,
        negative_sets,
        positive_literals: positive_literals.into_iter().collect(),
    })
}

fn complement_star_nested_horn(formula: &BooleanFormula) -> Option<BooleanFormula> {
    let witness = classify_star_nested_horn(formula)?;
    let formula = witness.formula();
    if formula.is_complete() {
        return Some(BooleanFormula::empty());
    }
    if formula.is_empty() {
        return Some(BooleanFormula::complete());
    }

    let unit_positives = formula
        .clauses
        .iter()
        .filter(|clause| clause.len() == 1 && clause[0].positive)
        .map(|clause| clause[0].feature)
        .collect::<BTreeSet<_>>();
    let remaining = formula
        .clauses
        .iter()
        .filter(|clause| !(clause.len() == 1 && clause[0].positive))
        .collect::<Vec<_>>();
    if remaining.is_empty() {
        return Some(BooleanFormula::new([unit_positives
            .into_iter()
            .map(|feature| BooleanLiteral::new(feature, false))]));
    }

    let first_negative_set = remaining
        .iter()
        .map(|clause| {
            clause
                .iter()
                .filter(|literal| !literal.positive)
                .map(|literal| literal.feature)
                .collect::<Vec<_>>()
        })
        .filter(|set| !set.is_empty())
        .min_by(|left, right| left.len().cmp(&right.len()).then(left.cmp(right)))?;
    let first = first_negative_set.iter().copied().collect::<BTreeSet<_>>();
    let phi = BooleanFormula::new(remaining.into_iter().map(|clause| {
        clause
            .iter()
            .copied()
            .filter(|literal| literal.positive || !first.contains(&literal.feature))
    }));
    let complement_phi = complement_star_nested_horn(&phi)?;
    let common_negatives = unit_positives
        .iter()
        .copied()
        .map(|feature| BooleanLiteral::new(feature, false))
        .collect::<Vec<_>>();
    let mut result = Vec::<Vec<BooleanLiteral>>::new();
    for feature in first_negative_set {
        let mut clause = common_negatives.clone();
        clause.push(BooleanLiteral::new(feature, true));
        result.push(clause);
    }
    for clause in complement_phi.clauses {
        let mut extended = common_negatives.clone();
        extended.extend(clause);
        result.push(extended);
    }
    let result = BooleanFormula::new(result);
    classify_star_nested_horn(&result).map(|witness| witness.formula())
}

/// Canonical witness for the polarity dual of star-nested Horn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarNestedAntiHorn {
    pub clauses: Vec<Vec<BooleanLiteral>>,
    pub positive_sets: Vec<Vec<FeatureId>>,
    pub negative_literals: Vec<FeatureId>,
}

impl StarNestedAntiHorn {
    pub fn formula(&self) -> BooleanFormula {
        BooleanFormula::new(self.clauses.clone())
    }

    pub fn complement(&self) -> Option<Self> {
        let dual = self.formula().polarity_dual();
        let complement_dual = classify_star_nested_horn(&dual)?.complement()?.formula();
        classify_star_nested_antihorn(&complement_dual.polarity_dual())
    }
}

pub fn is_star_nested_antihorn(formula: &BooleanFormula) -> bool {
    classify_star_nested_antihorn(formula).is_some()
}

pub fn classify_star_nested_antihorn(formula: &BooleanFormula) -> Option<StarNestedAntiHorn> {
    let normalized = BooleanFormula::new(formula.clauses.clone());
    let dual = normalized.polarity_dual();
    let witness = classify_star_nested_horn(&dual)?;
    Some(StarNestedAntiHorn {
        clauses: normalized.clauses,
        positive_sets: witness.negative_sets,
        negative_literals: witness.positive_literals,
    })
}

/// The three nonconstant Square2CNF witnesses from Theorem 6, plus constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Square2CnfForm {
    Empty,
    Complete,
    FormI,
    FormII,
    FormIII,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Square2CnfRelation {
    pub form: Square2CnfForm,
    pub literals: Option<[BooleanLiteral; 4]>,
    pub clauses: Vec<Vec<BooleanLiteral>>,
}

impl Square2CnfRelation {
    pub fn formula(&self) -> BooleanFormula {
        BooleanFormula::new(self.clauses.clone())
    }

    /// Lemma 8 / Theorem 6 complement map: I <-> III and II <-> II.
    pub fn complement(&self) -> Option<Self> {
        match self.form {
            Square2CnfForm::Empty => classify_square_2cnf(&BooleanFormula::complete()),
            Square2CnfForm::Complete => classify_square_2cnf(&BooleanFormula::empty()),
            Square2CnfForm::FormI | Square2CnfForm::FormII | Square2CnfForm::FormIII => {
                let [a, b, c, d] = self.literals?;
                let mapped = [a.negated(), c.negated(), b.negated(), d.negated()];
                let form = match self.form {
                    Square2CnfForm::FormI => Square2CnfForm::FormIII,
                    Square2CnfForm::FormII => Square2CnfForm::FormII,
                    Square2CnfForm::FormIII => Square2CnfForm::FormI,
                    Square2CnfForm::Empty | Square2CnfForm::Complete => unreachable!(),
                };
                let formula = square_formula(form, mapped);
                classify_square_2cnf(&formula)
            }
        }
    }
}

pub fn is_square_2cnf(formula: &BooleanFormula) -> bool {
    classify_square_2cnf(formula).is_some()
}

pub fn classify_square_2cnf(formula: &BooleanFormula) -> Option<Square2CnfRelation> {
    let formula = BooleanFormula::new(formula.clauses.clone());
    if formula.is_complete() {
        return Some(Square2CnfRelation {
            form: Square2CnfForm::Complete,
            literals: None,
            clauses: formula.clauses,
        });
    }
    if formula.is_empty() {
        return Some(Square2CnfRelation {
            form: Square2CnfForm::Empty,
            literals: None,
            clauses: formula.clauses,
        });
    }
    if formula.clauses.iter().any(|clause| clause.len() > 2) || formula.variables().len() > 4 {
        return None;
    }
    let pool = formula
        .clauses
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    for form in [
        Square2CnfForm::FormI,
        Square2CnfForm::FormII,
        Square2CnfForm::FormIII,
    ] {
        for &a in &pool {
            for &b in &pool {
                for &c in &pool {
                    for &d in &pool {
                        let literals = [a, b, c, d];
                        if square_formula(form, literals) == formula {
                            return Some(Square2CnfRelation {
                                form,
                                literals: Some(literals),
                                clauses: formula.clauses,
                            });
                        }
                    }
                }
            }
        }
    }
    None
}

pub fn square_formula(form: Square2CnfForm, [a, b, c, d]: [BooleanLiteral; 4]) -> BooleanFormula {
    match form {
        Square2CnfForm::Empty => BooleanFormula::empty(),
        Square2CnfForm::Complete => BooleanFormula::complete(),
        Square2CnfForm::FormI => BooleanFormula::new([[a, b], [c, d]]),
        Square2CnfForm::FormII => BooleanFormula::new([[a, b], [b, c], [c, d]]),
        Square2CnfForm::FormIII => BooleanFormula::new([[a, b], [b, c], [a, d], [c, d]]),
    }
}

/// One normalized Boolean linear equation over GF(2) (Theorem 5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleGf2Equation {
    pub variables: Vec<FeatureId>,
    pub rhs: bool,
}

impl SingleGf2Equation {
    pub fn new<I>(variables: I, rhs: bool) -> Self
    where
        I: IntoIterator<Item = FeatureId>,
    {
        let mut coefficients = BTreeSet::new();
        for variable in variables {
            if !coefficients.insert(variable) {
                coefficients.remove(&variable);
            }
        }
        Self {
            variables: coefficients.into_iter().collect(),
            rhs,
        }
    }

    pub fn complement(&self) -> Self {
        Self {
            variables: self.variables.clone(),
            rhs: !self.rhs,
        }
    }

    pub fn eval(&self, assignment: &[bool]) -> bool {
        self.variables.iter().fold(false, |parity, feature| {
            parity ^ assignment.get(*feature as usize).copied().unwrap_or(false)
        }) == self.rhs
    }
}

fn sorted_subset<T: Ord>(left: &[T], right: &[T]) -> bool {
    let mut right_index = 0;
    for wanted in left {
        while right_index < right.len() && &right[right_index] < wanted {
            right_index += 1;
        }
        if right_index == right.len() || &right[right_index] != wanted {
            return false;
        }
        right_index += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(feature: FeatureId) -> BooleanLiteral {
        BooleanLiteral::new(feature, true)
    }

    fn n(feature: FeatureId) -> BooleanLiteral {
        BooleanLiteral::new(feature, false)
    }

    fn complements_partition(left: &BooleanFormula, right: &BooleanFormula, variables: usize) {
        for mask in 0..(1usize << variables) {
            let assignment = (0..variables)
                .map(|bit| ((mask >> bit) & 1) == 1)
                .collect::<Vec<_>>();
            assert_ne!(left.eval(&assignment), right.eval(&assignment));
        }
    }

    #[test]
    fn nested_horn_and_its_complement_are_recognized() {
        let formula = BooleanFormula::new([vec![n(0)], vec![n(0), n(1)], vec![n(0), n(1), p(2)]]);
        let witness = classify_star_nested_horn(&formula).unwrap();
        // The first clause subsumes the two larger clauses; normalization is
        // semantic and therefore keeps only the nonredundant chain entries.
        assert_eq!(witness.negative_sets, vec![vec![], vec![0]]);
        let complement = witness.complement().unwrap().formula();
        assert!(is_star_nested_horn(&complement));
        complements_partition(&formula, &complement, 3);
    }

    #[test]
    fn incomparable_horn_negative_sets_are_rejected() {
        let formula = BooleanFormula::new([vec![n(0), p(1)], vec![n(2), p(3)]]);
        assert!(!is_star_nested_horn(&formula));
    }

    #[test]
    fn anti_horn_is_the_exact_polarity_dual() {
        let formula = BooleanFormula::new([vec![p(0)], vec![p(0), p(1)], vec![p(0), p(1), n(2)]]);
        let witness = classify_star_nested_antihorn(&formula).unwrap();
        let complement = witness.complement().unwrap().formula();
        assert!(is_star_nested_antihorn(&complement));
        complements_partition(&formula, &complement, 3);
    }

    #[test]
    fn square_forms_and_complements_are_recognized() {
        let literals = [p(0), n(1), p(2), n(3)];
        for form in [
            Square2CnfForm::FormI,
            Square2CnfForm::FormII,
            Square2CnfForm::FormIII,
        ] {
            let formula = square_formula(form, literals);
            let witness = classify_square_2cnf(&formula).unwrap();
            let complement = witness.complement().unwrap().formula();
            assert!(is_square_2cnf(&complement));
            complements_partition(&formula, &complement, 4);
        }
    }

    #[test]
    fn arbitrary_two_cnf_outside_the_square_is_rejected() {
        let formula = BooleanFormula::new([vec![p(0), p(1)], vec![p(2), p(3)], vec![p(4), p(5)]]);
        assert!(!is_square_2cnf(&formula));
    }

    #[test]
    fn gf2_duplicate_variables_cancel_and_complement_partitions() {
        let equation = SingleGf2Equation::new([2, 0, 2, 1], true);
        assert_eq!(equation.variables, vec![0, 1]);
        let complement = equation.complement();
        for mask in 0..8usize {
            let assignment = (0..3)
                .map(|bit| ((mask >> bit) & 1) == 1)
                .collect::<Vec<_>>();
            assert_ne!(equation.eval(&assignment), complement.eval(&assignment));
        }
    }
}
