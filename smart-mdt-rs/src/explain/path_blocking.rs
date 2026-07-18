//! Exact opposite-leaf path blocking for certified Boolean trees.

use crate::{
    logic::{next_theory_state, Literal, PathTheoryState, Predicate},
    sat::{antihorn_sat, gf2_system_satisfiable, horn_sat, two_sat, Cnf, Gf2Equation},
    tree::TreeNode,
    ClassId, FeatureId,
};
use std::collections::{BTreeMap, BTreeSet};

type RawLiteral = (FeatureId, bool);
type RawClause = Vec<RawLiteral>;

/// Returns whether a Boolean completion agreeing with `selected_features` can
/// reach an opposite-class leaf. The caller must first validate that the
/// supplied feature domain and instance are Boolean.
pub(crate) fn certified_opposite_completion_exists(
    tree: &TreeNode,
    instance: &[f64],
    target_class: ClassId,
    selected_features: &[FeatureId],
) -> std::result::Result<bool, String> {
    if selected_features
        .iter()
        .any(|&feature| feature as usize >= instance.len())
    {
        return Err("selected feature is out of bounds".into());
    }
    let mut path = Vec::new();
    visit_opposite_paths(
        tree,
        instance,
        target_class,
        selected_features,
        PathTheoryState::Uncommitted,
        &mut path,
    )
}

fn visit_opposite_paths<'a>(
    tree: &'a TreeNode,
    instance: &[f64],
    target_class: ClassId,
    selected_features: &[FeatureId],
    state: PathTheoryState,
    path: &mut Vec<(&'a Predicate, bool)>,
) -> std::result::Result<bool, String> {
    match tree {
        TreeNode::Leaf { class, .. } => {
            if *class == target_class {
                Ok(false)
            } else {
                path_is_satisfiable(path, state, instance, selected_features)
            }
        }
        TreeNode::Internal {
            predicate,
            left,
            right,
            ..
        } => {
            let next = next_theory_state(state, predicate).map_err(|error| error.to_string())?;
            path.push((predicate, true));
            let left_has_completion =
                visit_opposite_paths(left, instance, target_class, selected_features, next, path)?;
            path.pop();
            if left_has_completion {
                return Ok(true);
            }
            path.push((predicate, false));
            let right_has_completion =
                visit_opposite_paths(right, instance, target_class, selected_features, next, path)?;
            path.pop();
            Ok(right_has_completion)
        }
    }
}

fn path_is_satisfiable(
    path: &[(&Predicate, bool)],
    state: PathTheoryState,
    instance: &[f64],
    selected_features: &[FeatureId],
) -> std::result::Result<bool, String> {
    match state {
        PathTheoryState::AffineGf2 => affine_path_is_satisfiable(path, instance, selected_features),
        PathTheoryState::Uncommitted
        | PathTheoryState::Horn
        | PathTheoryState::AntiHorn
        | PathTheoryState::TwoSat => {
            cnf_path_is_satisfiable(path, state, instance, selected_features)
        }
    }
}

fn cnf_path_is_satisfiable(
    path: &[(&Predicate, bool)],
    state: PathTheoryState,
    instance: &[f64],
    selected_features: &[FeatureId],
) -> std::result::Result<bool, String> {
    let mut clauses = Vec::<RawClause>::new();
    for &(predicate, branch_true) in path {
        let literal_clauses = predicate_branch_cnf(predicate, branch_true)?;
        for clause in literal_clauses {
            if let Some(normalized) = normalize_clause(&clause, instance.len())? {
                clauses.push(normalized);
            }
        }
    }

    let path_features = clauses
        .iter()
        .flatten()
        .map(|&(feature, _)| feature)
        .collect::<BTreeSet<_>>();
    for &feature in selected_features {
        if path_features.contains(&feature) {
            clauses.push(vec![(feature, instance[feature as usize] == 1.0)]);
        }
    }
    let cnf = compact_cnf(&clauses)?;
    let num_vars = clauses
        .iter()
        .flatten()
        .map(|&(feature, _)| feature)
        .collect::<BTreeSet<_>>()
        .len();

    match state {
        PathTheoryState::Uncommitted | PathTheoryState::Horn => {
            if cnf
                .iter()
                .any(|clause| clause.iter().filter(|&&literal| literal > 0).count() > 1)
            {
                return Err("path encoding is not Horn after Boolean normalization".into());
            }
            Ok(horn_sat(num_vars, &cnf))
        }
        PathTheoryState::AntiHorn => {
            if cnf
                .iter()
                .any(|clause| clause.iter().filter(|&&literal| literal < 0).count() > 1)
            {
                return Err("path encoding is not AntiHorn after Boolean normalization".into());
            }
            Ok(antihorn_sat(num_vars, &cnf))
        }
        PathTheoryState::TwoSat => {
            if cnf.iter().any(|clause| clause.len() > 2) {
                return Err("path encoding is not 2-CNF".into());
            }
            Ok(two_sat(num_vars, &cnf))
        }
        PathTheoryState::AffineGf2 => Err("affine path was sent to a CNF backend".into()),
    }
}

fn predicate_branch_cnf(
    predicate: &Predicate,
    branch_true: bool,
) -> std::result::Result<Vec<Vec<Literal>>, String> {
    if !branch_true {
        return match predicate {
            Predicate::Affine { .. } => {
                Err("affine predicate complement requires the GF(2) backend".into())
            }
            Predicate::EmpiricalAffine { .. } => {
                Err("empirical affine predicate has no certified encoding".into())
            }
            _ => Ok(crate::logic::complement_cnf(predicate).clauses),
        };
    }
    match predicate {
        Predicate::Unary(literal) => Ok(vec![vec![*literal]]),
        Predicate::HornClause(literals) | Predicate::AntiHornClause(literals) => {
            Ok(vec![literals.clone()])
        }
        Predicate::Square2Cnf { a, b, c, d } => Ok(vec![vec![*a, *b], vec![*c, *d]]),
        Predicate::Affine { .. } => Err("affine predicate requires the GF(2) backend".into()),
        Predicate::EmpiricalAffine { .. } => {
            Err("empirical affine predicate has no certified encoding".into())
        }
    }
}

/// `None` denotes a tautological clause that can be discarded.
fn normalize_clause(
    literals: &[Literal],
    feature_count: usize,
) -> std::result::Result<Option<RawClause>, String> {
    let mut normalized = BTreeMap::<FeatureId, bool>::new();
    for literal in literals {
        if literal.atom.feature as usize >= feature_count {
            return Err("predicate feature is out of bounds".into());
        }
        let at_zero = literal.eval_value(0.0);
        let at_one = literal.eval_value(1.0);
        match (at_zero, at_one) {
            (true, true) => return Ok(None),
            (false, false) => {}
            (false, true) | (true, false) => {
                let positive = at_one;
                if normalized
                    .get(&literal.atom.feature)
                    .is_some_and(|existing| *existing != positive)
                {
                    return Ok(None);
                }
                normalized.insert(literal.atom.feature, positive);
            }
        }
    }
    Ok(Some(normalized.into_iter().collect()))
}

fn compact_cnf(clauses: &[RawClause]) -> std::result::Result<Cnf, String> {
    let features = clauses
        .iter()
        .flatten()
        .map(|&(feature, _)| feature)
        .collect::<BTreeSet<_>>();
    if features.len() > i32::MAX as usize {
        return Err("too many variables for signed CNF encoding".into());
    }
    let indices = features
        .into_iter()
        .enumerate()
        .map(|(index, feature)| (feature, index as i32 + 1))
        .collect::<BTreeMap<_, _>>();
    Ok(clauses
        .iter()
        .map(|clause| {
            clause
                .iter()
                .map(|&(feature, positive)| {
                    let variable = indices[&feature];
                    if positive {
                        variable
                    } else {
                        -variable
                    }
                })
                .collect()
        })
        .collect())
}

fn affine_path_is_satisfiable(
    path: &[(&Predicate, bool)],
    instance: &[f64],
    selected_features: &[FeatureId],
) -> std::result::Result<bool, String> {
    let mut raw_equations = Vec::<(BTreeSet<FeatureId>, bool)>::new();
    for &(predicate, branch_true) in path {
        match predicate {
            Predicate::Unary(literal) => {
                raw_equations.push(normalize_affine_literals(
                    std::slice::from_ref(literal),
                    branch_true,
                    instance.len(),
                )?);
            }
            Predicate::Affine { literals, rhs } => {
                raw_equations.push(normalize_affine_literals(
                    literals,
                    if branch_true { *rhs } else { !*rhs },
                    instance.len(),
                )?);
            }
            _ => return Err("non-affine predicate occurs on an affine path".into()),
        }
    }
    let path_features = raw_equations
        .iter()
        .flat_map(|(features, _)| features.iter().copied())
        .collect::<BTreeSet<_>>();
    if path_features.len() > 128 {
        return Err("GF(2) path contains more than 128 distinct variables".into());
    }
    let indices = path_features
        .iter()
        .copied()
        .enumerate()
        .map(|(index, feature)| (feature, index))
        .collect::<BTreeMap<_, _>>();
    let mut equations = raw_equations
        .into_iter()
        .map(|(features, rhs)| {
            Gf2Equation::from_vars(
                &features
                    .iter()
                    .map(|feature| indices[feature])
                    .collect::<Vec<_>>(),
                rhs,
            )
        })
        .collect::<Vec<_>>();
    for &feature in selected_features {
        if let Some(&index) = indices.get(&feature) {
            equations.push(Gf2Equation::new(
                1u128 << index,
                instance[feature as usize] == 1.0,
            ));
        }
    }
    Ok(gf2_system_satisfiable(&equations))
}

fn normalize_affine_literals(
    literals: &[Literal],
    mut rhs: bool,
    feature_count: usize,
) -> std::result::Result<(BTreeSet<FeatureId>, bool), String> {
    let mut features = BTreeSet::new();
    for literal in literals {
        if literal.atom.feature as usize >= feature_count {
            return Err("predicate feature is out of bounds".into());
        }
        let at_zero = literal.eval_value(0.0);
        let at_one = literal.eval_value(1.0);
        match (at_zero, at_one) {
            (false, false) => {}
            (true, true) => rhs = !rhs,
            (false, true) => {
                if !features.insert(literal.atom.feature) {
                    features.remove(&literal.atom.feature);
                }
            }
            (true, false) => {
                if !features.insert(literal.atom.feature) {
                    features.remove(&literal.atom.feature);
                }
                rhs = !rhs;
            }
        }
    }
    Ok((features, rhs))
}
