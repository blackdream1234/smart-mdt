use super::WeakAxpResult;
use crate::{
    data::ColumnMajorMatrix,
    logic::{
        complement_cnf, Backend, CertificateMetadata, LanguageFamily, Literal, PathCertificate,
        Predicate, ThresholdAtom, ThresholdOp,
    },
    sat::{
        affine_gf2_empirical::gf2_satisfiable_with_assumptions, antihorn_sat, horn_sat, two_sat,
        Cnf,
    },
    tree::TreeNode,
    ClassId, FeatureId,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct PathStep<'a> {
    predicate: &'a Predicate,
    branch_true: bool,
}

fn opposite_paths<'a>(
    tree: &'a TreeNode,
    target: ClassId,
    current: &mut Vec<PathStep<'a>>,
    out: &mut Vec<Vec<PathStep<'a>>>,
) {
    match tree {
        TreeNode::Leaf { class, .. } => {
            if *class != target {
                out.push(current.clone());
            }
        }
        TreeNode::Internal {
            predicate,
            left,
            right,
            ..
        } => {
            current.push(PathStep {
                predicate,
                branch_true: true,
            });
            opposite_paths(left, target, current, out);
            current.pop();
            current.push(PathStep {
                predicate,
                branch_true: false,
            });
            opposite_paths(right, target, current, out);
            current.pop();
        }
    }
}

fn path_family(
    paths: &[Vec<PathStep<'_>>],
) -> Result<(LanguageFamily, Backend, PathCertificate), String> {
    let mut families = Vec::new();
    for path in paths {
        for step in path {
            families.push(step.predicate.language());
        }
    }
    if families.is_empty() || families.iter().all(|f| *f == LanguageFamily::Unary) {
        return Ok((
            LanguageFamily::Unary,
            Backend::StructuralHorn,
            PathCertificate::HornCnf,
        ));
    }
    if families
        .iter()
        .all(|f| matches!(f, LanguageFamily::Unary | LanguageFamily::Horn))
    {
        return Ok((
            LanguageFamily::Horn,
            Backend::StructuralHorn,
            PathCertificate::HornCnf,
        ));
    }
    if families.iter().all(|f| *f == LanguageFamily::AntiHorn) {
        return Ok((
            LanguageFamily::AntiHorn,
            Backend::StructuralAntiHorn,
            PathCertificate::AntiHornCnf,
        ));
    }
    if families.iter().all(|f| *f == LanguageFamily::Square2Cnf) {
        return Ok((
            LanguageFamily::Square2Cnf,
            Backend::TwoSat,
            PathCertificate::TwoCnf,
        ));
    }
    if families
        .iter()
        .all(|f| *f == LanguageFamily::EmpiricalAffine)
    {
        return Ok((
            LanguageFamily::EmpiricalAffine,
            Backend::Gf2Gaussian,
            PathCertificate::Gf2System,
        ));
    }
    Err("mixed paths are not theorem-certified".into())
}

#[derive(Default)]
struct VarMap {
    vars: BTreeMap<(FeatureId, u8, u64), i32>,
}
impl VarMap {
    fn var(&mut self, atom: ThresholdAtom) -> i32 {
        let key = (atom.feature, op_key(atom.op), atom.threshold.to_bits());
        if let Some(v) = self.vars.get(&key) {
            *v
        } else {
            let v = self.vars.len() as i32 + 1;
            self.vars.insert(key, v);
            v
        }
    }
    fn len(&self) -> usize {
        self.vars.len()
    }
}

fn sat_lit(l: Literal, vars: &mut VarMap) -> i32 {
    let v = vars.var(l.atom);
    if l.positive {
        v
    } else {
        -v
    }
}

fn append_predicate_cnf(p: &Predicate, branch_true: bool, vars: &mut VarMap, cnf: &mut Cnf) {
    if !branch_true {
        for cl in complement_cnf(p).clauses {
            cnf.push(cl.into_iter().map(|l| sat_lit(l, vars)).collect());
        }
        return;
    }
    match p {
        Predicate::Unary(l) => cnf.push(vec![sat_lit(*l, vars)]),
        Predicate::HornClause(ls) | Predicate::AntiHornClause(ls) => {
            cnf.push(ls.iter().map(|l| sat_lit(*l, vars)).collect());
        }
        Predicate::Square2Cnf { a, b, c, d } => {
            cnf.push(vec![sat_lit(*a, vars), sat_lit(*b, vars)]);
            cnf.push(vec![sat_lit(*c, vars), sat_lit(*d, vars)]);
        }
        Predicate::EmpiricalAffine { .. } => {}
    }
}

fn add_assignment_units(
    vars: &VarMap,
    instance: &[f64],
    selected_features: &[FeatureId],
    cnf: &mut Cnf,
) {
    for ((feature, op, threshold_bits), var) in &vars.vars {
        if selected_features.iter().any(|f| f == feature) {
            let atom = ThresholdAtom {
                feature: *feature,
                threshold_id: 0,
                threshold: f64::from_bits(*threshold_bits),
                op: key_op(*op),
            };
            let truth = atom.eval_value(instance[*feature as usize]);
            cnf.push(vec![if truth { *var } else { -*var }]);
        }
    }
}

fn op_key(op: ThresholdOp) -> u8 {
    match op {
        ThresholdOp::LessThan => 0,
        ThresholdOp::GreaterEqual => 1,
    }
}

fn key_op(key: u8) -> ThresholdOp {
    if key == 0 {
        ThresholdOp::LessThan
    } else {
        ThresholdOp::GreaterEqual
    }
}

fn cnf_path_satisfiable(
    path: &[PathStep<'_>],
    backend: Backend,
    instance: &[f64],
    selected_features: &[FeatureId],
) -> bool {
    let mut vars = VarMap::default();
    let mut cnf = Vec::new();
    for step in path {
        append_predicate_cnf(step.predicate, step.branch_true, &mut vars, &mut cnf);
    }
    add_assignment_units(&vars, instance, selected_features, &mut cnf);
    match backend {
        Backend::StructuralHorn => horn_sat(vars.len(), &cnf),
        Backend::StructuralAntiHorn => antihorn_sat(vars.len(), &cnf),
        Backend::TwoSat => two_sat(vars.len(), &cnf),
        _ => false,
    }
}

fn affine_lit_coeff(l: Literal) -> Option<(FeatureId, bool)> {
    if l.atom.threshold != 0.5 {
        return None;
    }
    let eval_one = l.eval_value(1.0);
    Some((l.atom.feature, !eval_one))
}

fn affine_path_satisfiable(
    path: &[PathStep<'_>],
    domain: &ColumnMajorMatrix,
    instance: &[f64],
    selected_features: &[FeatureId],
) -> Result<bool, String> {
    let mut features = BTreeMap::new();
    let mut rows = Vec::new();
    for step in path {
        let Predicate::EmpiricalAffine { literals, parity } = step.predicate else {
            return Err("non-affine predicate in affine path".into());
        };
        let mut rhs = if step.branch_true { *parity } else { !*parity };
        let mut mask = 0u128;
        for &lit in literals {
            let (feature, flip) =
                affine_lit_coeff(lit).ok_or_else(|| "non-Boolean affine scope".to_string())?;
            if (0..domain.rows()).any(|r| {
                let v = domain.get(r, feature);
                v != 0.0 && v != 1.0
            }) {
                return Err("non-Boolean affine scope".into());
            }
            let next = features.len();
            let var = *features.entry(feature).or_insert(next);
            mask ^= 1u128 << var;
            rhs ^= flip;
        }
        rows.push((mask, rhs));
    }
    if features.len() > 20 {
        return Err("GF(2) variable limit exceeded".into());
    }
    let mut assumptions = Vec::new();
    for &feature in selected_features {
        if let Some(&var) = features.get(&feature) {
            let v = instance[feature as usize];
            if v != 0.0 && v != 1.0 {
                return Err("non-Boolean affine assumption".into());
            }
            assumptions.push((var, v == 1.0));
        }
    }
    Ok(gf2_satisfiable_with_assumptions(
        features.len(),
        &rows,
        &assumptions,
    ))
}

/// Checks weak AXp by executing the selected backend against every opposite path.
pub fn weak_axp_check(
    tree: &TreeNode,
    domain: &ColumnMajorMatrix,
    instance: &[f64],
    target_class: ClassId,
    selected_features: &[FeatureId],
    theorem_mode: bool,
) -> WeakAxpResult {
    let mut paths = Vec::new();
    opposite_paths(tree, target_class, &mut Vec::new(), &mut paths);
    let opposite_paths_checked = paths.len();
    if paths.is_empty() {
        return WeakAxpResult {
            is_weak_axp: true,
            metadata: CertificateMetadata::rejected(
                theorem_mode,
                LanguageFamily::Unary,
                "no opposite paths; no backend invoked",
            ),
            opposite_paths_checked,
        };
    }
    let Ok((family, backend, cert)) = path_family(&paths) else {
        return WeakAxpResult {
            is_weak_axp: false,
            metadata: CertificateMetadata::rejected(
                theorem_mode,
                LanguageFamily::EmpiricalMixed,
                "mixed paths are not theorem-certified",
            ),
            opposite_paths_checked,
        };
    };
    if theorem_mode && backend == Backend::Gf2Gaussian {
        return WeakAxpResult {
            is_weak_axp: false,
            metadata: CertificateMetadata::rejected(
                true,
                LanguageFamily::EmpiricalAffine,
                "affine backend is empirical, not theorem-certified",
            ),
            opposite_paths_checked,
        };
    }

    for path in &paths {
        let sat = if backend == Backend::Gf2Gaussian {
            match affine_path_satisfiable(path, domain, instance, selected_features) {
                Ok(v) => v,
                Err(reason) => {
                    return WeakAxpResult {
                        is_weak_axp: false,
                        metadata: CertificateMetadata::rejected(theorem_mode, family, reason),
                        opposite_paths_checked,
                    };
                }
            }
        } else {
            cnf_path_satisfiable(path, backend, instance, selected_features)
        };
        if sat {
            return WeakAxpResult {
                is_weak_axp: false,
                metadata: CertificateMetadata::new(theorem_mode, family, backend, cert),
                opposite_paths_checked,
            };
        }
    }

    WeakAxpResult {
        is_weak_axp: true,
        metadata: CertificateMetadata::new(theorem_mode, family, backend, cert),
        opposite_paths_checked,
    }
}
