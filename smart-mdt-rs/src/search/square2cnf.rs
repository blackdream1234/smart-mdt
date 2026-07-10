use crate::{
    data::{class_counts, predicate_mask, BitSet, Dataset},
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    search::{final_score, information_gain, ScoreWeights, SplitCandidate},
};
use std::collections::BTreeSet;

/// Diagnostics for Square2CNF candidate generation in the learner path.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Square2CnfSearchDiagnostics {
    pub candidate_count_before_filtering: usize,
    pub candidate_count_after_filtering: usize,
    pub best_final_score: f64,
    pub best_gain: f64,
    pub selected_predicate: String,
    pub leaf_reason: String,
}

/// Generates polarity-complete Square2CNF candidates from ranked literals.
pub fn generate_square2cnf(data: &Dataset, min_leaf: usize, beam: usize) -> Vec<SplitCandidate> {
    generate_square2cnf_with_diagnostics(data, min_leaf, beam).0
}

/// Generates Square2CNF candidates and diagnostics using the same path as the learner.
pub fn generate_square2cnf_with_diagnostics(
    data: &Dataset,
    min_leaf: usize,
    beam: usize,
) -> (Vec<SplitCandidate>, Square2CnfSearchDiagnostics) {
    let literals = ranked_literals(data);
    let selected: Vec<_> = literals.into_iter().take(beam.max(4)).collect();
    let mut clauses = Vec::new();
    for i in 0..selected.len() {
        for j in i + 1..selected.len() {
            let a = selected[i];
            let b = selected[j];
            if same_atom_opposite_polarity(a, b) {
                continue;
            }
            clauses.push((a, b));
        }
    }
    clauses.sort_by(|(a1, b1), (a2, b2)| {
        clause_gain(data, *a2, *b2).total_cmp(&clause_gain(data, *a1, *b1))
    });

    let mut before = 0usize;
    let mut seen_masks = BTreeSet::new();
    let mut out = Vec::new();
    for (idx, &(a, b)) in clauses.iter().enumerate() {
        before += 1;
        consider_candidate(
            data,
            min_leaf,
            &mut seen_masks,
            &mut out,
            Predicate::Square2Cnf { a, b, c: a, d: b },
        );
        for &(c, d) in clauses.iter().skip(idx + 1) {
            before += 1;
            consider_candidate(
                data,
                min_leaf,
                &mut seen_masks,
                &mut out,
                Predicate::Square2Cnf { a, b, c, d },
            );
        }
    }

    out.sort_by(|a, b| b.score.final_score.total_cmp(&a.score.final_score));
    let best = out.first();
    let diag = Square2CnfSearchDiagnostics {
        candidate_count_before_filtering: before,
        candidate_count_after_filtering: out.len(),
        best_final_score: best.map_or(0.0, |c| c.score.final_score),
        best_gain: best.map_or(0.0, |c| c.score.predictive_gain),
        selected_predicate: best.map_or_else(String::new, |c| format!("{:?}", c.predicate)),
        leaf_reason: if out.is_empty() {
            "no_square2cnf_candidate_after_filtering".into()
        } else {
            String::new()
        },
    };
    (out, diag)
}

fn consider_candidate(
    data: &Dataset,
    min_leaf: usize,
    seen_masks: &mut BTreeSet<Vec<bool>>,
    out: &mut Vec<SplitCandidate>,
    p: Predicate,
) {
    let m = predicate_mask(&data.features, &p);
    let l = m.count_ones();
    let r = data.labels.len().saturating_sub(l);
    if l < min_leaf || r < min_leaf {
        return;
    }
    let sig = mask_signature(&m);
    if !seen_masks.insert(sig) {
        return;
    }
    out.push(score_candidate(data, p, &m, l, r));
}

fn ranked_literals(data: &Dataset) -> Vec<Literal> {
    let mut lits = Vec::new();
    for f in 0..data.features.cols() {
        let mut vals = data.features.column(f as u32).to_vec();
        vals.sort_by(f64::total_cmp);
        vals.dedup();
        for w in vals.windows(2) {
            let atom = ThresholdAtom {
                feature: f as u32,
                threshold_id: 0,
                threshold: (w[0] + w[1]) / 2.0,
                op: ThresholdOp::GreaterEqual,
            };
            lits.push(Literal {
                atom,
                positive: true,
            });
            lits.push(Literal {
                atom,
                positive: false,
            });
        }
    }
    lits.sort_by(|a, b| unary_gain(data, b).total_cmp(&unary_gain(data, a)));
    lits
}

fn unary_gain(data: &Dataset, lit: &Literal) -> f64 {
    let p = Predicate::Unary(*lit);
    let m = predicate_mask(&data.features, &p);
    let l = m.count_ones();
    let r = data.labels.len().saturating_sub(l);
    score_candidate(data, p, &m, l, r).score.predictive_gain
}

fn clause_gain(data: &Dataset, a: Literal, b: Literal) -> f64 {
    let p = Predicate::Square2Cnf { a, b, c: a, d: b };
    let m = predicate_mask(&data.features, &p);
    let l = m.count_ones();
    let r = data.labels.len().saturating_sub(l);
    score_candidate(data, p, &m, l, r).score.predictive_gain
}

fn score_candidate(data: &Dataset, p: Predicate, m: &BitSet, l: usize, r: usize) -> SplitCandidate {
    let classes = data.class_count().max(2);
    let mut parent = vec![0; classes];
    for &y in &data.labels {
        parent[y as usize] += 1;
    }
    let lc = class_counts(&data.labels, m, classes);
    let rc: Vec<_> = parent.iter().zip(&lc).map(|(a, b)| a - b).collect();
    let gain = information_gain(&parent, &lc, &rc);
    SplitCandidate {
        predicate: p,
        score: final_score(gain, 4.0, 4.0, 4.0, true, ScoreWeights::default()),
        left_count: l,
        right_count: r,
    }
}

fn same_atom_opposite_polarity(a: Literal, b: Literal) -> bool {
    a.atom.feature == b.atom.feature
        && a.atom.threshold == b.atom.threshold
        && a.atom.op == b.atom.op
        && a.positive != b.positive
}

fn mask_signature(mask: &BitSet) -> Vec<bool> {
    (0..mask.len()).map(|i| mask.get(i)).collect()
}
