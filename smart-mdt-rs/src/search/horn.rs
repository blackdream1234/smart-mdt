use crate::{
    data::{class_counts, predicate_mask, Dataset},
    logic::{Literal, Predicate},
    search::unary::generate_unary,
    search::{final_score, information_gain, ScoreWeights, SplitCandidate},
};
/// Generates Horn OR clauses of arity 2 from top unary literals.
pub fn generate_horn(data: &Dataset, min_leaf: usize, beam: usize) -> Vec<SplitCandidate> {
    combine(data, min_leaf, beam, true)
}
pub(crate) fn combine(
    data: &Dataset,
    min_leaf: usize,
    beam: usize,
    horn: bool,
) -> Vec<SplitCandidate> {
    let base = generate_unary(data, min_leaf);
    let lits: Vec<Literal> = base
        .iter()
        .take(beam.max(2))
        .filter_map(|c| {
            if let Predicate::Unary(l) = c.predicate {
                Some(l)
            } else {
                None
            }
        })
        .collect();
    let classes = data.class_count();
    let mut parent = vec![0; classes];
    for &l in &data.labels {
        parent[l as usize] += 1;
    }
    let mut out = Vec::new();
    for i in 0..lits.len() {
        for j in i + 1..lits.len() {
            let ls = vec![lits[i], lits[j]];
            let positives = ls.iter().filter(|l| l.positive).count();
            if (horn && positives > 1) || (!horn && ls.len() - positives > 1) {
                continue;
            }
            let p = if horn {
                Predicate::HornClause(ls)
            } else {
                Predicate::AntiHornClause(ls)
            };
            let m = predicate_mask(&data.features, &p);
            let l = m.count_ones();
            let r = data.labels.len() - l;
            if l < min_leaf || r < min_leaf {
                continue;
            }
            let lc = class_counts(&data.labels, &m, classes);
            let rc: Vec<_> = parent.iter().zip(&lc).map(|(a, b)| a - b).collect();
            let gain = information_gain(&parent, &lc, &rc);
            out.push(SplitCandidate {
                predicate: p,
                score: final_score(gain, 2.0, 2.0, 2.0, true, ScoreWeights::default()),
                left_count: l,
                right_count: r,
            });
        }
    }
    out
}
