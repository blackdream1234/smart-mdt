use crate::{
    data::{class_counts, predicate_mask, Dataset},
    logic::Predicate,
    search::unary::generate_unary,
    search::{final_score, information_gain, ScoreWeights, SplitCandidate},
};
/// Generates Square2CNF candidates `(a or b) and (c or d)` from top literals.
pub fn generate_square2cnf(data: &Dataset, min_leaf: usize, beam: usize) -> Vec<SplitCandidate> {
    let base = generate_unary(data, min_leaf);
    let lits: Vec<_> = base
        .iter()
        .take(beam.max(4))
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
    for w in lits.windows(4) {
        let p = Predicate::Square2Cnf {
            a: w[0],
            b: w[1],
            c: w[2],
            d: w[3],
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
            score: final_score(gain, 4.0, 4.0, 4.0, true, ScoreWeights::default()),
            left_count: l,
            right_count: r,
        });
    }
    out
}
