use crate::{
    data::{class_counts, predicate_mask, Dataset},
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    search::{final_score, information_gain, ScoreWeights, SplitCandidate},
};
/// Generates certified unary threshold candidates.
pub fn generate_unary(data: &Dataset, min_leaf: usize) -> Vec<SplitCandidate> {
    let classes = data.class_count();
    let parent = {
        let mut c = vec![0; classes];
        for &l in &data.labels {
            c[l as usize] += 1;
        }
        c
    };
    let mut out = Vec::new();
    for f in 0..data.features.cols() {
        let mut vals = data.features.column(f as u32).to_vec();
        vals.sort_by(f64::total_cmp);
        vals.dedup();
        for w in vals.windows(2) {
            let t = (w[0] + w[1]) / 2.0;
            let lit = Literal {
                atom: ThresholdAtom {
                    feature: f as u32,
                    threshold_id: 0,
                    threshold: t,
                    op: ThresholdOp::LessThan,
                },
                positive: true,
            };
            let p = Predicate::Unary(lit);
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
                score: final_score(gain, 1.0, 1.0, 1.0, true, ScoreWeights::default()),
                left_count: l,
                right_count: r,
            });
        }
    }
    out
}
