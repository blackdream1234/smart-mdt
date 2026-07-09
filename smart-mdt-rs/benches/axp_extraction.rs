use smart_mdt_rs::{
    data::ColumnMajorMatrix,
    explain::extract_axp_deletion,
    logic::{Literal, Predicate, ThresholdAtom, ThresholdOp},
    tree::TreeNode,
};
use std::time::Instant;

fn ge(feature: u32) -> Literal {
    Literal {
        atom: ThresholdAtom {
            feature,
            threshold_id: 0,
            threshold: 0.5,
            op: ThresholdOp::GreaterEqual,
        },
        positive: true,
    }
}

fn main() {
    let tree = TreeNode::Internal {
        predicate: Predicate::HornClause(vec![ge(0), ge(1)]),
        majority_class: 0,
        left: Box::new(TreeNode::Leaf {
            class: 1,
            samples: 3,
        }),
        right: Box::new(TreeNode::Leaf {
            class: 0,
            samples: 1,
        }),
    };
    let rows: Vec<Vec<f64>> = (0..16)
        .map(|m| (0..4).map(|j| ((m >> j) & 1) as f64).collect())
        .collect();
    let x = ColumnMajorMatrix::from_rows(&rows).expect("valid bench matrix");
    let t = Instant::now();
    let mut last = Vec::new();
    for _ in 0..100 {
        last = extract_axp_deletion(&tree, &x, 3, true).features;
    }
    println!("AXp extraction: {:?} last={last:?}", t.elapsed());
}
