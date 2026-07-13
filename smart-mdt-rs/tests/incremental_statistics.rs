use smart_mdt_rs::{
    data::{class_counts, predicate_mask, BitSet, ColumnMajorMatrix, Dataset},
    logic::{Literal, PathTheoryState, Predicate, ThresholdAtom, ThresholdOp},
    search::{
        affine::generate_affine, antihorn::generate_antihorn, entropy, gini, horn::generate_horn,
        information_gain, square2cnf::generate_square2cnf, unary::generate_unary, SplitCandidate,
    },
    tree::{LanguagePolicy, NodeView, TrainingContext},
};
use std::sync::Arc;

fn literal(feature: u32, positive: bool) -> Literal {
    Literal {
        atom: ThresholdAtom {
            feature,
            threshold_id: 0,
            threshold: 0.5,
            op: ThresholdOp::GreaterEqual,
        },
        positive,
    }
}

fn predicates() -> Vec<Predicate> {
    vec![
        Predicate::Unary(literal(0, true)),
        Predicate::HornClause(vec![literal(0, false), literal(1, true)]),
        Predicate::AntiHornClause(vec![literal(1, true), literal(2, false)]),
        Predicate::Square2Cnf {
            a: literal(0, true),
            b: literal(1, false),
            c: literal(2, true),
            d: literal(3, false),
        },
        Predicate::Affine {
            literals: vec![literal(0, true), literal(2, true), literal(3, true)],
            rhs: true,
        },
    ]
}

fn random_dataset(seed: u64, rows: usize, features: usize) -> Dataset {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        state
    };
    let matrix: Vec<Vec<f64>> = (0..rows)
        .map(|_| (0..features).map(|_| ((next() >> 63) & 1) as f64).collect())
        .collect();
    let labels = (0..rows).map(|_| ((next() >> 63) & 1) as u32).collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&matrix).unwrap(), labels).unwrap()
}

fn random_rows(seed: u64, len: usize) -> BitSet {
    let mut state = seed;
    let mut rows = BitSet::zeros(len);
    for index in 0..len {
        state = state
            .wrapping_mul(2_862_933_555_777_941_757)
            .wrapping_add(3_037_000_493);
        rows.set(index, (state >> 63) == 1);
    }
    if rows.count_ones() == 0 {
        rows.set(0, true);
    }
    rows
}

#[test]
fn incremental_statistics_match_naive_random_boolean_data() {
    for seed in 0..16 {
        let dataset = random_dataset(seed + 1, 19, 4);
        let context = TrainingContext::new(Arc::new(dataset.clone()));
        let node = NodeView {
            rows: random_rows(seed + 91, dataset.labels.len()),
            depth: 2,
            theory_state: PathTheoryState::Uncommitted,
        };
        let classes = dataset.class_count().max(2);
        let naive_parent = class_counts(&dataset.labels, &node.rows, classes);

        assert_eq!(context.sample_count(&node), node.rows.count_ones());
        assert_eq!(context.class_counts(&node).unwrap(), naive_parent);
        let naive_majority = naive_parent
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| **count)
            .map_or(0, |(class, _)| class as u32);
        assert_eq!(context.majority_class(&node).unwrap(), naive_majority);
        assert_eq!(
            gini(&context.class_counts(&node).unwrap()),
            gini(&naive_parent)
        );
        assert_eq!(
            entropy(&context.class_counts(&node).unwrap()),
            entropy(&naive_parent)
        );

        for predicate in predicates() {
            let full = predicate_mask(&dataset.features, &predicate);
            let expected_true = node.rows.and(&full).unwrap();
            let expected_false = node.rows.and_not(&full).unwrap();
            let (actual_true, actual_false) = context.split_masks(&node, &predicate).unwrap();
            assert_eq!(actual_true, expected_true, "true mask for {predicate:?}");
            assert_eq!(actual_false, expected_false, "false mask for {predicate:?}");
            assert_eq!(
                context.predicate_mask(&node, &predicate).unwrap(),
                expected_true
            );

            let naive_true = class_counts(&dataset.labels, &expected_true, classes);
            let naive_false = class_counts(&dataset.labels, &expected_false, classes);
            assert_eq!(
                context.child_class_counts(&actual_true).unwrap(),
                naive_true
            );
            assert_eq!(
                context.child_class_counts(&actual_false).unwrap(),
                naive_false
            );
            assert_eq!(
                information_gain(
                    &context.class_counts(&node).unwrap(),
                    &context.child_class_counts(&actual_true).unwrap(),
                    &context.child_class_counts(&actual_false).unwrap(),
                ),
                information_gain(&naive_parent, &naive_true, &naive_false)
            );
            let naive_balance = expected_true.count_ones().min(expected_false.count_ones()) as f64
                / node.rows.count_ones() as f64;
            assert_eq!(context.balance(&actual_true, &actual_false), naive_balance);
        }
    }
}

#[test]
fn repeated_predicates_hit_the_mask_cache_and_report_avoided_work() {
    let dataset = random_dataset(42, 32, 4);
    let context = TrainingContext::new(Arc::new(dataset));
    let node = context.root_view();
    let predicate = predicates().remove(0);
    context.predicate_mask(&node, &predicate).unwrap();
    context.predicate_mask(&node, &predicate).unwrap();
    context.record_child_views();

    let diagnostics = context.diagnostics();
    assert_eq!(diagnostics.predicate_mask_cache_misses, 1);
    assert!(diagnostics.predicate_mask_cache_hits >= 1);
    assert_eq!(diagnostics.dataset_subset_allocations_avoided, 2);
    assert!(diagnostics.row_rescans_avoided >= node.rows.len());
}

fn subset(dataset: &Dataset, selected: &BitSet) -> Dataset {
    let rows: Vec<Vec<f64>> = (0..dataset.labels.len())
        .filter(|&row| selected.get(row))
        .map(|row| {
            (0..dataset.features.cols())
                .map(|feature| dataset.features.get(row, feature as u32))
                .collect()
        })
        .collect();
    let labels = (0..dataset.labels.len())
        .filter(|&row| selected.get(row))
        .map(|row| dataset.labels[row])
        .collect();
    Dataset::new(ColumnMajorMatrix::from_rows(&rows).unwrap(), labels).unwrap()
}

fn candidate_signatures(mut candidates: Vec<SplitCandidate>) -> Vec<String> {
    let mut signatures: Vec<_> = candidates
        .drain(..)
        .map(|candidate| {
            format!(
                "{:?}|{}|{}|{}|{}",
                candidate.predicate,
                candidate.score.predictive_gain.to_bits(),
                candidate.score.final_score.to_bits(),
                candidate.left_count,
                candidate.right_count
            )
        })
        .collect();
    signatures.sort();
    signatures
}

#[test]
fn masked_candidate_generation_matches_naive_materialized_subsets() {
    let dataset = random_dataset(7, 24, 4);
    let context = TrainingContext::new(Arc::new(dataset.clone()));
    let rows = random_rows(19, dataset.labels.len());
    let node = NodeView {
        rows: rows.clone(),
        depth: 1,
        theory_state: PathTheoryState::Uncommitted,
    };
    let naive = subset(&dataset, &rows);
    let beam = 6;
    for policy in [
        LanguagePolicy::UnaryOnly,
        LanguagePolicy::HornOnly,
        LanguagePolicy::AntiHornOnly,
        LanguagePolicy::Square2CnfOnly,
        LanguagePolicy::AffineOnly,
    ] {
        let expected = match policy {
            LanguagePolicy::UnaryOnly => generate_unary(&naive, 1),
            LanguagePolicy::HornOnly => generate_horn(&naive, 1, beam),
            LanguagePolicy::AntiHornOnly => generate_antihorn(&naive, 1, beam),
            LanguagePolicy::Square2CnfOnly => generate_square2cnf(&naive, 1, beam),
            LanguagePolicy::AffineOnly => generate_affine(&naive, 1, beam),
            _ => unreachable!(),
        };
        let actual = context.generate_candidates(&node, policy, 1, beam).unwrap();
        assert_eq!(
            candidate_signatures(actual),
            candidate_signatures(expected),
            "candidate mismatch for {policy:?}"
        );
    }
}
