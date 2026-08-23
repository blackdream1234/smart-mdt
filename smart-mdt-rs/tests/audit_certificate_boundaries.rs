use smart_mdt_rs::{
    logic::{
        Backend, CertificateMetadata, LanguageFamily, Literal, PathCertificate, Predicate,
        ThresholdAtom, ThresholdOp,
    },
    sat::{antihorn_sat, brute_force_sat, horn_sat},
    tree::{tree_is_certified, TreeNode},
};

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

fn less_than_literal(feature: u32, positive: bool) -> Literal {
    Literal {
        atom: ThresholdAtom {
            feature,
            threshold_id: 0,
            threshold: 0.5,
            op: ThresholdOp::LessThan,
        },
        positive,
    }
}

fn stump(predicate: Predicate) -> TreeNode {
    TreeNode::Internal {
        predicate,
        left: Box::new(TreeNode::Leaf {
            class: 0,
            samples: 1,
        }),
        right: Box::new(TreeNode::Leaf {
            class: 1,
            samples: 1,
        }),
        majority_class: 0,
    }
}

#[test]
fn malformed_structural_predicates_fail_closed_and_affine_normalizes() {
    let malformed_horn = Predicate::HornClause(vec![literal(0, true), literal(1, true)]);
    assert!(!malformed_horn.certificate(true).theorem_certified);
    assert!(!tree_is_certified(&stump(malformed_horn)));

    let malformed_antihorn = Predicate::AntiHornClause(vec![literal(0, false), literal(1, false)]);
    assert!(!malformed_antihorn.certificate(true).theorem_certified);
    assert!(!tree_is_certified(&stump(malformed_antihorn)));

    // Input ordering is not a theorem violation: the exact representation
    // sorts variables and cancels duplicates modulo two.
    let normalized_affine = Predicate::Affine {
        literals: vec![literal(1, true), literal(0, true), literal(1, true)],
        rhs: false,
    };
    assert!(normalized_affine.certificate(true).theorem_certified);
    assert!(tree_is_certified(&stump(normalized_affine)));
}

#[test]
fn mismatched_certificate_metadata_triples_are_not_theorem_certified() {
    let mismatched = CertificateMetadata::new(
        true,
        LanguageFamily::Horn,
        Backend::Gf2Gaussian,
        PathCertificate::AffineGf2,
    );
    assert!(!mismatched.theorem_certified);

    let names_only = CertificateMetadata::new(
        true,
        LanguageFamily::Horn,
        Backend::StructuralHorn,
        PathCertificate::HornCnf,
    );
    assert!(!names_only.theorem_certified);
    assert!(names_only.theorem_certificate.is_none());

    let structurally_checked =
        Predicate::HornClause(vec![literal(0, false), literal(1, true)]).certificate(true);
    assert!(structurally_checked.theorem_certified);
    assert!(structurally_checked.theorem_certificate.is_some());

    let matching_backend_outside_theorem_mode = CertificateMetadata::new(
        false,
        LanguageFamily::Horn,
        Backend::StructuralHorn,
        PathCertificate::HornCnf,
    );
    assert!(!matching_backend_outside_theorem_mode.theorem_certified);
}

#[test]
fn structural_clause_shape_uses_effective_boolean_polarity() {
    // On {0,1}, !(x < 0.5) is the positive Boolean variable x. Two such
    // literals therefore do not form a Horn clause even though both wrapper
    // polarities are `false`.
    let effectively_non_horn = Predicate::HornClause(vec![
        less_than_literal(0, false),
        less_than_literal(1, false),
    ]);
    assert!(!effectively_non_horn.certificate(true).theorem_certified);
    assert!(!tree_is_certified(&stump(effectively_non_horn)));

    // Conversely, x < 0.5 is the negative Boolean literal !x, so duplicates
    // and multiple distinct negative literals remain Horn.
    let effectively_horn = Predicate::HornClause(vec![
        less_than_literal(0, true),
        less_than_literal(1, true),
        less_than_literal(1, true),
    ]);
    assert!(effectively_horn.certificate(true).theorem_certified);
    assert!(tree_is_certified(&stump(effectively_horn)));

    let effectively_non_antihorn =
        Predicate::AntiHornClause(vec![less_than_literal(0, true), less_than_literal(1, true)]);
    assert!(!effectively_non_antihorn.certificate(true).theorem_certified);

    // A Boolean tautology is valid in both fragments after normalization.
    let tautology = Predicate::HornClause(vec![
        less_than_literal(0, true),
        less_than_literal(0, false),
        literal(1, true),
        literal(2, true),
    ]);
    assert!(tautology.certificate(true).theorem_certified);
}

fn affine_range(start: u32, count: u32) -> Predicate {
    Predicate::Affine {
        literals: (start..start + count)
            .map(|feature| literal(feature, true))
            .collect(),
        rhs: false,
    }
}

fn internal(predicate: Predicate, left: TreeNode, right: TreeNode) -> TreeNode {
    TreeNode::Internal {
        predicate,
        left: Box::new(left),
        right: Box::new(right),
        majority_class: 0,
    }
}

fn leaf(class: u32) -> TreeNode {
    TreeNode::Leaf { class, samples: 1 }
}

#[test]
fn gf2_certification_enforces_per_predicate_and_path_variable_limits() {
    let oversized_predicate = affine_range(0, 129);
    assert!(!oversized_predicate.certificate(true).theorem_certified);
    assert!(!tree_is_certified(&stump(oversized_predicate)));

    let cumulative_affine = internal(
        affine_range(0, 64),
        internal(affine_range(64, 65), leaf(0), leaf(1)),
        leaf(0),
    );
    assert!(!tree_is_certified(&cumulative_affine));

    // Unary features before and after the first affine commitment are GF(2)
    // variables too, even though unary does not itself commit the path.
    let preceding_unary = internal(
        Predicate::Unary(literal(128, true)),
        internal(affine_range(0, 128), leaf(0), leaf(1)),
        leaf(0),
    );
    assert!(!tree_is_certified(&preceding_unary));

    let following_unary = internal(
        affine_range(0, 128),
        internal(Predicate::Unary(literal(128, true)), leaf(0), leaf(1)),
        leaf(0),
    );
    assert!(!tree_is_certified(&following_unary));

    assert!(tree_is_certified(&stump(affine_range(0, 128))));
}

#[test]
fn duplicate_literals_do_not_change_horn_or_antihorn_satisfiability() {
    let horn = vec![vec![1, 1]];
    assert_eq!(horn_sat(1, &horn), brute_force_sat(1, &horn));
    assert!(horn_sat(1, &horn));

    let antihorn = vec![vec![-1, -1]];
    assert_eq!(antihorn_sat(1, &antihorn), brute_force_sat(1, &antihorn));
    assert!(antihorn_sat(1, &antihorn));

    let tautological_horn = vec![vec![1, -1, 2]];
    assert_eq!(
        horn_sat(2, &tautological_horn),
        brute_force_sat(2, &tautological_horn)
    );
    assert!(horn_sat(2, &tautological_horn));
}

#[test]
fn malformed_sat_variable_ids_are_rejected_without_panicking() {
    for malformed in [vec![vec![0]], vec![vec![2]], vec![vec![i32::MIN]]] {
        assert!(!horn_sat(1, &malformed));
        assert!(!antihorn_sat(1, &malformed));
        assert!(!smart_mdt_rs::sat::two_sat(1, &malformed));
        assert!(!brute_force_sat(1, &malformed));
    }
}
