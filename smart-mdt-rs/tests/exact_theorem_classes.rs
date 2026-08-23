use smart_mdt_rs::logic::{
    classify_square_2cnf, classify_star_nested_antihorn, classify_star_nested_horn, is_square_2cnf,
    is_star_nested_antihorn, is_star_nested_horn, square_formula, BooleanFormula, BooleanLiteral,
    DomainRegime, Literal, Predicate, Square2CnfForm, StructuralCheck, TheoremSource,
    ThresholdAtom, ThresholdOp,
};

fn p(feature: u32) -> BooleanLiteral {
    BooleanLiteral::new(feature, true)
}

fn n(feature: u32) -> BooleanLiteral {
    BooleanLiteral::new(feature, false)
}

fn threshold(feature: u32, positive: bool) -> Literal {
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

fn assert_exact_complement(
    formula: &BooleanFormula,
    complement: &BooleanFormula,
    variables: usize,
) {
    for mask in 0..(1usize << variables) {
        let assignment = (0..variables)
            .map(|bit| ((mask >> bit) & 1) == 1)
            .collect::<Vec<_>>();
        assert_eq!(formula.eval(&assignment), !complement.eval(&assignment));
    }
}

#[test]
fn required_star_nested_horn_examples_and_boundary() {
    let valid = BooleanFormula::new([vec![n(0)], vec![n(0), n(1)], vec![n(0), n(1), p(2)]]);
    assert!(is_star_nested_horn(&valid));

    let incomparable = BooleanFormula::new([vec![n(0), p(1)], vec![n(2), p(3)]]);
    assert!(!is_star_nested_horn(&incomparable));
    let predicate = Predicate::HornClause(vec![threshold(0, false), threshold(1, true)]);
    let certificate = predicate.certificate(true).theorem_certificate.unwrap();
    assert_eq!(certificate.domain_regime, DomainRegime::Boolean);
    assert_eq!(certificate.theorem_id, TheoremSource::Theorem3);
    assert_eq!(
        certificate.structural_check,
        StructuralCheck::StarNestedHorn
    );
}

#[test]
fn generated_star_nested_chains_preserve_complement_closure() {
    // Deterministically cover many nested clause combinations without an
    // external random-number dependency.
    let mut state = 0x9e37_79b9_u64;
    for _case in 0..128 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let chain_len = 1 + (state as usize % 4);
        let mut clauses = Vec::new();
        for positive in 0..3u32 {
            state ^= state.rotate_left(17);
            let level = state as usize % (chain_len + 1);
            let mut clause = (0..level as u32).map(n).collect::<Vec<_>>();
            clause.push(p(4 + positive));
            clauses.push(clause);
        }
        if state & 1 == 1 {
            clauses.push((0..chain_len as u32).map(n).collect());
        }
        let formula = BooleanFormula::new(clauses);
        let witness = classify_star_nested_horn(&formula).unwrap();
        let complement = witness.complement().unwrap().formula();
        assert!(is_star_nested_horn(&complement));
        assert_exact_complement(&formula, &complement, 7);

        let anti = formula.polarity_dual();
        let anti_witness = classify_star_nested_antihorn(&anti).unwrap();
        let anti_complement = anti_witness.complement().unwrap().formula();
        assert!(is_star_nested_antihorn(&anti_complement));
        assert_exact_complement(&anti, &anti_complement, 7);
    }
}

#[test]
fn all_three_square_forms_and_duals_partition_the_domain() {
    let signed = [p(0), n(1), p(2), n(3)];
    for form in [
        Square2CnfForm::FormI,
        Square2CnfForm::FormII,
        Square2CnfForm::FormIII,
    ] {
        let formula = square_formula(form, signed);
        let witness = classify_square_2cnf(&formula).unwrap();
        let complement = witness.complement().unwrap().formula();
        assert!(is_square_2cnf(&complement));
        assert_exact_complement(&formula, &complement, 4);
    }
}

#[test]
fn two_sat_solvability_does_not_certify_an_arbitrary_node_relation() {
    let outside = BooleanFormula::new([vec![p(0), p(1)], vec![p(2), p(3)], vec![p(4), p(5)]]);
    assert!(!is_square_2cnf(&outside));
}

#[test]
fn constants_are_explicit_members_of_affine_and_square_classes() {
    for formula in [BooleanFormula::complete(), BooleanFormula::empty()] {
        let square = classify_square_2cnf(&formula).unwrap();
        assert_exact_complement(&formula, &square.complement().unwrap().formula(), 2);
    }

    // x xor x = 0 and x xor x = 1 normalize to the complete and empty
    // single-equation relations, respectively.
    for rhs in [false, true] {
        let predicate = Predicate::Affine {
            literals: vec![threshold(0, true), threshold(0, true)],
            rhs,
        };
        let certificate = predicate.certificate(true);
        assert!(certificate.theorem_certified);
        assert_eq!(
            certificate.theorem_certificate.unwrap().structural_check,
            StructuralCheck::SingleGf2Equation
        );
    }
}
