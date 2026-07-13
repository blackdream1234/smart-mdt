use smart_mdt_rs::{
    eval::{theorem_table_filter, ResultRow},
    logic::{Backend, LanguageFamily},
};
#[test]
fn theorem_table_excludes_forbidden() {
    let good = ResultRow {
        dataset: "d".into(),
        run: 0,
        depth: 1,
        method: "horn".into(),
        accuracy: 1.0,
        train_time: 0.0,
        predict_time: 0.0,
        tree_nodes: 1,
        leaves: 1,
        max_depth_reached: 0,
        mean_axp_length: 0.0,
        axp_time: 0.0,
        theorem_certified: true,
        language_family: LanguageFamily::Horn,
        backend: Backend::StructuralHorn,
        path_theory_state: "horn".into(),
        path_backend: "StructuralHorn".into(),
        path_certified: true,
        git_sha: "x".into(),
        config: "{}".into(),
        random_state: 1,
        n_runs: 1,
        train_test_split_protocol: "deterministic_hash_70_30_first_label".into(),
    };
    assert!(theorem_table_filter(&good));

    // Certified Boolean affine with the GF(2) backend is admitted.
    let certified_affine = ResultRow {
        method: "affine".into(),
        language_family: LanguageFamily::Affine,
        backend: Backend::Gf2Gaussian,
        ..good.clone()
    };
    assert!(theorem_table_filter(&certified_affine));

    // Empirical affine (family EmpiricalAffine, backend Affine) is excluded.
    let bad = ResultRow {
        method: "affine".into(),
        language_family: LanguageFamily::EmpiricalAffine,
        backend: Backend::Affine,
        ..good.clone()
    };
    assert!(!theorem_table_filter(&bad));

    // Affine is admitted ONLY with the GF(2) backend: a non-GF(2) backend fails.
    let affine_wrong_backend = ResultRow {
        method: "affine".into(),
        language_family: LanguageFamily::Affine,
        backend: Backend::TwoSat,
        ..good.clone()
    };
    assert!(!theorem_table_filter(&affine_wrong_backend));

    let bad2 = ResultRow {
        method: "bestpn".into(),
        ..good
    };
    assert!(!theorem_table_filter(&bad2));
}
