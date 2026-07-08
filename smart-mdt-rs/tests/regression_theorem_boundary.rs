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
        git_sha: "x".into(),
        config: "{}".into(),
    };
    assert!(theorem_table_filter(&good));
    let bad = ResultRow {
        method: "affine".into(),
        language_family: LanguageFamily::EmpiricalAffine,
        backend: Backend::Affine,
        ..good.clone()
    };
    assert!(!theorem_table_filter(&bad));
    let bad2 = ResultRow {
        method: "bestpn".into(),
        ..good
    };
    assert!(!theorem_table_filter(&bad2));
}
