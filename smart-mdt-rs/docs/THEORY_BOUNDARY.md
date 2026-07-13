# Theory boundary for CGS-MDT

A multivariate decision tree (MDT) is a decision tree whose internal node asks a constraint over one or more features. An `L-DT` restricts all internal constraints to a language `L`.

Because each binary node uses a true branch `C` and a false branch `¬C`, theorem-certified modes require a complement representation accepted by the explanation engine. This crate stores the language family, complement representation, certificate type and backend with each candidate.

A subset of features is a weak AXp for instance `v` and prediction `c` when every completion that agrees with `v` on those features still predicts `c`. A subset-minimal AXp is extracted by deletion: start with all features and remove a feature exactly when the remaining set is still a weak AXp.

The implementation boundary follows the theorem stated in the seminar: if weak AXp checking for a language is polynomial, AXp extraction follows by the deletion algorithm; if a language is closed under complement and `CSP(L ∪ Asst)` is polynomial, weak AXp checking is polynomial for that language.

Certified support is limited to the polynomial fragments implemented and tested here: structural Horn, structural AntiHorn, 2-SAT, and Boolean affine systems solved over GF(2). Unary predicates do not commit a path. The first non-unary predicate commits that path to Horn, AntiHorn, TwoSat, or AffineGf2, after which only unary predicates or predicates from the committed family are allowed. Children of a unary path may commit independently.

Certified result tables may contain the five single-family methods plus `smart_certified` when every path passes this state transition check. Empirical affine, empirical mixed, tuned, fallback, and path-incompatible trees are excluded.

The learner is greedy/heuristic and beam-limited; it is not a global optimizer. The Rust implementation is tested by unit, regression and brute-force tests, but it is not formally verified. The current non-binary weak-AXp path uses dataset-domain completions and must not be described as a full real-valued CSP proof.
