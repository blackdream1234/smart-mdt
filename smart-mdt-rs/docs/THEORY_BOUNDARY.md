# Theory boundary for CGS-MDT

A multivariate decision tree (MDT) is a decision tree whose internal node asks a constraint over one or more features. An `L-DT` restricts all internal constraints to a language `L`.

Because each binary node uses a true branch `C` and a false branch `¬C`, theorem-certified modes require a complement representation accepted by the explanation engine. This crate stores the language family, complement representation, certificate type and backend with each candidate.

A subset of features is a weak AXp for instance `v` and prediction `c` when every completion that agrees with `v` on those features still predicts `c`. A subset-minimal AXp is extracted by deletion: start with all features and remove a feature exactly when the remaining set is still a weak AXp.

The implementation boundary follows the theorem stated in the seminar: if weak AXp checking for a language is polynomial, AXp extraction follows by the deletion algorithm; if a language is closed under complement and `CSP(L ∪ Asst)` is polynomial, weak AXp checking is polynomial for that language.

Certified incompatibility is limited to polynomial fragments implemented and tested here: structural Horn, structural AntiHorn and 2-SAT. Weak-AXp checking extracts each root-to-opposite-class-leaf path, encodes the taken true/false branch constraints plus the selected feature assignments, and runs the corresponding backend. A row may report `StructuralHorn`, `StructuralAntiHorn` or `TwoSat` only when that backend was actually invoked for the path checks. Certified result tables may contain only Unary, Horn, AntiHorn and Square2CNF rows with certified backends. Affine, empirical mixed, tuned and fallback methods are excluded.

Affine/XOR predicates are handled only by an empirical bounded GF(2) Gaussian-elimination checker for Boolean scopes. True affine branches encode the original equation, false branches flip the right-hand side, and selected features become single-variable assumptions. Non-Boolean affine scopes and systems exceeding the supported variable bound are rejected. This does not make Affine theorem-certified.

The learner is greedy/heuristic and beam-limited; it is not a global optimizer. The Rust implementation is tested by unit, regression and brute-force tests, but it is not formally verified. Theorem mode does not use the old dataset-row completion fallback.
