# Theory boundary for CGS-MDT

A multivariate decision tree (MDT) is a decision tree whose internal node asks a constraint over one or more features. An `L-DT` restricts all internal constraints to a language `L`.

Because each binary node uses a true branch `C` and a false branch `¬C`, theorem-certified modes require a complement representation accepted by the explanation engine. This crate stores the language family, complement representation, certificate type and backend with each candidate.

A subset of features is a weak AXp for instance `v` and prediction `c` when every completion that agrees with `v` on those features still predicts `c`. A subset-minimal AXp is extracted by deletion: start with all features and remove a feature exactly when the remaining set is still a weak AXp.

Weak AXp checking is implemented as path blocking: every opposite-class leaf path must be incompatible with the selected partial assignment. Certified incompatibility is limited to polynomial fragments implemented here: structural Horn, structural AntiHorn and 2-SAT.

Certified result tables may contain only Unary, Horn, AntiHorn and Square2CNF rows with certified backends. Affine, empirical mixed, tuned and fallback methods are excluded. The learner is heuristic and not globally optimal, and this implementation is not formally verified.
