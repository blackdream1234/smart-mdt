# Algorithm design: Certificate-Guided Smart MDT

CGS-MDT is certificate-first: candidate splits carry logical family, complement encoding, certificate type, backend, and theorem-mode admissibility before scoring.

The split objective is

`gain - lambda_size * complexity - lambda_axp * explanation_risk - lambda_time * cost + lambda_cert * certificate_bonus`.

The implementation provides certified unary thresholds, Horn clauses, AntiHorn clauses and Square2CNF predicates. Beam caps rank unary literals first and combine top candidates for fast transparent search. Beam search affects training optimality only; explanation metadata is computed from the learned tree.

Theorem-certified paths are limited to StructuralHorn, StructuralAntiHorn and TwoSat backends. Empirical Affine, empirical mixed policies and tuned/experimental policies are present only as separated hooks and must not enter theorem-certified result tables.

Weak AXp extraction uses the deletion algorithm over weak AXp checks. For binary-domain tests, weak AXp checking enumerates all Boolean completions and compares against brute force. For broader numeric data, the current prototype checks completions represented by the supplied dataset domain; this is a tested prototype behavior, not a full proof of real-valued CSP tractability.
