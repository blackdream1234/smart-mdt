# Algorithm design: Certificate-Guided Smart MDT

CGS-MDT is certificate-first: candidate splits carry logical family, complement encoding, certificate type, backend, and theorem-mode admissibility before scoring.

The split objective is

`gain - lambda_size * complexity - lambda_axp * explanation_risk - lambda_time * cost + lambda_cert * certificate_bonus`.

The current implementation provides certified unary thresholds, Horn clauses, AntiHorn clauses and Square2CNF predicates. Beam caps rank unary literals first and combine top candidates for fast transparent search. Beam search affects training optimality only; explanation metadata is computed from the learned tree.
