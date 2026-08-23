# Controlled theorem-valid targets

- `unary`: `x0`
- `horn_simple`: `not x0 OR x1`
- `horn_chain`: `x3 AND (not x0 OR x4) AND (not x0 OR not x1 OR x5)`
- `antihorn_chain`: exact polarity dual of `horn_chain`
- `square_form_i`: `(x0 OR x1) AND (not x2 OR x3)`
- `square_form_ii`: `(x0 OR x1) AND (x1 OR not x2) AND (not x2 OR x3)`
- `square_form_iii`: `(x0 OR x1) AND (x1 OR not x2) AND (x0 OR x3) AND (not x2 OR x3)`
- `affine_xor3`: `x0 XOR x1 XOR x2`

Each target is emitted at 0%, 5%, and 10% deterministic label noise. The
complete six-variable Boolean domain is repeated eight times. Labels are
generated before any train/test split; no test label participates in candidate
construction.
