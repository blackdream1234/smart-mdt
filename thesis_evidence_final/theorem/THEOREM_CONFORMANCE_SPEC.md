# Exact Theorem Conformance Specification

## Status and normative source

The normative source is Carbonnel, Cooper, Hebrard, Morales, and
Marques-Silva, *Explaining Multivariate Decision Trees: Characterising
Tractable Languages* (journal version supplied with this repository, 2025).
This document is an implementation contract, not a broader claim about
ordinary Horn-SAT, 2-SAT, or affine CSPs.

The repository's certified execution regime is **Boolean only**. All 46 frozen
benchmark datasets contain only feature values 0 and 1 after loading and
constant-column removal. Ordered finite-domain UI generalizations are described
below so that their boundary is explicit, but the current explanation engine
does not issue an `OrderedFiniteUI` certificate.

## Proposition 1: weak AXp contract

For a constraint language `L`, weak AXp checking is theorem-certified only if:

1. `L` is closed under logical complement;
2. assignment constraints are admitted in a tractable CSP super-language `C`;
3. for every opposite-class root-to-leaf path, the partial assignment is tested
   for incompatibility with the exact conjunction of branch relations; and
4. `CSP(C)` is solved by the corresponding polynomial-time backend.

The implementation checks every node relation and its complement before a path
state is admitted. It then normalizes the complete path and validates Horn,
Anti-Horn, 2CNF, or GF(2) structure before dispatch. Backend and method names
are never sufficient evidence: the legacy `CertificateMetadata::new` tuple
constructor deliberately fails closed.

The adaptive methods may use different theorem families on different paths.
They are certified through a direct Proposition-1 path argument only when each
individual path stays inside one supported CSP super-language. They are not
claimed to be a new complement-closed union language under Theorem 7.

## Boolean-domain theorem classes

For a finite Boolean language closed under complement, Theorem 7 permits a
tractability certificate exactly when every relation in the language belongs
to at least one of the following classes.

### Theorem 3: star-nested Horn

A Horn CNF is star-nested when there are sets

`empty = S0 subset S1 subset ... subset Sq`

such that every member of `Sq` is a negative literal and every clause is
either `OR(Si)` or `l OR OR(Si)` for one positive literal `l`. Equivalently,
after normalization, the negative-literal sets of all clauses form a chain
under set inclusion and every clause contains at most one positive literal.

The code:

- removes duplicate literals and clauses;
- discards tautological clauses and removes subsumed clauses;
- sorts features, literals, clauses, and negative sets deterministically;
- checks the Horn polarity bound and total inclusion order;
- represents the witness as `StarNestedHorn`;
- constructs the complement recursively using Proposition 2 and revalidates
  the result as star-nested Horn; and
- fails closed if either structural or complement validation fails.

The certified learner currently emits one two-literal Horn clause per node.
Every such Horn clause is a special case of star-nested Horn. A Horn path is a
conjunction of the true-branch clauses and exact false-branch complements; its
larger CSP super-language is Horn CNF, checked before StructuralHorn/HornSAT
dispatch. The entire path need not itself be one node relation or a
star-nested formula.

Ordinary Horn formulas with incomparable negative sets are not certified as
node relations.

### Theorem 4: star-nested Anti-Horn

Star-nested Anti-Horn is the exact literal-polarity dual: negating every
literal must produce a star-nested Horn formula. The implementation represents
the dual witness as `StarNestedAntiHorn`, checks at most one negative literal
per clause, checks the chain of positive-literal sets, uses the dual of the
Proposition-2 complement construction, and revalidates the complement.

The certified learner currently emits one two-literal Anti-Horn clause per
node. Anti-Horn paths are normalized and validated as Anti-Horn CNF before
StructuralAntiHorn/AntiHornSAT dispatch. Arbitrary Anti-Horn node formulas are
not certified merely because the backend can solve their path conjunction.

### Theorem 5: one Boolean GF(2) equation

A certified affine **node relation** is exactly one equation

`a1*x1 + ... + ar*xr = b (mod 2)`, where each coefficient is 0 or 1.

The canonical `SingleGf2Equation` representation sorts variables and cancels
duplicates modulo two. Negated or constant Boolean literals are absorbed into
the right-hand side. The degenerate equations `0 = 0` (complete) and `0 = 1`
(empty) are explicit valid relations. Complementation keeps the coefficients
and flips the right-hand side.

Every feature in the actual node scope and reference domain must be Boolean.
A path may contain several certified node equations; the path CSP is then a
system of GF(2) equations, including assignment equations, solved by exact
Gaussian elimination. This does not turn a multi-equation node representation
into a Theorem-5 relation.

The path implementation has a 128-distinct-variable representation limit and
fails closed above it.

### Theorem 6: square 2CNF

A certified Boolean square-2CNF node relation is empty, complete, or is
equivalent to one of these exact forms, with literals not necessarily distinct:

1. `(a OR b) AND (c OR d)`;
2. `(a OR b) AND (b OR c) AND (c OR d)`;
3. `(a OR b) AND (b OR c) AND (a OR d) AND (c OR d)`.

`Square2CnfRelation` records the witnessed form. Normalized formulas are
matched deterministically against these three templates. Complement maps Form
I to Form III, Form II to Form II, and Form III to Form I with the theorem's
literal permutation and polarity reversal; the resulting relation is checked
again.

The current learner emits Form I nodes only, which is a theorem-safe sublanguage.
The conjunction of all branch encodings on a path may be a general 2CNF and is
solved by 2-SAT. An arbitrary 2CNF is not thereby admitted as a node relation.

## Ordered finite-domain UI generalizations

For an ordered finite domain, a positive UI literal is `x_i >= a` and its
negative literal is `x_i < a`, where `a` is non-minimal. The paper establishes:

- generalized square 2CNF plus unary constraints (Proposition 5);
- generalized star-nested Horn and Anti-Horn plus unary constraints
  (Proposition 6); and
- the restricted UI-generalization dichotomy (Theorem 8), containing only
  square 2CNF, star-nested Horn, and star-nested Anti-Horn Boolean bases.

UI-generalized affine relations of arity at least three on domains of size at
least three are NP-hard in the stated setting (Lemma 11) and are excluded by
Theorem 8. Binary UI-affine relations reduce to generalized square 2CNF and
would have to be certified through that route; unary relations use the unary
baseline.

This repository does not implement the required finite-domain median,
generalized arc-consistency, or exact UI path backends. Therefore all direct
ordered-domain theorem requests receive `domain_regime = Unsupported`, not a
Boolean or affine certificate.

## Structured certificate contract

Every new theorem-certified result is backed by `TheoremCertificate`:

| Field | Meaning |
|---|---|
| `domain_regime` | `Boolean`, `OrderedFiniteUI`, or `Unsupported` |
| `language_family` | Actual fixed family or path-certified adaptive policy |
| `theorem_id` | `Theorem3`, `Theorem4`, `Theorem5`, `Theorem6`, `Proposition1`, or `UnaryBaseline` in current certified output |
| `structural_check` | Exact normalized witness, never a backend alias |
| `complement_check` | Exact theorem-preserving complement construction used |
| `backend` | Solver selected only after the preceding checks pass |
| `assumptions_supported` | Whether partial assignments are admitted by the checked path solver |
| `path_check` | Validated Horn CNF, Anti-Horn CNF, 2CNF, GF(2) system, or per-path theory |

`theorem_certified` is the conjunction of theorem mode, a valid structured
certificate, an actual Boolean reference domain, exact per-node certificates,
path compatibility, backend validation, and successful full-held-out-row AXp
sufficiency/minimality checks.

## Theorem-to-code map

| Contract | Implementation |
|---|---|
| Normalized Boolean formulas and exact class recognizers | `smart-mdt-rs/src/logic/theorem.rs` |
| Structured certificate and fail-closed tuple validation | `smart-mdt-rs/src/logic/certificate.rs` |
| Predicate-level structural/complement proof | `smart-mdt-rs/src/logic/predicate.rs` |
| Exact complements used on false branches | `smart-mdt-rs/src/logic/complement.rs` |
| Path-family compatibility | `smart-mdt-rs/src/logic/path_theory.rs` |
| Opposite-path normalization and solver dispatch | `smart-mdt-rs/src/explain/path_blocking.rs` |
| Boolean-domain and assignment guard | `smart-mdt-rs/src/explain/weak_axp.rs` |
| GF(2) Gaussian elimination | `smart-mdt-rs/src/sat/affine_gf2.rs` |
| Certified generators | `smart-mdt-rs/src/search/{horn,antihorn,square2cnf,affine}.rs` |
| Exact theorem table boundary | `smart-mdt-rs/src/eval/{benchmark,report}.rs` |
| Exhaustive/small-domain regressions | `smart-mdt-rs/tests/exact_theorem_classes.rs` and existing `audit_*` tests |

## Explicit non-certification boundary

The following are not theorem-certified:

- generic Horn or Anti-Horn node formulas without a star-nested witness;
- arbitrary 2CNF node formulas outside Theorem 6;
- arbitrary affine CNF or a multi-equation node relation;
- affine predicates on any non-Boolean feature scope;
- arity-three-or-higher UI-affine predicates on non-Boolean ordered domains;
- empirical affine and empirically mixed policies;
- a tree containing an incompatible mixture on one root-to-leaf path;
- a certificate inferred from a method, family, backend, or path-state name;
- any result lacking a verified complement, supported assumptions, exact path
  check, or full-row AXp verification.
