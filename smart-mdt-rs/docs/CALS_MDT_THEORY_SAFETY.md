# CALS-MDT theory safety

## Why per-predicate certification is insufficient

A Horn predicate, a 2-CNF predicate, and an affine equation can each have a
polynomial-time satisfiability backend. Their arbitrary conjunction on one
root-to-leaf path does not necessarily remain in any one of those tractable
languages. Therefore, “every node is individually certified” is weaker than
“the path has one certified backend.”

## PathTheoryState invariant

Search begins at `Uncommitted`. Unary predicates are compatible with every
state and do not change it. From `Uncommitted`, Horn commits to `Horn`, AntiHorn
to `AntiHorn`, Square2CNF to `TwoSat`, and Boolean Affine to `AffineGf2`.
Committed states admit only Unary plus their matching family.

`candidate_is_compatible` is called before scoring or beam insertion.
`next_theory_state` is fallible and produces each child state independently.
Final trees are recursively validated from `Uncommitted`. Thus sibling branches
may use different backends, while one path cannot mix incompatible theories.

## Cache safety

A transposition key includes exact row-mask words, depth remaining, node budget,
theory state, and full search-configuration identity. Equality verifies the
complete key; a hash collision cannot reuse a different subproblem. Cached
subtrees store `path_certified`, and theorem-mode reuse rejects a false value.
Caches live inside one training invocation, preventing cross-dataset reuse.

## Pruning safety

Pruning only replaces an internal subtree with an existing grow-majority leaf.
It removes predicates and cannot add a new conjunction to any path. The pruned
tree is nevertheless recursively validated before theorem reporting.

## Boolean affine boundary

Certified Affine means one GF(2) equation over Boolean feature columns. The
generator checks the entire training-domain column metadata, reporting uses the
same predicate-scope Boolean guard, and explanations require `Gf2Gaussian`.
`EmpiricalAffine` and the `Affine` empirical backend are not aliases for this
certified case and are excluded from theorem rows.

## Theorem reporting boundary

A CALS row enters `theorem_certified_results.csv` only if all conditions hold:

- recursive path certification succeeds and violation count is zero;
- all selected predicate families/backends are allowed;
- every affine scope passes the Boolean guard;
- AXp/tree theorem metadata remains certified;
- no empirical fallback was used;
- no incompatible cached subtree was reused;
- the declared method uses `SmartCertified` with `PathCertified` metadata;
- every reported path backend is StructuralHorn, StructuralAntiHorn, TwoSat, or
  Gf2Gaussian.

Otherwise the row is written outside the theorem table with an explicit
rejection reason. Certification cannot be recovered by a higher numerical
score.

