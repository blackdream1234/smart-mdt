# CALS-MDT algorithm

## Problem and safety objective

CALS-MDT learns a binary multivariate decision tree from a labelled training
matrix. It optimizes predictive fit, tree compactness, explanation length, and
fit cost while enforcing theorem certification as an admissibility condition.
Certification is never a numerical bonus.

The certified split families are Unary, Horn, AntiHorn, Square2CNF, and Boolean
Affine. Affine candidates are generated only from columns whose complete
training domain is Boolean and use the `Gf2Gaussian` explanation backend.

## Path theory

Every recursive node carries `PathTheoryState`: `Uncommitted`, `Horn`,
`AntiHorn`, `TwoSat`, or `AffineGf2`. Unary predicates preserve the state. A
non-unary predicate commits an uncommitted path, after which only Unary and the
committed family are admissible. Child states are copied independently, so two
branches may commit to different tractable languages. See
`CALS_MDT_THEORY_SAFETY.md` for the formal boundary.

## Incremental bitset training

One `TrainingContext` owns an immutable root `Dataset`, class masks, Boolean
column masks, feature domains, predicate masks, and bounded caches. A node is a
`NodeView` containing a row bitset, depth, and path theory. Splitting uses
bitwise intersection and difference; class counts use intersection popcounts.
Recursive dataset copies and repeated full-row predicate scans are avoided.

For `n` rows, a mask operation costs `O(n / 64)` machine-word operations.
Candidate construction still depends on feature domains and family arity; the
bitsets reduce the cost of evaluating each completed predicate.

## Candidate generation and scoring

Only compatible certified families are generated. Completed, non-degenerate
candidates receive an auditable score containing information gain, gain ratio,
balance, literal/family complexity, fragmentation, estimated subtree cost, and
instability. `SparseCertified` is accuracy-first with small deterministic
compactness penalties. Ties use final score, information gain, literal count,
family order, and canonical predicate key.

Adaptive language allocation first gives every compatible family a deterministic
pilot. It then assigns an exact retained-candidate budget using pilot score,
gain, score per literal, generation cost, duplicate masks, and bounded-search
diagnostics. Minimum quotas preserve family diversity.

## Bounded branch-and-bound

The current safe branch-and-bound operates on a completed bounded candidate
frontier. Each frontier state's upper bound equals its exact completed score;
states are pruned only when that bound is strictly below the current kth score.
Small exact tests prove equality with bounded exhaustive top-k selection.

This is not construction-time branch-and-bound over every possible literal
extension. Generators without a verified construction bound retain bounded
exhaustive generation, and limit or numerical uncertainty triggers exhaustive
fallback. No claim of globally optimal candidate construction is made.

## Memoization

Per-fit bounded caches store node statistics, predicate masks, candidate pools,
best greedy subtrees, and lookahead objectives. `SearchStateKey` contains the
complete row-mask words (not only a hash), remaining depth, node budget, path
theory, and full score/candidate configuration identity. Cached theorem-mode
subtrees are checked for path certification before reuse.

## Tree search

Three strategies are supported:

- `Greedy`: recursively selects the best local candidate.
- `SparseLookahead`: retains a partial-tree beam near the root, then finishes
  deterministically with width one.
- `GlobalBeam`: retains a bounded set of competing partial trees.

Partial trees hold frontier leaf row masks and independent path states. The
lower bound includes only error already unavoidable at completed leaves and
current complexity; future penalty is zero unless proven. A complete greedy tree
is the anytime incumbent. On timeout or expansion exhaustion, open frontier
leaves are already valid majority leaves. Global beam is heuristic anytime
search, not exact optimal tree search.

## Parallel evaluation

Rayon evaluates independent compatible-family candidate batches. Each fit may
use a bounded custom pool. Results are merged and sorted by the same total order
as serial execution. Shared caches use separate locks, mutable trees are not
shared, and nested unbounded parallelism is avoided. Tests compare one, two,
four, and default thread counts.

## Pruning

When enabled, the input training partition is deterministically stratified into
grow and pruning-validation subsets. The external benchmark test partition is
not available to the learner. Bottom-up pruning compares each subtree with its
stored grow-majority leaf under epsilon-Pareto, cost-complexity, or classic CART
leaf-penalty selection. The original tree is not mutated and the selected tree
is path-validated again.

## AXp shortlist reranking

Only an already certified shortlist is reranked. Each candidate becomes a
majority-leaf stump and certified AXps are computed for a deterministic sample
of current training-node rows. Mean/max AXp penalties are configurable. Timeout
retains the original score and records the fallback; AXp reranking is disabled
in the default thesis profile.

## Determinism and fallbacks

Without a time budget, candidate order, family budgets, validation splits,
frontier selection, cache eviction, and parallel result merging are
deterministic. Safe fallbacks include exhaustive candidate selection, serial
evaluation if a thread pool cannot be created, no pruning when an internal
validation split is impossible, majority completion on search limits, and
original-score retention on AXp timeout.

