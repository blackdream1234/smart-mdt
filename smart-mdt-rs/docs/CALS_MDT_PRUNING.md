# CALS-MDT pruning

## Data protocol

Pruning never consumes benchmark test labels. When enabled, the learner
deterministically hashes and stratifies the supplied training partition into a
grow subset and a pruning-validation subset. Each class retains a grow example
when possible. The selected validation indices are included in diagnostics.

If there are fewer than four rows or the fraction is invalid, the learner safely
falls back to training on the complete supplied training partition without
pruning.

## Objectives

Cost-complexity mode minimizes validation error plus configurable internal-node,
leaf, literal, and estimated path-explanation penalties. CART mode uses
validation error plus an alpha leaf penalty. The recommended epsilon-Pareto mode
permits a majority-leaf replacement only when its local validation accuracy is
within `accuracy_epsilon` (plus an optional one-standard-error allowance), then
prefers the strictly smaller representation.

## Bottom-up algorithm

Validation rows are routed to each node. Children are pruned first, then the
resulting subtree is compared with a leaf using the internal node's grow-majority
class. Each accepted replacement is recorded as a nested pruning-path step. The
implementation returns a new tree and leaves the grown tree unchanged.

Pruning cannot increase nodes or literals. It cannot introduce an incompatible
predicate because it only removes them. Final path certification is still
required and recorded.

## Metrics

The benchmark records grow/validation sizes, nodes/leaves/literals before and
after, validation accuracy before and after, estimated AXp path length before
and after, pruning time, path entries, and post-prune certification.

