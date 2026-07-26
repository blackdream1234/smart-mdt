# CALS-MDT search

## Candidate beam versus tree beam

`candidate_beam_width` bounds candidates retained for one leaf. It is distinct
from `tree_beam_width`, which bounds competing partial trees. The legacy
`beam_width` remains available to unchanged single-family greedy policies.

## Partial-tree state

A `PartialTreeState` contains an immutable-by-cloning partial tree, open
`FrontierLeaf` records, completed-leaf error lower bound, complexity cost,
objective lower bound, expanded-node count, and deterministic generation order.
Each frontier leaf holds its exact row mask, depth, majority class, and
`PathTheoryState`.

Frontier selection supports highest current error, most samples, and best
potential local gain. Ties prefer shallower depth and stable node identity.

## Expansion and bounds

An expansion selects one frontier leaf, generates only compatible candidates,
copies the state for each retained candidate, and gives both children the same
resulting path state as independent values. Terminal children remain majority
leaves. States are ordered by safe objective lower bound, complexity, expanded
nodes, canonical tree key, and generation order.

The lower bound counts error only for leaves that cannot be improved further.
Open leaves contribute zero future error, and no speculative future penalty is
subtracted. This is safe for beam ordering but does not make beam truncation an
exact proof procedure.

## Strategies and anytime behavior

Greedy recursively expands one candidate. Sparse lookahead uses a wider beam for
the configured early depth, then width one. Global beam keeps up to
`tree_beam_width` partial trees until completion or a depth, node, expansion, or
time budget. A complete greedy tree is retained as an incumbent when it respects
the node budget. Any unfinished state can be returned immediately because open
frontier nodes are represented by majority leaves.

Runs without time truncation are deterministic. Time-limited runs return valid,
certified trees but are not promised to be bit-for-bit reproducible across
machines.

## Exactness boundary

Bounded candidate branch-and-bound is exact relative to its generated completed
frontier. Global beam and sparse lookahead are heuristic anytime tree searches.
They do not prove the globally optimal tree for a depth/node bound.

