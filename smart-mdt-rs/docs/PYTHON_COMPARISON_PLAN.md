# Python comparison plan

A fair comparison with the older Python implementation must use the same:

- datasets and preprocessing;
- train/validation/test splits;
- maximum depth;
- minimum samples per leaf and split;
- split families enabled at each node;
- random seeds;
- theorem-certified versus empirical result separation.

Metrics to collect:

- accuracy mean/std;
- tree nodes, leaves and reached depth;
- average predicate arity and literal count;
- mean AXp length;
- training time;
- prediction time;
- explanation time;
- theorem metadata and backend.

No speedup claim should be made until both implementations are run under the same protocol on the same machine in release/optimized modes.
