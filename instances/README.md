# Instances

## The reference instance

Every document in this project refers to "the reference instance." It is, as a
single callable expression:

```python
from spnn import random_graph, compile_network, Params

n, edges, w = random_graph(60, 0.05, seed=19)     # G(60, 0.05), weights U(0, 1]
net = compile_network(n, edges, w, 0, n - 1, Params(B=2 * float(w.max())))
```

with `B = 2·w_max` (`w_max ≈ 0.993186`, so `B ≈ 1.986372`).

| quantity | value | status |
|---|---|---|
| `|E|` | **142** | ✅ verified against `spnn/graphs.py` |
| optimum hop count `L*` | **8** | ✅ verified (`dijkstra`, opt cost ≈ `0.548769`) |
| `Ẽ(P)` (normalized path energy) | **0.276267** | ✅ verified — `experiments/reference_counts.py` |
| required `c` (path/scatter crossing) | **3.699428677298893** | ✅ verified — `experiments/reference_counts.py` |

All four are reproduced exactly. `Ẽ(P)` and the crossing `c` come from
`python -m experiments 7` (`results/counts/reference.json`) — pure combinatorics
(exact path DFS + the matching-polynomial counter), no compiler or sampler
needed; the CNT-PATH (1754) and CNT-SCAT (3.352349e19) anchors match too.

### Generator guard (do not skip)

The reference numbers above are tied to the exact generator in `spnn/graphs.py`
(weights floored at `1e-3` via `np.maximum(w, 1e-3)`). Because the data deposit
ships **seeds plus the generator**, not serialized graphs, any change to the
generator that moved the graph produced at a given seed would silently invalidate
every result.

That is guarded by a regression that runs in CI (`verify/check_instance.py`, via
`python -m verify`): it asserts `random_graph(60, 0.05, seed=19)` yields
**|E| = 142** with an **8-hop optimum**, and fails the build loudly if the
generator ever shifts the reference instance.

## Conventions

- **`L ≥ 3` by terminal selection** — source/target are chosen so the optimum is
  at least 3 hops (shorter optima are degenerate for the construction).
- **`MIN_EDGE_WEIGHT = 1e-3`** — the generator floors weights at `1e-3`
  (`np.maximum(w, 1e-3)`); no zero-weight *graph* edges.
- **Auxiliary S/T edges are exactly `0.0`** — the Ch. 4 step-1 augmentation adds
  two new vertices joined to source/target by zero-weight edges (distinct from the
  floored graph weights above).
- **Weights `U(0, 1]`** for the main arm; **`U[1, 2]`** for the weight-convention
  rider arm.

## Ship seeds, not serialized graphs

The Data availability statement promises the graphs are *reconstructible exactly
rather than only in distribution*. So this directory ships **seeds + the
generator**, never pickled/serialized graph objects. `manifest.json` (family ×
size × seed grid) and `population.csv` (all rows, censored ones flagged not
dropped) are added alongside this file.
