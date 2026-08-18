# spnn-shortest-path

Public supporting repository for the paper on **stochastic spiking neural
networks for shortest-path computation** — the code, data, and figures needed
to reproduce the results.

This lineage traces to the network construction in Engler (2018),
*Stochastic Neural Algorithms*, Ch. 4, rebuilt for scale: the intra-cluster
alpha weights are separable, so the dense per-timestep spmv collapses to two
scalar reductions per winner-take-all component, dropping cost and memory from
`O(N d^2)` to `O(|E|)`.

## Status

Early scaffolding. Contents are being built out toward a self-contained,
reproducible release accompanying the paper.

## Planned layout

| Path | Purpose |
|---|---|
| `code/` | Reference implementation + experiment drivers |
| `data/` | Input graphs and derived datasets |
| `figures/` | Paper figures and the scripts that generate them |
| `paper/` | Manuscript sources / preprint |

## License

To be finalized before public release.
