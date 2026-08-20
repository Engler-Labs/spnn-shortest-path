"""Experiment 7 -- reference-instance counts and the discrimination crossing c.

Claims: CNT-PATH, CNT-SCAT, CNT-CROSS (CLAIMS.tsv, section V-B).
Output: results/counts/reference.json

The anchor of Part B, reduced to the reference instance and extracted from an
internal reference harness.  Everything here is EXACT and
combinatorial -- exhaustive depth-limited path DFS + the validated frontier-DP
matching counter -- so there is no sampler and no compiler in the loop:

  * CNT-PATH  = simple s->t paths at the matched depth L (count_paths_by_length).
  * CNT-SCAT  = matched-size scatter microstates on the AUGMENTED graph
                (scatter_counts, ``micro_aux`` at matched size L+2).
  * CNT-CROSS = the alpha-scale c at which the path sector and the scatter sector
                cross.  With the microstate convention the discrimination margin
                per unit c is ``margin_c1 = (L+1) - 2*Etilde`` and the entropy the
                margin must overcome is ``log_ratio = ln(N_scatter/N_path)``, so
                the crossing is ``c* = log_ratio / margin_c1``.

Reference instance: ``random_graph(60, 0.05, seed=19)``, source 0, target 59.
"""

from __future__ import annotations

import math

from spnn.counting import count_paths_by_length, scatter_counts
from spnn.graphs import dijkstra, random_graph

from experiments._base import rel, write_json

REF_N, REF_P, REF_SEED = 60, 0.05, 19
REF_S, REF_T = 0, 59


def run(argv=None) -> dict:
    n, edges, w = random_graph(REF_N, REF_P, seed=REF_SEED)
    path, cost = dijkstra(n, edges, w, REF_S, REF_T)
    L = len(path) - 1
    B = 2.0 * float(w.max())

    # optimal path weight (float32 weights, matched to the source harness)
    eidx = {}
    for i in range(len(edges)):
        u, v = int(edges[i][0]), int(edges[i][1])
        eidx[(u, v)] = eidx[(v, u)] = float(w[i])
    wsum = sum(eidx[(path[i], path[i + 1])] for i in range(L))
    Etilde = wsum / B

    # CNT-PATH: exact simple paths at the matched depth
    counts, minw, _witness = count_paths_by_length(n, edges, w, REF_S, REF_T, L)
    n_path = counts[L]

    # CNT-SCAT: matched-size scatter microstates on the augmented graph (L+2 edges)
    sc = scatter_counts(n, edges, REF_S, REF_T, L + 2)
    n_scatter_micro = sc["micro_aux"]

    # CNT-CROSS: the alpha-scale crossing under the microstate convention
    path_micro = n_path * (2 ** (L + 1))
    margin_c1 = (L + 1) - 2.0 * Etilde
    log_ratio = math.log(n_scatter_micro) - math.log(path_micro)
    crossing_c = log_ratio / margin_c1

    result = {
        "instance": {
            "generator": f"random_graph({REF_N}, {REF_P}, seed={REF_SEED})",
            "n": n, "n_edges": int(edges.shape[0]),
            "source": REF_S, "target": REF_T,
            "L": L, "path_weight": wsum, "B": B, "Etilde": Etilde,
        },
        "CNT_PATH": {
            "paths_at_matched_depth": n_path,
            "path_min_weight": minw[L],
        },
        "CNT_SCAT": {
            "matched_size": L + 2,
            "n_scatter_micro_aux": n_scatter_micro,
            "N0": sc["N0"], "N1": sc["N1"], "N2": sc["N2"],
        },
        "CNT_CROSS": {
            "path_micro": path_micro,
            "log_ratio_nats": log_ratio,
            "margin_at_c1": margin_c1,
            "crossing_c": crossing_c,
        },
    }

    out = write_json("counts/reference.json", result)
    print(f"[7] reference_counts -> {rel(out)}")
    print(f"    CNT-PATH  paths @ L={L} : {n_path}")
    print(f"    CNT-SCAT  micro_aux     : {n_scatter_micro:.6e}")
    print(f"    CNT-CROSS crossing c    : {crossing_c:.15g}"
          f"  (log-ratio {log_ratio:.2f} nats, margin@c1 {margin_c1:.3f})")
    return result


if __name__ == "__main__":
    run()
