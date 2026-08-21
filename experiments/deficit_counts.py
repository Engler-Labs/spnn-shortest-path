"""Experiment 18 -- N_d and N'0 via the degree-<=2 deficit DP (section V-D).

Claims: DEF-COUNTS, DEF-33, DEF-36 (CLAIMS.tsv, section V-D).
Output: results/counts/deficit.json

Closes the last unsigned direction in section V-D.  The admissible matched-activity
configurations are the degree-<=2 subgraphs of the augmented graph with m = L+2 edges
(vertex-disjoint paths + cycles), graded by the deficit d = k_p - 1 (k_p = number of
path components).  ``spnn.counting.count_degree2_by_deficit`` counts N_d exactly for
d = 0..m-1; validated against a brute-force reference on tiny instances AND against
``scatter_counts`` (the d=m-1 all-disjoint term equals its N2).

On the pinned reference instance ``random_graph(60,0.05,seed=19)`` (L=8, m=10):

  * DEF-COUNTS -- N_0 (all d=0 configs) = 1754 s-t paths + N'0 shorter-path-plus-cycle
    configs.  N'0 = N_0 - #(L-hop s-t paths).
  * DEF-33 -- (33): c > max_{d>=1} ln(N_d/N_0)/(d*abar).  Reports whether d = m-1 is
    the argmax (i.e. whether the published condition (29) is the binding term -- an
    open question in section V-D).  Microstate convention: a deficit-d config covers
    L+d+1 real vertices, so its microstate weight is 2^(L+d+1); this multiplies
    N_d/N_0 by 2^d, i.e. adds d*ln2 to the numerator.
  * DEF-36 -- (36): the d=0 non-path sector, c > (B/2)*ln(N'0/#path)/W_cyc^min, using
    the reference girth W_cyc^min = 0.755323 as the conservative denominator.  Signs
    the direction the paper left unsigned.
"""

from __future__ import annotations

import math

import numpy as np

from spnn.counting import (brute_degree2_by_deficit, count_degree2_by_deficit,
                           count_paths_by_length, scatter_counts)
from spnn.graphs import dijkstra, min_weight_cycle, random_graph

from experiments._base import rel, write_json

REF = dict(n=60, p=0.05, seed=19, s=0, t=59)


def _selfcheck():
    """DP == brute on two tiny instances (all deficits)."""
    for N, p, seed in [(11, 0.28, 7), (10, 0.30, 11)]:
        n, edges, w = random_graph(N, p, seed=seed)
        L = len(dijkstra(n, edges, w, 0, n - 1)[0]) - 1
        dp, _ = count_degree2_by_deficit(n, edges, 0, n - 1, L)
        bf = brute_degree2_by_deficit(n, edges, 0, n - 1, L)
        if dp != bf:
            raise AssertionError(f"DP != brute on N={N} seed={seed}: {dp} vs {bf}")


def run(argv=None) -> dict:
    _selfcheck()

    n, edges, w = random_graph(REF["n"], REF["p"], seed=REF["seed"])
    s, t = REF["s"], REF["t"]
    path, cost = dijkstra(n, edges, w, s, t)
    L = len(path) - 1
    m = L + 2
    wmax = float(np.max(w))
    B = 2.0 * wmax
    Wstar = float(cost)
    Etilde = Wstar / B
    margin = (L + 1) - 2 * Etilde
    abar = margin / (L + 1)
    girth, _ = min_weight_cycle(n, edges, w)

    Nd, stats = count_degree2_by_deficit(n, edges, s, t, L)
    N0 = Nd[0]
    npath = count_paths_by_length(n, edges, w, s, t, L)[0][L]
    Nprime0 = N0 - npath

    # cross-check: the d=m-1 (all-disjoint) term == scatter_counts N2
    n2 = scatter_counts(n, edges, s, t, m)["N2"]
    assert Nd[m - 1] == n2, f"N_{m-1}={Nd[m-1]} != scatter N2={n2}"

    # (33): c > max_{d>=1} ln(N_d/N_0)/(d*abar), microstate convention (+ d*ln2)
    ln2 = math.log(2.0)
    grade = {}
    for d in range(1, m):
        edge_ratio = math.log(Nd[d] / N0)
        micro = (edge_ratio + d * ln2) / (d * abar)     # microstate convention
        edge = edge_ratio / (d * abar)                  # edge-count convention
        grade[d] = dict(N_d=Nd[d], ln_ratio_edge=edge_ratio,
                        micro=micro, edge=edge)
    argmax_d = max(grade, key=lambda d: grade[d]["micro"])

    # (36): d=0 non-path sector, c > (B/2)*ln(N'0/#path)/W_cyc^min
    ln_np = math.log(Nprime0 / npath)
    d0_bound = (B / 2.0) * ln_np / girth

    result = {
        "instance": f"random_graph({REF['n']},{REF['p']},seed={REF['seed']})",
        "L": L, "m": m, "B": B, "abar": abar, "W_cyc_min": girth,
        "max_states": stats["max_states"], "order_max_frontier": stats["order_max_frontier"],
        "micro_multiplicity": "config at deficit d covers L+d+1 real vertices; "
                              "microstate weight 2^(L+d+1) (d=0 -> 2^(L+1), the path)",
        "N_d": {str(d): Nd[d] for d in sorted(Nd)},
        "N_0": N0, "n_stpaths_full": npath, "N_prime_0": Nprime0,
        "crosscheck_Nm1_eq_scatter_N2": True,
        "grade_33": {str(d): grade[d] for d in grade},
        "argmax_d": argmax_d, "argmax_is_m_minus_1": argmax_d == m - 1,
        "binding_term_micro": grade[argmax_d]["micro"],
        "published_term_d_m1_micro": grade[m - 1]["micro"],
        "d0_sector_36": {
            "N_prime_0": Nprime0, "n_path": npath,
            "ln_ratio": ln_np, "bound_c": d0_bound,
            "vacuous": d0_bound < 0,
            "reading": ("N'0 < #path so the bound is negative: the d=0 non-path sector "
                        "imposes NO constraint on c -- the direction the paper left "
                        "unsigned is favorable" if d0_bound < 0 else
                        "binds at c > %.3f" % d0_bound),
        },
    }
    out = write_json("counts/deficit.json", result)

    print(f"[18] deficit_counts -> {rel(out)}  (degree-<=2 DP, reference L={L}, m={m})")
    print(f"    max_states={stats['max_states']:,}; N_{m-1} == scatter N2  OK")
    print(f"    N_0={N0}  = #paths({npath}) + N'0({Nprime0})")
    print(f"    N_d: " + ", ".join(f"{d}:{Nd[d]:.3g}" for d in sorted(Nd)))
    print(f"    (33) c > max_d ln(N_d/N_0)/(d*abar) [micro], abar={abar:.4f}:")
    for d in range(1, m):
        mark = "  <-- argmax" if d == argmax_d else ""
        print(f"      d={d}: {grade[d]['micro']:.3f}{mark}")
    print(f"    --> argmax d={argmax_d}; d=m-1={m-1} is argmax? {result['argmax_is_m_minus_1']} "
          f"(published (29) term = {grade[m-1]['micro']:.3f})")
    print(f"    (36) d=0 sector: N'0={Nprime0} {'<' if Nprime0<npath else '>='} #path={npath}; "
          f"bound c > {d0_bound:.3f}  -> {'VACUOUS (favorable)' if d0_bound<0 else 'binds'}")
    return result


if __name__ == "__main__":
    run()
