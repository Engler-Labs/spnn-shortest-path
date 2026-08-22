"""Experiment 18 -- N_d, N'0 and the d=-1 sector via the corrected deficit DP (section V-D).

Claims: DEF-COUNTS, DEF-33, DEF-36, DEF-DMINUS1 (CLAIMS.tsv, section V-D).
Output: results/counts/deficit.json

The admissible matched-activity configurations are the degree-<=2 subgraphs of the
AUGMENTED graph (original graph + terminals S,T + the two zero-weight auxiliary edges
(S,s),(t,T)) with exactly m = L+2 edges: vertex-disjoint unions of paths and cycles.
The auxiliary edges are NOT forced -- the S/T drive is a beta coupling in ``sim.energy``,
not a static activation reward, so S,T-inactive configurations have the SAME static
energy and matched activity as the S,T-active ones and compete on equal footing
(exhaustive microstate ground truth confirmed ``scatter_counts``
reproduces the true scatter EXACTLY).  Each configuration carries its microstate
multiplicity 2^(#real vertices covered).  ``spnn.counting.count_degree2_by_deficit``
returns the summed weight N_d, graded by the deficit d = k_p - 1 (k_p = number of PATH
components); the degree-1 count includes S,T, so the full S-s-...-t-T path is d=0 and a
pure disjoint-cycle set is d=-1 -- the sector strictly BELOW the path.

(An earlier version forced both aux edges (the both-aux subset); it under-counted the
scatter by ~2^(L+1) and mis-signed (36).  That is superseded here -- the DP is now the
full augmented space, validated vs a full-space brute on tiny instances AND vs
``scatter_counts``' ``micro_aux`` (the d=m-1 all-disjoint = matching term).)

On the pinned reference instance ``random_graph(60,0.05,seed=19)`` (L=8, m=10):

  * DEF-COUNTS -- N_0 (weighted, d=0) = #path microstates + N'0.  #path microstates =
    (#L-hop s-t paths) * 2^(L+1); N'0 = N_0 - #path.
  * DEF-33 -- (33): c > max_{d>=1} ln(N_d/#path)/(d*abar).  The denominator is #path
    (the s-t path microstates the decoder accepts), NOT N_0: the d=0 class is
    overwhelmingly non-paths (N'0 ~ N_0), so N_0 would divide by a set ~10^5x larger
    than the accepted answers.  With #path the d=m-1 term equals the section V-B
    reference crossing (3.699), i.e. the published condition (29) IS that term; the
    argmax (the binding requirement) is d=1 (~18.11).
  * DEF-36 -- (36): the d=0 non-path sector, c > (B/2)*ln(N'0/#path)/W_cyc^min, using
    the reference girth W_cyc^min = 0.755323 as the conservative denominator.  Binds
    (N'0 >> #path once the microstate weight is counted).
  * DEF-DMINUS1 -- the d=-1 disjoint-cycle sector below the path is non-empty on the
    reference (N_{-1} > 0); see cycle_prevalence.py (CYC-PREV/CYC-RHO) for prevalence
    and the energy comparison.
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
    """DP == brute on tiny instances (all deficits, full augmented space)."""
    for N, p, seed in [(11, 0.28, 7), (10, 0.30, 11), (9, 0.35, 3)]:
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
    path_micro = npath * 2 ** (L + 1)               # each L-hop path covers L+1 real vertices
    Nprime0 = N0 - path_micro

    # cross-check: the d=m-1 (all-disjoint = matching) term == scatter micro_aux
    micro_aux = scatter_counts(n, edges, s, t, m)["micro_aux"]
    assert Nd[m - 1] == micro_aux, f"N_{m-1}={Nd[m-1]} != scatter micro_aux={micro_aux}"

    # (33): c > max_{d>=1} ln(N_d/#path)/(d*abar).  The denominator is #path (the s-t
    # PATH microstates the decoder accepts), NOT N_0: the d=0 class is overwhelmingly
    # non-paths (N'0 ~ N_0), so dividing by N_0 would compare against a set ~10^5x larger
    # than the accepted answers.  With #path the d=m-1 term equals the section V-B
    # reference crossing exactly, i.e. the published condition (29) IS that term.
    grade = {}
    for d in range(1, m):
        val = (math.log(Nd[d]) - math.log(path_micro)) / (d * abar)
        grade[d] = dict(N_d=Nd[d], bound=val)
    argmax_d = max(grade, key=lambda d: grade[d]["bound"])
    vb_crossing = grade[m - 1]["bound"]           # the d=m-1 term == section V-B's 3.699
    assert abs(vb_crossing - 3.699) < 2e-3, f"d=m-1 term {vb_crossing} != V-B crossing 3.699"

    # (36): d=0 non-path sector, c > (B/2)*ln(N'0/#path)/W_cyc^min
    ln_np = math.log(Nprime0 / path_micro)
    d0_bound = (B / 2.0) * ln_np / girth

    Nminus1 = Nd.get(-1, 0)

    result = {
        "instance": f"random_graph({REF['n']},{REF['p']},seed={REF['seed']})",
        "L": L, "m": m, "B": B, "abar": abar, "W_cyc_min": girth,
        "aux_forced": False,
        "max_states": stats["max_states"], "order_max_frontier": stats["order_max_frontier"],
        "micro_weight": "each config weighted by 2^(#real vertices covered); aux NOT "
                        "forced (S,T may be inactive). N_d is the summed weight.",
        "N_d": {str(d): Nd[d] for d in sorted(Nd)},
        "N_0": N0, "n_stpaths": npath, "path_micro": path_micro, "N_prime_0": Nprime0,
        "crosscheck_Nm1_eq_scatter_micro_aux": True,
        "grade_33_denominator": "#path (s-t path microstates the decoder accepts), not N_0",
        "grade_33": {str(d): grade[d] for d in grade},
        "argmax_d": argmax_d, "argmax_is_m_minus_1": argmax_d == m - 1,
        "binding_bound": grade[argmax_d]["bound"],
        "published_term_d_m1_bound": grade[m - 1]["bound"],
        "d_m1_eq_vb_crossing": vb_crossing,
        "d0_sector_36": {
            "N_prime_0": Nprime0, "path_micro": path_micro,
            "ln_ratio": ln_np, "bound_c": d0_bound,
            "vacuous": d0_bound < 0,
            "reading": ("binds at c > %.4f" % d0_bound if d0_bound >= 0 else
                        "vacuous (N'0 < #path)"),
        },
        "dminus1_sector": {
            "N_minus_1": Nminus1, "exists": Nminus1 > 0,
            "reading": "the disjoint-cycle sector strictly below the path is non-empty "
                       "on the reference (see CYC-PREV/CYC-RHO for prevalence + energies)",
        },
    }
    out = write_json("counts/deficit.json", result)

    print(f"[18] deficit_counts -> {rel(out)}  (full-space deficit DP, reference L={L}, m={m})")
    print(f"    max_states={stats['max_states']:,}; N_{m-1} == scatter micro_aux  OK")
    print(f"    N_0={N0}  = #path_micro({path_micro}) + N'0({Nprime0:.4e})  [#paths={npath}]")
    print(f"    N_-1={Nminus1} (d=-1 disjoint-cycle sector exists: {Nminus1>0})")
    print(f"    (33) c > max_d ln(N_d/#path)/(d*abar), abar={abar:.4f}:")
    for d in range(1, m):
        mark = "  <-- argmax" if d == argmax_d else ""
        print(f"      d={d}: {grade[d]['bound']:.3f}{mark}")
    print(f"    --> argmax d={argmax_d} ({grade[argmax_d]['bound']:.3f}); "
          f"d=m-1={m-1} term = {vb_crossing:.4f} == V-B crossing (so (29) IS the d=m-1 term)")
    print(f"    (36) d=0 sector: N'0={Nprime0:.4e} {'<' if Nprime0<path_micro else '>='} "
          f"#path_micro={path_micro}; bound c > {d0_bound:.4f}  "
          f"-> {'VACUOUS (favorable)' if d0_bound<0 else 'BINDS'}")
    return result


if __name__ == "__main__":
    run()
