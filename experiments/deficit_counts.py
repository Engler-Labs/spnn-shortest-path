"""Experiment 18 -- N_d, N'0 and the d=-1 sector via the corrected deficit DP (section V-D).

Claims: DEF-COUNTS, DEF-33, DEF-36, DEF-DMINUS1, DEF-D0ABOVE (CLAIMS.tsv, section V-D).
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
  * DEF-36 -- (36): the d=0 non-path sector, c > (B/2)*ln(N'0/#path)/delta_min, where
    delta_min is the MINIMUM energy gap between a d=0 non-path config and the path (a light
    one-aux PATH, ~0.043), NOT the girth.  The whole d=0 non-path class is above the path
    (DEF-D0ABOVE); the girth (0.755) is instead the rigorous proof that every CYCLE-bearing
    member is above (any cycle >= girth > W*).  Using the girth as the denominator
    understates the required c by ~17.6x (15.6 vs ~273.5).
  * DEF-D0ABOVE -- no d=0 member is below the path at c=4: cycle-bearing members via the
    girth argument, pure-path members via an exhaustive DFS (min g = 0.6121 > W*).  So the
    below-path / c-unsuppressible sector is exactly d=-1 (pure cycles), removed by the drive.
  * DEF-DMINUS1 -- the d=-1 disjoint-cycle sector below the path is non-empty on the
    reference (N_{-1} > 0); see cycle_prevalence.py (CYC-PREV/CYC-RHO) for prevalence
    and the energy comparison.
"""

from __future__ import annotations

import math

import numpy as np

from spnn import Params, compile_network
from spnn.counting import (brute_degree2_by_deficit, count_degree2_by_deficit,
                           count_paths_by_length, scatter_counts)
from spnn.graphs import dijkstra, min_weight_cycle, random_graph
from spnn.sim import energy, ideal_state

from experiments._base import rel, write_json

REF = dict(n=60, p=0.05, seed=19, s=0, t=59)


def _d0_min_gap(n, edges, w, s, t, path, Wstar, wmax, B):
    """Minimum energy gap between a d=0 NON-path config and the optimal path, over all
    aux usages -- the correct denominator for (36).

    The d=0 non-path class is ENTIRELY above the path (no member below): any CYCLE-bearing
    member weighs >= the weighted girth > W* (rigorous), and the PURE-path members (one-aux
    and aux-free) are searched EXHAUSTIVELY.  The binding gap is a light one-aux PATH, not a
    light cycle -- so the girth is NOT the (36) denominator; delta_min is.  gap is measured as
    ``delta = W_x - W* - (path component's two end-edge weights)/2`` (aux edges weight 0, so a
    one-aux member discounts only its real end); each is checked against the compiled network.
    """
    C = D = 4.0
    bk = -(D + C) / 2.0                      # -> lambda = 0
    m = len(path) - 1 + 2
    wl = {}
    adj = {}
    for i in range(len(edges)):
        a, b = int(edges[i][0]), int(edges[i][1])
        wl[(a, b)] = wl[(b, a)] = float(w[i])
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    adj_s = {v: sorted(adj[v], key=lambda u: wl[(v, u)]) for v in adj}
    net = compile_network(n, edges, w, s, t,
                          Params(D=D, alpha_scale=C, bias=bk, b_st=None, st_mode="always_on"),
                          augment_st=True)
    S_, T_ = net.aug_source, net.aug_target
    E_path = float(sum(energy(net, ideal_state(net, net.augmented_path(path)))))

    def st_edges(es):
        nb = {}
        for a, b in es:
            nb.setdefault(a, []).append(b)
            nb.setdefault(b, []).append(a)
        x = np.zeros(net.n_neurons, dtype=bool)
        for v, us in nb.items():
            for slot, u in enumerate(us):
                x[net.neuron_at(v, u, slot)] = True
        return x

    def min_oneaux(anchor):     # aux + (m-1) real edges; the aux attaches at the anchor, so
        start = s if anchor == "s" else t   # the FAR (last) real end is the deg-1 discount
        best = [math.inf, None]

        def dfs(v, depth, wsum, seq, seen):
            if (wsum - wmax / 2) - Wstar >= best[0]:
                return
            if depth == m - 1:
                far = wl[(seq[-2], seq[-1])]
                d = wsum - far / 2 - Wstar
                if d < best[0]:
                    best[0] = d
                    best[1] = list(seq)
                return
            for u in adj_s[v]:
                if u in seen:
                    continue
                nw = wsum + wl[(v, u)]
                if (nw - wmax / 2) - Wstar >= best[0]:
                    break
                seen.add(u)
                seq.append(u)
                dfs(u, depth + 1, nw, seq, seen)
                seq.pop()
                seen.discard(u)
        dfs(start, 0, 0.0, [start], {start})
        return best

    def min_auxfree():          # m real edges, both ends real -> discount both
        best = [math.inf, None]

        def dfs(v, depth, wsum, wa, seq, seen):
            if (wsum - wa / 2) - wmax / 2 - Wstar >= best[0]:
                return
            if depth == m:
                wb = wl[(seq[-2], seq[-1])]
                d = wsum - (wa + wb) / 2 - Wstar
                if d < best[0]:
                    best[0] = d
                    best[1] = list(seq)
                return
            for u in adj_s[v]:
                if u in seen:
                    continue
                we = wl[(v, u)]
                nwa = we if depth == 0 else wa
                nw = wsum + we
                if (nw - nwa / 2) - wmax / 2 - Wstar >= best[0]:
                    break
                seen.add(u)
                seq.append(u)
                dfs(u, depth + 1, nw, nwa, seq, seen)
                seq.pop()
                seen.discard(u)
        for st0 in range(n):
            dfs(st0, 0, 0.0, 0.0, [st0], {st0})
        return best

    classes = {}
    for a in ("s", "t"):
        b = min_oneaux(a)
        P = b[1]
        es = ([(S_, s)] + [(P[i], P[i + 1]) for i in range(m - 1)]) if a == "s" else \
             ([(P[i], P[i + 1]) for i in range(m - 1)] + [(t, T_)])
        E = float(sum(energy(net, st_edges(es))))
        classes["one_aux_" + a] = dict(delta=b[0], energy=E,
                                       delta_energy=(E - E_path) * B / (2 * C))
    bf = min_auxfree()
    P = bf[1]
    E = float(sum(energy(net, st_edges([(P[i], P[i + 1]) for i in range(m)]))))
    classes["aux_free"] = dict(delta=bf[0], energy=E, delta_energy=(E - E_path) * B / (2 * C))
    for k, v in classes.items():            # each formula delta must match the compiled net
        assert abs(v["delta"] - v["delta_energy"]) < 1e-4, (k, v)
    minkey = min(classes, key=lambda k: classes[k]["delta"])
    return dict(E_path=E_path, delta_min=classes[minkey]["delta"], minimiser=minkey,
                min_energy=classes[minkey]["energy"], by_class=classes,
                any_below_path=any(v["energy"] < E_path - 1e-9 for v in classes.values()))


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

    # (36): d=0 non-path sector, c > (B/2)*ln(N'0/#path)/delta_min. The denominator is the
    # MINIMUM energy gap between a d=0 non-path config and the path (delta_min), NOT the
    # girth. The whole d=0 non-path class is above the path; the binding gap is a light
    # one-aux PATH (delta_min), while the girth is the rigorous proof that every
    # cycle-bearing member is above (any cycle >= girth > W*) -- its role, not the denom.
    ln_np = math.log(Nprime0 / path_micro)
    d0 = _d0_min_gap(n, edges, w, s, t, path, Wstar, wmax, B)
    delta_min = d0["delta_min"]
    d0_bound = (B / 2.0) * ln_np / delta_min           # corrected (36)
    girth_bound = (B / 2.0) * ln_np / girth            # girth version -- understates c

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
            "N_prime_0": Nprime0, "path_micro": path_micro, "ln_ratio": ln_np,
            "delta_min": delta_min, "delta_min_minimiser": d0["minimiser"],
            "bound_c": d0_bound,                       # corrected (36), denominator delta_min
            "girth_bound_c": girth_bound,              # girth version -- understates
            "girth_understates_by": d0_bound / girth_bound,
            "d0_min_energy": d0["min_energy"], "E_path": d0["E_path"],
            "any_d0_member_below_path": d0["any_below_path"],
            "by_class": d0["by_class"],
            "girth_role": "rigorous proof that every CYCLE-bearing d=0 member is above the "
                          "path (any cycle >= girth W_cyc_min > W*); NOT the (36) denominator",
            "reading": ("binds at c > %.1f (delta_min=%.5f, a light one-aux PATH); the "
                        "girth-based %.1f understates it %.1fx"
                        % (d0_bound, delta_min, girth_bound, d0_bound / girth_bound)),
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
    print(f"    (36) d=0 non-path sector: delta_min={delta_min:.5f} ({d0['minimiser']}, a light "
          f"path) -> c > {d0_bound:.1f}  (girth-based {girth_bound:.1f} understated {d0_bound/girth_bound:.1f}x)")
    print(f"         no d=0 member below the path: {not d0['any_below_path']}  "
          f"(min d=0 non-path E_N={d0['min_energy']:.4f} vs E_path={d0['E_path']:.4f})")
    return result


if __name__ == "__main__":
    run()
