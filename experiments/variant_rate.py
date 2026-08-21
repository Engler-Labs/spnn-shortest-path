"""Experiment 17 -- variant-vs-Dijkstra disagreement rate + augmented control (IV-E).

Claims: VAR-RATE, ORD-CONTROL (CLAIMS.tsv, section IV-E).
Output: results/variant/disagreement.json

Section IV-E states outright that it does NOT measure how often the VARIANT
(unaugmented) construction's energy minimum disagrees with Dijkstra, nor by how much
weight.  This makes that measurement, over a population of enumerable instances, at
lambda = 0 (b_k = -(D+c)/2), c = 4.

For each instance we enumerate every simple s->t path, read each path state's E_N
from the compiler, and take the argmin -- once on the VARIANT network (augment_st =
False; interior edges charged twice, so the minimum is not the least-weight path) and
once on the AUGMENTED network as a CONTROL.

  * VAR-RATE   -- fraction of instances where the variant's E_N-minimum path is NOT
                 Dijkstra's, and where they differ, the relative weight excess
                 (W_variant - W*)/W* of the variant's choice.  The paper's constructed
                 example is 24 % heavier; this gives the population distribution.
  * ORD-CONTROL -- the same on the AUGMENTED network.  Corollary 1(iii) says the
                 augmented ordering at lambda = 0 IS by weight, so the disagreement
                 rate must be EXACTLY ZERO.  A nonzero rate would contradict the
                 central ordering theorem -- this is a population-scale empirical
                 check the paper otherwise lacks.
"""

from __future__ import annotations

import numpy as np

from spnn.compile import Params, compile_network
from spnn.graphs import dijkstra, random_graph
from spnn.sim import energy, ideal_state

from experiments._base import rel, write_json

D = 4.0
C = 4.0
LAM = 0.0
PATH_CAP = 4000                 # skip instances with more simple s->t paths than this
# population: small enumerable G(n,p), scanned deterministically
SCAN_N = (8, 10, 12, 14)
SCAN_P = (0.15, 0.22, 0.30)
SCAN_SEEDS = range(40)


def _enum_paths(n, edges, s, t, cap):
    adj = {i: [] for i in range(n)}
    for a, b in edges:
        a, b = int(a), int(b)
        adj[a].append(b)
        adj[b].append(a)
    out, on, stack = [], [False] * n, [s]
    on[s] = True
    over = [False]

    def rec(v):
        if len(out) > cap:
            over[0] = True
            return
        if v == t:
            out.append(tuple(stack))
            return
        for u in adj[v]:
            if not on[u]:
                on[u] = True
                stack.append(u)
                rec(u)
                stack.pop()
                on[u] = False
    rec(s)
    return (None if over[0] else out)


def _argmin_path(n, edges, w, t, paths, augment):
    bk = (LAM - D - C) / 2.0
    params = Params(D=D, alpha_scale=C, bias=bk, b_st=None if augment else 0.0)
    net = compile_network(n, edges, w, 0, t, params=params, augment_st=augment)
    best, best_e = None, None
    for q in paths:
        e = float(sum(energy(net, ideal_state(net, net.augmented_path(list(q))))))
        if best_e is None or e < best_e:
            best, best_e = q, e
    return best


def run(argv=None) -> dict:
    var_excess = []           # relative weight excess where the variant disagrees
    n_inst = n_var_disagree = n_aug_disagree = 0
    aug_disagreements = []

    for N in SCAN_N:
        for p in SCAN_P:
            for seed in SCAN_SEEDS:
                n, edges, w = random_graph(N, p, seed=seed)
                t = n - 1
                P, _ = dijkstra(n, edges, w, 0, t)
                if P is None:
                    continue
                paths = _enum_paths(n, edges, 0, t, PATH_CAP)
                if paths is None or len(paths) < 2:
                    continue
                idx = {}
                for i, (a, b) in enumerate(edges):
                    idx[(int(a), int(b))] = idx[(int(b), int(a))] = float(w[i])
                wsum = lambda q: sum(idx[(q[i], q[i + 1])] for i in range(len(q) - 1))
                P = tuple(P)
                Wstar = wsum(P)
                n_inst += 1

                var = _argmin_path(n, edges, w, t, paths, augment=False)
                if var != P:
                    n_var_disagree += 1
                    var_excess.append((wsum(var) - Wstar) / Wstar)

                aug = _argmin_path(n, edges, w, t, paths, augment=True)
                if aug != P:
                    n_aug_disagree += 1
                    aug_disagreements.append(dict(
                        instance=f"random_graph({N},{p},seed={seed})",
                        aug_weight=wsum(aug), opt_weight=Wstar))

    ex = sorted(var_excess)
    q = lambda pp: ex[min(len(ex) - 1, int(pp * (len(ex) - 1) + 0.5))] if ex else None
    result = {
        "params": {"lambda": LAM, "c": C, "D": D, "path_cap": PATH_CAP},
        "n_instances": n_inst,
        "VAR_RATE": {
            "n_disagree": n_var_disagree,
            "disagreement_fraction": n_var_disagree / n_inst if n_inst else None,
            "weight_excess_over_disagreements": {
                "n": len(ex), "min": ex[0] if ex else None, "median": q(0.5),
                "q75": q(0.75), "q90": q(0.90), "max": ex[-1] if ex else None,
                "frac_ge_0.24": (sum(1 for x in ex if x >= 0.24) / len(ex)) if ex else None,
            },
            "note": "variant E_N-min path vs Dijkstra; excess = (W_var - W*)/W*. The "
                    "paper's constructed VAR-INV4 example is 0.24 (24% heavier).",
        },
        "ORD_CONTROL": {
            "n_disagree": n_aug_disagree,
            "disagreement_fraction": n_aug_disagree / n_inst if n_inst else None,
            "corollary_1iii_holds": n_aug_disagree == 0,
            "disagreements": aug_disagreements,
            "note": "augmented ordering at lambda=0 must equal Dijkstra (Cor 1(iii)); "
                    "a nonzero rate contradicts the central ordering theorem.",
        },
    }
    out = write_json("variant/disagreement.json", result)

    vr = result["VAR_RATE"]
    we = vr["weight_excess_over_disagreements"]
    print(f"[17] variant_rate -> {rel(out)}  ({n_inst} enumerable instances, "
          f"lambda={LAM:.0f}, c={C:.0f})")
    print(f"    VAR-RATE   variant disagrees with Dijkstra on "
          f"{vr['n_disagree']}/{n_inst} = {100*vr['disagreement_fraction']:.1f}%")
    if ex:
        print(f"      weight excess (W_var-W*)/W* over disagreements: median "
              f"{100*we['median']:.1f}%, q90 {100*we['q90']:.1f}%, max {100*we['max']:.1f}%; "
              f"{100*we['frac_ge_0.24']:.0f}% are >= the 24% constructed example")
    oc = result["ORD_CONTROL"]
    verdict = "OK -- Cor 1(iii) holds" if oc["corollary_1iii_holds"] else "*** VIOLATED ***"
    print(f"    ORD-CONTROL augmented disagrees on {oc['n_disagree']}/{n_inst}  "
          f"(must be 0)  {verdict}")
    if not oc["corollary_1iii_holds"]:
        print("    !!! AUGMENTED ORDERING VIOLATES Corollary 1(iii) -- escalate !!!")
    return result


if __name__ == "__main__":
    run()
