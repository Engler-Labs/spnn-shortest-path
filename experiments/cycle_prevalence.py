"""Experiment 19 -- prevalence of the d=-1 (disjoint-cycle) sector (section V-D).

Claims: CYC-PREV, CYC-RHO, CYC-ENERGY (CLAIMS.tsv, section V-D).
Output: results/counts/cycle_prevalence.json

The d=-1 sector (Exp 18): a set of vertex-disjoint cycles on exactly
m = L+2 edges whose total weight is below w_max + W* is CHEAPER than the optimal path,
and the energy gap grows linearly in c.  This measures how common that is, TWO-SIDED
and bounded, never imputing the undetermined cases:

  * ABSENT (rigorous)  -- the sum of the m lightest edges of G is >= w_max + W*.  Then
    no m-edge set, cycle or otherwise, can beat the threshold.  O(|E| log|E|), no search.
  * PRESENT (rigorous) -- a bounded heuristic (lightest simple cycles, greedily combined
    vertex-disjoint to exactly m edges) EXHIBITS a disjoint-cycle set below threshold.
    A witness is a proof.
  * UNDETERMINED       -- neither certificate fired.  A heuristic that fails to find one
    is NOT a proof of absence; reported as its own bucket (like the population's censored
    cells), never imputed either way.

For PRESENT cases the energy gap below the path at c=4 is reported via the verified
identity gap = -c[1 + 2(W* - W_cyc)/B] (B = 2 w_max).  The absence certificate is cheap
enough to be a per-instance designer check, like the ordering tolerance and the girth.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

import numpy as np

from spnn import Params, compile_network
from spnn.graphs import dijkstra, grid_graph, mountain_spatial_graph, random_graph
from spnn.sim import energy, ideal_state

from experiments._base import rel, write_json

# same generators / cells as creq_sweep
DEG_SPARSE, DEG_DENSE = 3.0, 6.0
GRID_JITTER, MTN_KNN_K = 0.25, 4
MIN_OPT_DEPTH = 3
FAM_ORDER = ["sparse", "dense", "grid", "mtn_knn"]
# a subsample: a few sizes per family x seeds, enough to state a per-family rate
SUBSAMPLE = {
    "sparse": [(20, s) for s in range(1, 13)] + [(40, s) for s in range(1, 13)]
              + [(60, s) for s in range(1, 13)],
    "dense": [(20, s) for s in range(1, 13)] + [(40, s) for s in range(1, 13)]
             + [(60, s) for s in range(1, 13)],
    "grid": [(5, s) for s in range(1, 13)] + [(7, s) for s in range(1, 13)]
            + [(9, s) for s in range(1, 13)],
    "mtn_knn": [(20, s) for s in range(1, 13)] + [(40, s) for s in range(1, 13)]
               + [(60, s) for s in range(1, 13)],
}
CYCLE_POOL = 80          # keep this many lightest cycles for the presence search
SEARCH_NODES = 200000    # cap on the combination search


def _build(family, size, seed):
    if family == "sparse":
        return random_graph(size, DEG_SPARSE / (size - 1), seed=seed)
    if family == "dense":
        return random_graph(size, DEG_DENSE / (size - 1), seed=seed)
    if family == "grid":
        return grid_graph(size, seed=seed, jitter=GRID_JITTER)
    if family == "mtn_knn":
        return mountain_spatial_graph(size, seed=seed, k=MTN_KNN_K)
    raise ValueError(family)


def _terminals(n, edges, w):
    s = 0
    path, _ = dijkstra(n, edges, w, s, n - 1)
    if path is not None and len(path) - 1 >= MIN_OPT_DEPTH:
        return s, n - 1, path
    best = None
    for tv in range(1, n):
        p2, _ = dijkstra(n, edges, w, s, tv)
        if p2 is None:
            continue
        d = len(p2) - 1
        if d >= MIN_OPT_DEPTH and (best is None or d < best[0]):
            best = (d, tv, p2)
    return (s, best[1], best[2]) if best else (None, None, None)


def _light_cycles(n, adj, wl):
    """Lightest simple cycle through each edge (girth-through-edge), deduped."""
    pool = {}
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            dist = {v: 0.0}; prev = {}; pq = [(0.0, v)]
            while pq:
                d, x = heapq.heappop(pq)
                if x == u:
                    cyc = [u]; y = prev.get(u)
                    ok = True; seen = {u}
                    while y is not None and y != v:
                        if y in seen:
                            ok = False; break
                        cyc.append(y); seen.add(y); y = prev.get(y)
                    cyc.append(v)
                    if ok and len(cyc) >= 3:
                        key = frozenset(cyc); wt = d + wl[(u, v)]
                        if key not in pool or wt < pool[key][0]:
                            pool[key] = (wt, cyc)
                    break
                if d > dist.get(x, 1e18):
                    continue
                for y in adj[x]:
                    if x == v and y == u:
                        continue
                    nd = d + wl[(x, y)]
                    if nd < dist.get(y, 1e18):
                        dist[y] = nd; prev[y] = x; heapq.heappush(pq, (nd, y))
    return sorted(pool.values())[:CYCLE_POOL]


def _min_cycle_set(cycles, m, threshold):
    """Lightest vertex-disjoint cycle combination on exactly m edges (bounded DFS)."""
    best = [math.inf]
    nodes = [0]

    def rec(idx, used, nedge, wsum):
        if nodes[0] > SEARCH_NODES:
            return
        nodes[0] += 1
        if nedge == m:
            if wsum < best[0]:
                best[0] = wsum
            return
        if nedge > m or wsum >= best[0]:
            return
        for j in range(idx, len(cycles)):
            wt, cyc = cycles[j]
            if nedge + len(cyc) > m:
                continue
            if used.isdisjoint(cyc):
                rec(j + 1, used | set(cyc), nedge + len(cyc), wsum + wt)
    rec(0, frozenset(), 0, 0.0)
    return best[0]


def _min_cycle_set_witness(cycles, m, threshold):
    """As _min_cycle_set, but also return the witnessing list of cycles (ordered walks)."""
    best = [math.inf, None]
    nodes = [0]

    def rec(idx, used, nedge, wsum, chosen):
        if nodes[0] > SEARCH_NODES:
            return
        nodes[0] += 1
        if nedge == m:
            if wsum < best[0]:
                best[0] = wsum
                best[1] = list(chosen)
            return
        if nedge > m or wsum >= best[0]:
            return
        for j in range(idx, len(cycles)):
            wt, cyc = cycles[j]
            if nedge + len(cyc) > m:
                continue
            if used.isdisjoint(cyc):
                rec(j + 1, used | set(cyc), nedge + len(cyc), wsum + wt, chosen + [cyc])
    rec(0, frozenset(), 0, 0.0, [])
    return best[0], best[1]


def _reference_energies(n, edges, w, s, t, path):
    """The two absolute energy anchors on the reference (CYC-ENERGY), showing the path
    is NOT the matched-activity minimum: the compiled-network energy of the optimal path
    versus that of the lightest disjoint-cycle set, at the operating point
    c = 4, lambda = 0, b_st = b_k.

    Built-in cross-check: the path energy is c(1 + 2*Etilde) in closed form (Prop 2 at
    b_S = b_k, lambda = 0), so if the compiled network returns anything else the V-E
    operating point does not match III-A -- a manuscript problem, and this asserts on it.
    The cycle state activates, at each cycle vertex, its two cycle edges in opposite WTA
    slots (like a path-interior vertex), inducing exactly the cycle.
    """
    C = D = 4.0
    bk = -(D + C) / 2.0                     # -> lambda = 2*bk + D + c = 0 exactly
    wl = {}
    adj = defaultdict(list)
    for i in range(len(edges)):
        a, b = int(edges[i][0]), int(edges[i][1])
        wl[(a, b)] = wl[(b, a)] = float(w[i])
        adj[a].append(b)
        adj[b].append(a)
    L = len(path) - 1
    m = L + 2
    Wstar = sum(wl[(path[i], path[i + 1])] for i in range(L))
    wmax = float(np.max(w))
    B = 2 * wmax
    Etilde = Wstar / B

    net = compile_network(n, edges, w, s, t,
                          Params(D=D, alpha_scale=C, bias=bk, b_st=None, st_mode="always_on"),
                          augment_st=True)
    e_nc, e_spc = energy(net, ideal_state(net, net.augmented_path(path)))
    E_path = float(e_nc + e_spc)
    E_path_analytic = C * (1 + 2 * Etilde)
    assert abs(E_path - E_path_analytic) < 1e-3, \
        f"compiled path E_N {E_path} != analytic c(1+2Et) {E_path_analytic} -- V-E vs III-A"

    wcyc, cyc_set = _min_cycle_set_witness(_light_cycles(n, adj, wl), m, wmax + Wstar)
    x = np.zeros(net.n_neurons, dtype=bool)
    for cyc in cyc_set:
        Lc = len(cyc)
        for i, v in enumerate(cyc):
            x[net.neuron_at(v, cyc[i - 1], 0)] = True
            x[net.neuron_at(v, cyc[(i + 1) % Lc], 1)] = True
    e_nc2, e_spc2 = energy(net, x)
    E_cyc = float(e_nc2 + e_spc2)

    return dict(
        operating_point="c=4, D=4, b_k=-(D+c)/2=-4 (lambda=0), b_st=b_k, augment_st, drive on",
        path_energy_c4=E_path, path_energy_analytic=E_path_analytic,
        cycle_energy_c4=E_cyc, energy_gap_c4=E_cyc - E_path,
        W_cyc=wcyc, cycle_lengths=[len(c) for c in cyc_set],
        cross_check_path_eq_analytic=bool(abs(E_path - E_path_analytic) < 1e-6),
        note="path E_N = c(1+2*Etilde) (Prop 2 at b_S=b_k, lambda=0); cycle E_N via the "
             "lightest disjoint-cycle-set state; print as 6.210 / 4.929, gap -1.281")


def classify(n, edges, w, s, t, path):
    L = len(path) - 1; m = L + 2
    idx = {}
    wl = {}; adj = defaultdict(list)
    for i in range(len(edges)):
        a, b = int(edges[i][0]), int(edges[i][1])
        wl[(a, b)] = wl[(b, a)] = float(w[i]); adj[a].append(b); adj[b].append(a)
        idx[(a, b)] = idx[(b, a)] = float(w[i])
    Wstar = sum(idx[(path[i], path[i + 1])] for i in range(L))
    wmax = float(np.max(w)); B = 2 * wmax; thr = wmax + Wstar
    m_lightest = float(np.sum(np.sort(w)[:m]))
    # rho-law form: the absence certificate rewritten via
    #   rho = mu/(2 w_max),  gamma = mean(m lightest)/mu,  mu = W*/L
    # is  gamma >= (L + 1/(2 rho))/(L+2)  -- an algebraic identity, threshold crosses
    # 1 exactly at rho = 1/4 (L cancels), so rho < 1/4 => absence needs gamma > 1
    # (impossible) => the cheaper-cycle sector is unavoidable.
    mu = Wstar / L
    rho = mu / (2 * wmax)
    gamma = (m_lightest / m) / mu
    thr_gamma = (L + 1.0 / (2 * rho)) / (L + 2)
    rho_law = dict(mu=mu, rho=rho, gamma=gamma, threshold_gamma=thr_gamma,
                   predicted_absent=bool(gamma >= thr_gamma))
    if m_lightest >= thr:
        return dict(cls="absent", L=L, m=m, threshold=thr, m_lightest=m_lightest,
                    rho_law=rho_law)
    wcyc = _min_cycle_set(_light_cycles(n, adj, wl), m, thr)
    if math.isfinite(wcyc) and wcyc < thr:
        gap_c4 = -4.0 * (1 + 2 * (Wstar - wcyc) / B)
        return dict(cls="present", L=L, m=m, threshold=thr, W_cyc=wcyc,
                    gap_below_path_c4=gap_c4, rho_law=rho_law)
    return dict(cls="undetermined", L=L, m=m, threshold=thr, m_lightest=m_lightest,
                best_cycle_set=(wcyc if math.isfinite(wcyc) else None), rho_law=rho_law)


def run(argv=None) -> dict:
    # ---- sanity: the reference instance must come back PRESENT ----
    n, e, w = random_graph(60, 0.05, seed=19)
    s, t, path = _terminals(n, e, w)
    ref = classify(n, e, w, s, t, path)
    assert ref["cls"] == "present", f"reference not present: {ref}"
    assert abs(ref["W_cyc"] - 1.2239) < 0.05, ref["W_cyc"]
    assert abs(ref["gap_below_path_c4"] - (-1.281)) < 0.02, ref["gap_below_path_c4"]

    # CYC-ENERGY: the two absolute compiled-network energy anchors (path vs cheapest
    # cycle) -- computed on the reference BEFORE the family loop reassigns n,e,w,path.
    ref_energy = _reference_energies(n, e, w, s, t, path)
    assert abs(ref_energy["path_energy_c4"] - 6.2101344) < 1e-3, ref_energy
    assert abs(ref_energy["cycle_energy_c4"] - 4.9291904) < 2e-3, ref_energy

    by_family = {}
    all_rows = []
    for fam in FAM_ORDER:
        rows = []
        for size, seed in SUBSAMPLE[fam]:
            n, e, w = _build(fam, size, seed)
            s, t, path = _terminals(n, e, w)
            if path is None:
                continue
            r = classify(n, e, w, s, t, path)
            r["family"] = fam
            rows.append(r)
        all_rows.extend(rows)
        cnt = {"present": 0, "absent": 0, "undetermined": 0}
        gaps = []
        for r in rows:
            cnt[r["cls"]] += 1
            if r["cls"] == "present":
                gaps.append(r["gap_below_path_c4"])
        by_family[fam] = dict(
            n=len(rows), present=cnt["present"], absent=cnt["absent"],
            undetermined=cnt["undetermined"],
            present_frac=cnt["present"] / len(rows) if rows else None,
            present_gap_c4_median=(float(np.median(gaps)) if gaps else None),
            present_gap_c4_max=(float(np.min(gaps)) if gaps else None))  # most negative

    tot = {k: sum(by_family[f][k] for f in FAM_ORDER)
           for k in ("n", "present", "absent", "undetermined")}

    # rho-law: the absence certificate rewritten as gamma >= threshold(rho)
    identity_ok = all(r["rho_law"]["predicted_absent"] == (r["cls"] == "absent")
                      for r in all_rows)
    confusion = defaultdict(int)
    for r in all_rows:
        confusion[(r["rho_law"]["predicted_absent"], r["cls"])] += 1
    n_gamma_gt1 = sum(1 for r in all_rows if r["rho_law"]["gamma"] > 1 + 1e-9)
    rho_law = dict(
        identity_predict_eq_certificate=identity_ok,
        confusion={f"pred_absent={p}|{c}": n for (p, c), n in sorted(confusion.items())},
        n_gamma_gt_1=n_gamma_gt1, n_total=len(all_rows),
        rho_quarter_boundary=("threshold=[L+1/(2rho)]/(L+2)=1 iff rho=1/4 (L cancels); "
                              "rho<1/4 -> absence needs gamma>1 (impossible) -> the "
                              "cheaper-cycle sector is unavoidable; rho>=1/4 -> possible"),
        by_family_gamma={f: [round(min(r["rho_law"]["gamma"] for r in all_rows if r["family"] == f), 3),
                             round(max(r["rho_law"]["gamma"] for r in all_rows if r["family"] == f), 3)]
                         for f in FAM_ORDER},
        by_family_rho={f: [round(min(r["rho_law"]["rho"] for r in all_rows if r["family"] == f), 3),
                           round(max(r["rho_law"]["rho"] for r in all_rows if r["family"] == f), 3)]
                       for f in FAM_ORDER})

    result = dict(
        threshold_rule="a disjoint-cycle set on exactly m=L+2 edges with total weight < "
                       "w_max + W* is cheaper than the path (gap = -c[1+2(W*-W_cyc)/B] "
                       "grows linearly in c)",
        reference_sanity=ref,
        reference_energies=ref_energy,
        by_family=by_family, pooled=tot, rho_law=rho_law,
        note="ABSENT/PRESENT are rigorous certificates; UNDETERMINED is neither (a "
             "heuristic miss is not a proof of absence) and is NOT imputed.")
    out = write_json("counts/cycle_prevalence.json", result)

    print(f"[19] cycle_prevalence -> {rel(out)}  (subsample per family; two-sided certs)")
    print(f"    reference: PRESENT, W_cyc={ref['W_cyc']:.4f}, gap@c4={ref['gap_below_path_c4']:.3f}  OK")
    print(f"    CYC-ENERGY: path E_N={ref_energy['path_energy_c4']:.4f} "
          f"(analytic {ref_energy['path_energy_analytic']:.4f}) vs cycle E_N="
          f"{ref_energy['cycle_energy_c4']:.4f}; gap={ref_energy['energy_gap_c4']:.3f}  OK")
    print(f"    {'family':<8} {'n':>4} {'present':>8} {'absent':>7} {'undet':>6} "
          f"{'pres%':>6} {'gap@c4 med':>10}")
    for fam in FAM_ORDER:
        b = by_family[fam]
        gm = f"{b['present_gap_c4_median']:.2f}" if b["present_gap_c4_median"] is not None else "-"
        print(f"    {fam:<8} {b['n']:>4} {b['present']:>8} {b['absent']:>7} "
              f"{b['undetermined']:>6} {100*b['present_frac']:>5.0f}% {gm:>10}")
    print(f"    POOLED  n={tot['n']}  present={tot['present']} absent={tot['absent']} "
          f"undetermined={tot['undetermined']}")
    print(f"    rho-law: identity(pred==cert)={rho_law['identity_predict_eq_certificate']}; "
          f"gamma>1 in {rho_law['n_gamma_gt_1']}/{rho_law['n_total']}; boundary rho=1/4 (L-independent)")
    for k, v in rho_law["confusion"].items():
        print(f"      {k}: {v}")
    return result


if __name__ == "__main__":
    run()
