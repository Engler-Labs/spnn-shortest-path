"""Experiment 19 -- prevalence of the d=-1 (disjoint-cycle) sector (section V-D).

Claims: CYC-PREV (CLAIMS.tsv, section V-D).
Output: results/counts/cycle_prevalence.json

The d=-1 sector (Exp 18 / channel #229): a set of vertex-disjoint cycles on exactly
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

from spnn.graphs import dijkstra, grid_graph, mountain_spatial_graph, random_graph

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
    if m_lightest >= thr:
        return dict(cls="absent", L=L, m=m, threshold=thr, m_lightest=m_lightest)
    wcyc = _min_cycle_set(_light_cycles(n, adj, wl), m, thr)
    if math.isfinite(wcyc) and wcyc < thr:
        gap_c4 = -4.0 * (1 + 2 * (Wstar - wcyc) / B)
        return dict(cls="present", L=L, m=m, threshold=thr, W_cyc=wcyc,
                    gap_below_path_c4=gap_c4)
    return dict(cls="undetermined", L=L, m=m, threshold=thr, m_lightest=m_lightest,
                best_cycle_set=(wcyc if math.isfinite(wcyc) else None))


def run(argv=None) -> dict:
    # ---- sanity: the reference instance must come back PRESENT ----
    n, e, w = random_graph(60, 0.05, seed=19)
    s, t, path = _terminals(n, e, w)
    ref = classify(n, e, w, s, t, path)
    assert ref["cls"] == "present", f"reference not present: {ref}"
    assert abs(ref["W_cyc"] - 1.2239) < 0.05, ref["W_cyc"]
    assert abs(ref["gap_below_path_c4"] - (-1.281)) < 0.02, ref["gap_below_path_c4"]

    by_family = {}
    for fam in FAM_ORDER:
        rows = []
        for size, seed in SUBSAMPLE[fam]:
            n, e, w = _build(fam, size, seed)
            s, t, path = _terminals(n, e, w)
            if path is None:
                continue
            rows.append(classify(n, e, w, s, t, path))
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
    result = dict(
        threshold_rule="a disjoint-cycle set on exactly m=L+2 edges with total weight < "
                       "w_max + W* is cheaper than the path (gap = -c[1+2(W*-W_cyc)/B] "
                       "grows linearly in c)",
        reference_sanity=ref,
        by_family=by_family, pooled=tot,
        note="ABSENT/PRESENT are rigorous certificates; UNDETERMINED is neither (a "
             "heuristic miss is not a proof of absence) and is NOT imputed.")
    out = write_json("counts/cycle_prevalence.json", result)

    print(f"[19] cycle_prevalence -> {rel(out)}  (subsample per family; two-sided certs)")
    print(f"    reference: PRESENT, W_cyc={ref['W_cyc']:.4f}, gap@c4={ref['gap_below_path_c4']:.3f}  OK")
    print(f"    {'family':<8} {'n':>4} {'present':>8} {'absent':>7} {'undet':>6} "
          f"{'pres%':>6} {'gap@c4 med':>10}")
    for fam in FAM_ORDER:
        b = by_family[fam]
        gm = f"{b['present_gap_c4_median']:.2f}" if b["present_gap_c4_median"] is not None else "-"
        print(f"    {fam:<8} {b['n']:>4} {b['present']:>8} {b['absent']:>7} "
              f"{b['undetermined']:>6} {100*b['present_frac']:>5.0f}% {gm:>10}")
    print(f"    POOLED  n={tot['n']}  present={tot['present']} absent={tot['absent']} "
          f"undetermined={tot['undetermined']}")
    return result


if __name__ == "__main__":
    run()
