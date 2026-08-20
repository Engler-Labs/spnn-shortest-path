"""Yen's algorithm -- the K shortest LOOPLESS (simple) s->t paths, exact.

A vendored, dependency-free reference implementation (stdlib only). It underlies
the *optimality gap*: with ``K=2`` the second returned path is the best distinct
simple s->t path other than the optimum, over all hop counts -- the rival the
stochastic spiking network must be able to tell the shortest path apart from. The
tolerance results (CLAIMS.tsv ``TOL-*``) are computed from this, e.g. at K=4000.

Provenance
----------
Extracted verbatim from an internal reference harness. A near-duplicate copy in
that source had drifted only in that it rebuilt the ``wmap`` weight table inside
the candidate loop (slower, byte-identical output); this module takes the
canonical build-once form. Because it is a copy, it must be CROSS-CHECKED, not
trusted -- run ``python -m spnn.yen`` (or the ``verify/`` DFS check) to compare
it against exhaustive enumeration.

Public API
----------
``k_shortest_paths(n, edges, w, s, t, K=2)`` -> list of ``(path, weight)`` ascending
``path_weight(wmap, path)`` -> float
"""

from __future__ import annotations

import heapq
import math


def path_weight(wmap, path):
    return float(sum(wmap[(path[i], path[i + 1])] for i in range(len(path) - 1)))


def _dijkstra_banned(n, adj, s, t, banned_edges, banned_nodes):
    """Dijkstra avoiding a set of edges and nodes (Yen's inner call)."""
    dist = {s: 0.0}
    prev = {}
    pq = [(0.0, s)]
    seen = set()
    while pq:
        d, v = heapq.heappop(pq)
        if v in seen:
            continue
        seen.add(v)
        if v == t:
            path = [v]
            while path[-1] != s:
                path.append(prev[path[-1]])
            return path[::-1], d
        for u, ww in adj[v]:
            if u in banned_nodes or (v, u) in banned_edges:
                continue
            nd = d + ww
            if nd < dist.get(u, math.inf):
                dist[u] = nd
                prev[u] = v
                heapq.heappush(pq, (nd, u))
    return None, math.inf


def k_shortest_paths(n, edges, w, s, t, K=2):
    """Yen's algorithm -- the K shortest LOOPLESS s->t paths, exact.

    Returns a list of (path, weight), ascending.  The second entry is the best DISTINCT
    simple path other than the optimum, over ALL hop counts (not just the optimum's own
    depth) -- that is the rival the network has to be able to tell the optimum from.
    """
    adj = {i: [] for i in range(n)}
    for (a, b), ww in zip(edges, w):
        a, b, ww = int(a), int(b), float(ww)
        adj[a].append((b, ww))
        adj[b].append((a, ww))
    p0, d0 = _dijkstra_banned(n, adj, s, t, set(), set())
    if p0 is None:
        return []
    A = [(p0, d0)]
    B = []
    seen = {tuple(p0)}
    wmap = {}
    for (a, b), ww in zip(edges, w):
        wmap[(int(a), int(b))] = wmap[(int(b), int(a))] = float(ww)
    while len(A) < K:
        prev_path = A[-1][0]
        for i in range(len(prev_path) - 1):
            spur = prev_path[i]
            root = prev_path[:i + 1]
            banned_edges = set()
            for pth, _ in A:
                if pth[:i + 1] == root and len(pth) > i + 1:
                    banned_edges.add((pth[i], pth[i + 1]))
                    banned_edges.add((pth[i + 1], pth[i]))
            banned_nodes = set(root[:-1])
            sp, _ = _dijkstra_banned(n, adj, spur, t, banned_edges, banned_nodes)
            if sp is None:
                continue
            total = root[:-1] + sp
            if tuple(total) in seen:
                continue
            B.append((path_weight(wmap, total), total))
        if not B:
            break
        B.sort(key=lambda x: (x[0], x[1]))
        cost, best = B.pop(0)
        if tuple(best) in seen:
            continue
        seen.add(tuple(best))
        A.append((best, cost))
    return A


# --------------------------------------------------------------------------- #
# Self-contained cross-check: exhaustive DFS over all simple s->t paths.
# The vendored copy is never trusted -- it is compared against ground truth.
# Kept dependency-free (stdlib only) so the module proves itself in isolation.
# --------------------------------------------------------------------------- #
def _two_lightest_simple_paths(n, edges, w, s, t):
    """Exhaustive DFS over every simple s->t path; return the two lightest as
    ``[(weight, path_tuple), ...]``. Ground-truth oracle for small graphs only."""
    adj = {i: [] for i in range(n)}
    for (a, b), ww in zip(edges, w):
        adj[int(a)].append((int(b), float(ww)))
        adj[int(b)].append((int(a), float(ww)))
    found, on, st = [], [False] * n, [s]
    on[s] = True

    def rec(v, ws):
        if v == t:
            found.append((ws, tuple(st)))
            return
        for u, ww in adj[v]:
            if on[u]:
                continue
            on[u] = True
            st.append(u)
            rec(u, ws + ww)
            st.pop()
            on[u] = False

    rec(s, 0.0)
    found.sort()
    return found[:2]


def _random_graph(n, p, seed):
    """Tiny stdlib random weighted graph (spanning path forces connectivity)."""
    import random
    rng = random.Random(seed)
    edges, seen = [], set()
    perm = list(range(n))
    rng.shuffle(perm)
    for a, b in zip(perm[:-1], perm[1:]):
        e = (min(a, b), max(a, b))
        edges.append(e)
        seen.add(e)
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) not in seen and rng.random() < p:
                edges.append((a, b))
                seen.add((a, b))
    w = [rng.random() for _ in edges]
    return n, edges, w


def _selfcheck():
    """Compare Yen K=2 against exhaustive DFS on small graphs; raise on mismatch."""
    bad = tot = 0
    for nn, p in ((12, 0.3), (14, 0.25), (16, 0.22), (10, 0.4)):
        for seed in range(1, 16):
            n, e, w = _random_graph(nn, p, seed)
            bf = _two_lightest_simple_paths(n, e, w, 0, nn - 1)
            if len(bf) < 2:
                continue
            yen = k_shortest_paths(n, e, w, 0, nn - 1, K=2)
            tot += 1
            if not (len(yen) == 2
                    and abs(yen[0][1] - bf[0][0]) < 1e-9
                    and abs(yen[1][1] - bf[1][0]) < 1e-9):
                bad += 1
    print(f"Yen cross-check vs exhaustive DFS: {tot - bad}/{tot} agree, {bad} mismatches")
    if bad:
        raise SystemExit("Yen K=2 disagrees with exhaustive enumeration")


if __name__ == "__main__":
    _selfcheck()
