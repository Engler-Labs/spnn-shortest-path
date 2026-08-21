"""Graph generators matching Ch. 4 sec. 4.3 and Ch. 5 Fig. 5.2."""

from __future__ import annotations

import heapq

import numpy as np


def random_graph(n: int, p: float, seed: int = 0, wlo=0.0, whi=1.0):
    """G(n, p) with weights uniform on (wlo, whi]; connected, no isolates."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, 1)
    keep = rng.random(iu[0].size) < p
    edges = np.stack([iu[0][keep], iu[1][keep]], axis=1).astype(np.int32)

    # force connectivity with a random spanning path
    perm = rng.permutation(n)
    spine = np.stack([perm[:-1], perm[1:]], axis=1)
    spine = np.sort(spine, axis=1).astype(np.int32)
    edges = np.unique(np.vstack([edges, spine]), axis=0)

    w = rng.uniform(wlo, whi, size=edges.shape[0]).astype(np.float32)
    w = np.maximum(w, 1e-3)
    return n, edges, w


def spatial_graph(n: int, radius: float, seed: int = 0):
    """Vertices uniform in the unit square, edges within `radius`,
    weights = Euclidean distance (Ch. 5 spatial graph (i))."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2))
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    iu = np.triu_indices(n, 1)
    keep = d[iu] <= radius
    edges = np.stack([iu[0][keep], iu[1][keep]], axis=1).astype(np.int32)

    perm = rng.permutation(n)
    spine = np.sort(np.stack([perm[:-1], perm[1:]], axis=1), axis=1).astype(np.int32)
    edges = np.unique(np.vstack([edges, spine]), axis=0)
    w = d[edges[:, 0], edges[:, 1]].astype(np.float32)
    return n, edges, np.maximum(w, 1e-3), pos


def grid_graph(k: int, seed: int = 0, jitter: float = 0.0):
    """City-streets: a k x k lattice (4-neighbour), source/target opposite corners.

    jitter=0  -> every edge weight 1: ALL monotone corner-to-corner routes are optimal
                 (C(2k-2,k-1) of them) -- a pure tie lattice.
    jitter>0  -> weights 1 + U(0,jitter): near-ties, one optimum among many close ones.
    Returns (n, edges, w) with source=0 and target=n-1 (opposite corners).
    """
    rng = np.random.default_rng(seed)
    n = k * k
    def vid(i, j):
        return i * k + j
    e = []
    for i in range(k):
        for j in range(k):
            if j + 1 < k:
                e.append((vid(i, j), vid(i, j + 1)))
            if i + 1 < k:
                e.append((vid(i, j), vid(i + 1, j)))
    edges = np.array(sorted(e), dtype=np.int32)
    w = (1.0 + rng.uniform(0.0, jitter, size=edges.shape[0])).astype(np.float32)
    return n, edges, w


def mountain_graph(k: int, seed: int = 0, n_peaks: int = 4, relief: float = 6.0):
    """Terrain: a k x k lattice whose edge cost is horizontal step + climb over a smooth
    elevation surface (sum of Gaussian peaks).  Shortest paths follow valleys / passes
    around the peaks -- spatially-correlated weights, unlike G(n,p).  Source/target are
    opposite corners.  Returns (n, edges, w).
    """
    rng = np.random.default_rng(seed)
    n = k * k
    xs, ys = np.meshgrid(np.linspace(0, 1, k), np.linspace(0, 1, k), indexing="ij")
    elev = np.zeros((k, k))
    for _ in range(n_peaks):
        cx, cy = rng.uniform(0.15, 0.85, size=2)
        width = rng.uniform(0.10, 0.22)
        elev += rng.uniform(0.6, 1.0) * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * width ** 2))
    def vid(i, j):
        return i * k + j
    e, w = [], []
    for i in range(k):
        for j in range(k):
            for di, dj in ((0, 1), (1, 0)):
                ni, nj = i + di, j + dj
                if ni < k and nj < k:
                    e.append((vid(i, j), vid(ni, nj)))
                    w.append(1.0 + relief * abs(float(elev[ni, nj] - elev[i, j])))
    order = np.argsort([a * n + b for a, b in e])
    edges = np.array(e, dtype=np.int32)[order]
    weights = np.array(w, dtype=np.float32)[order]
    return n, edges, weights


def _terrain_peaks(rng, n_peaks):
    """Random Gaussian peaks (cx, cy, width, amp) -- the shared 'mountain' terrain."""
    return [(float(cx), float(cy), float(rng.uniform(0.10, 0.22)), float(rng.uniform(0.6, 1.0)))
            for cx, cy in rng.uniform(0.15, 0.85, size=(n_peaks, 2))]


def _elevation(x, y, peaks):
    """Elevation at point(s) (x, y) = sum of the Gaussian peaks (arrays ok)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.zeros(np.broadcast(x, y).shape)
    for cx, cy, width, amp in peaks:
        z = z + amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width ** 2))
    return z


def mountain_spatial_graph(n: int, seed: int = 0, n_peaks: int = 4, relief: float = 6.0,
                           k: int = 4):
    """Same Gaussian-peak terrain as ``mountain_graph`` but with nodes scattered RANDOMLY
    in the unit square (like ``spatial_graph``) instead of on a lattice.

    Fixed-degree **k-nearest-neighbour** construction: each node links to its ``k`` nearest
    others (symmetrised to an undirected graph), plus the Euclidean minimum-spanning-tree
    edges to guarantee a single connected component.  This holds mean degree roughly
    constant in ``n`` (unlike a global radius, which over-connects dense regions as ``n``
    grows) and gives every node degree >= k.  Edge cost = horizontal Euclidean distance +
    climb over the terrain (``relief * |Delta elevation|``).  Source/target are pinned to
    opposite corners so the path crosses the landscape.  Returns (n, edges, w).
    """
    from scipy.sparse.csgraph import minimum_spanning_tree

    rng = np.random.default_rng(seed)
    peaks = _terrain_peaks(rng, n_peaks)
    pos = rng.random((n, 2))
    pos[0] = (0.05, 0.05)      # source corner
    pos[-1] = (0.95, 0.95)     # target corner
    elev = _elevation(pos[:, 0], pos[:, 1], peaks)

    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    kk = min(k, n - 1)
    nbr = np.argsort(d, axis=1)[:, 1:kk + 1]            # k nearest (col 0 is self)
    pairs = {(min(i, int(j)), max(i, int(j)))           # symmetrise -> undirected
             for i in range(n) for j in nbr[i]}
    mst = minimum_spanning_tree(d).tocoo()              # guarantee connectivity
    pairs.update((min(int(a), int(b)), max(int(a), int(b)))
                 for a, b in zip(mst.row, mst.col))

    edges = np.array(sorted(pairs), dtype=np.int32)
    a, b = edges[:, 0], edges[:, 1]
    w = (d[a, b] + relief * np.abs(elev[a] - elev[b])).astype(np.float32)
    return n, edges, np.maximum(w, 1e-3)


def min_weight_cycle(n, edges, w):
    """Weighted girth: the minimum total weight over all cycles of G, or inf if G is
    acyclic (undirected, positive weights).

    For each edge e = (u, v) with weight w_e, the lightest cycle through e is the
    shortest u->v path in G with e removed, plus w_e; the girth is the minimum over
    all edges.  With positive weights the shortest path is simple and avoids e, so
    path + e is a simple cycle (>= 3 edges for a simple graph).  Returns
    ``(weight, (u, v))`` -- the girth and one witness edge on the lightest cycle.
    """
    adj: dict[int, list] = {i: [] for i in range(n)}
    for i, (a, b) in enumerate(edges):
        a, b = int(a), int(b)
        adj[a].append((b, float(w[i]), i))
        adj[b].append((a, float(w[i]), i))

    def _sp(src, dst, ban):
        dist = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            d, v = heapq.heappop(pq)
            if v == dst:
                return d
            if d > dist.get(v, np.inf):
                continue
            for nb, ww, ei in adj[v]:
                if ei == ban:
                    continue
                nd = d + ww
                if nd < dist.get(nb, np.inf):
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb))
        return float(np.inf)

    best, witness = float(np.inf), None
    for i, (a, b) in enumerate(edges):
        cyc = _sp(int(a), int(b), i) + float(w[i])
        if cyc < best:
            best, witness = cyc, (int(a), int(b))
    return best, witness


def dijkstra(n, edges, w, source, target):
    adj: dict[int, list] = {i: [] for i in range(n)}
    for (a, b), ww in zip(edges, w):
        adj[int(a)].append((int(b), float(ww)))
        adj[int(b)].append((int(a), float(ww)))
    dist = {source: 0.0}
    prev: dict[int, int] = {}
    pq = [(0.0, source)]
    seen = set()
    while pq:
        d, v = heapq.heappop(pq)
        if v in seen:
            continue
        seen.add(v)
        if v == target:
            path = [v]
            while path[-1] != source:
                path.append(prev[path[-1]])
            return path[::-1], d
        for nb, ww in adj[v]:
            nd = d + ww
            if nd < dist.get(nb, np.inf):
                dist[nb] = nd
                prev[nb] = v
                heapq.heappush(pq, (nd, nb))
    return None, np.inf
