"""Reference-instance regression guard.

The whole paper is pinned to one graph, ``random_graph(60, 0.05, seed=19)``, and
the data deposit ships SEEDS plus the generator (not serialized graphs), so a
future change to the generator that moved the graph produced at a given seed would
silently invalidate every result. This check fails loudly if that happens:

    random_graph(60, 0.05, seed=19)  ->  |E| = 142, an 8-hop shortest-path optimum.

Runs in CI (via ``python -m verify``), so any generator change that shifts the
reference instance breaks the build rather than the paper.
"""

from __future__ import annotations

from spnn.graphs import dijkstra, random_graph


def run(argv=None) -> dict:
    n, edges, w = random_graph(60, 0.05, seed=19)
    path, cost = dijkstra(n, edges, w, 0, n - 1)
    n_edges = int(edges.shape[0])
    hops = len(path) - 1
    ok_e = n_edges == 142
    ok_l = hops == 8
    print("[check instance] reference random_graph(60, 0.05, seed=19)")
    print(f"  |E| = {n_edges} (want 142)  {'OK' if ok_e else 'FAIL'}; "
          f"optimum {hops} hops (want 8)  {'OK' if ok_l else 'FAIL'}")
    if not (ok_e and ok_l):
        raise AssertionError(
            f"reference instance moved: |E|={n_edges} (want 142), "
            f"L={hops} (want 8) -- the generator changed; results are invalidated")
    return {"n_edges": n_edges, "optimum_hops": hops, "ok": True}


if __name__ == "__main__":
    run()
