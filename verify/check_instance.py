"""Reference-instance regression guard.

The whole paper is pinned to one graph, ``random_graph(60, 0.05, seed=19)``, and
the data deposit ships SEEDS plus the generator (not serialized graphs), so a
future change to the generator that moved the graph produced at a given seed would
silently invalidate every result. This check fails loudly if that happens:

    random_graph(60, 0.05, seed=19)  ->  |E| = 142, an 8-hop shortest-path optimum,
                                         weighted girth W_cyc^min = 0.755323.

The weighted girth is the reference value behind bound (35)/(36) in section V-D
(the d = 0 sector floor): it enters the paper, so it is pinned here and written to
``results/instance/reference.json`` (the REF-GIRTH claim).

Runs in CI (via ``python -m verify``), so any generator change that shifts the
reference instance breaks the build rather than the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

from spnn.graphs import dijkstra, min_weight_cycle, random_graph

REF = dict(n=60, p=0.05, seed=19, source=0)
WANT = dict(n_edges=142, optimum_hops=8, weighted_girth=0.755323)
OUT = Path(__file__).resolve().parent.parent / "results" / "instance" / "reference.json"


def run(argv=None) -> dict:
    n, edges, w = random_graph(REF["n"], REF["p"], seed=REF["seed"])
    path, cost = dijkstra(n, edges, w, REF["source"], n - 1)
    n_edges = int(edges.shape[0])
    hops = len(path) - 1
    girth, girth_edge = min_weight_cycle(n, edges, w)

    ok_e = n_edges == WANT["n_edges"]
    ok_l = hops == WANT["optimum_hops"]
    ok_g = abs(girth - WANT["weighted_girth"]) < 1e-4
    print("[check instance] reference random_graph(60, 0.05, seed=19)")
    print(f"  |E| = {n_edges} (want 142)  {'OK' if ok_e else 'FAIL'}; "
          f"optimum {hops} hops (want 8)  {'OK' if ok_l else 'FAIL'}")
    print(f"  weighted girth W_cyc^min = {girth:.6f} (want 0.755323)  "
          f"{'OK' if ok_g else 'FAIL'}  via edge {girth_edge}")

    result = {
        "instance": "random_graph(60, 0.05, seed=19)",
        "n_edges": n_edges, "optimum_hops": hops, "opt_cost": float(cost),
        "weighted_girth": float(girth), "girth_witness_edge": list(girth_edge),
        "ok": bool(ok_e and ok_l and ok_g),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    if not result["ok"]:
        raise AssertionError(
            f"reference instance moved: |E|={n_edges} (want 142), L={hops} (want 8), "
            f"girth={girth:.6f} (want 0.755323) -- the generator changed; results "
            f"are invalidated")
    return result


if __name__ == "__main__":
    run()
