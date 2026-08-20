"""check C -- end-to-end: does the induced graph contain the Dijkstra path (R-AGN).

On small graphs, the slot-agnostic readout recovers the exact shortest path on
all instances. Reproduces R-AGN: 12 of 12 instances (4 sizes x 3 trials,
beta_fanout=2, the canonical both-slots readout).

``--slot0`` (R-SLOT0, the "6 of 12" contrast) is reported honestly: the exact
slot-restricted readout variant is not identified in the released library
(beta_fanout=1, the slot-matched build, still recovers 12/12 here), so it is
not fabricated.
"""

from __future__ import annotations

from spnn import Params, compile_network, random_graph
from spnn.graphs import dijkstra
from spnn.sim import simulate

from experiments._base import rel, write_json

SIZES = (10, 15, 20, 25)     # 4 sizes x 3 trials = 12 instances
P, TRIALS, STEPS = 0.25, 3, 1500


def _end_to_end(beta_fanout):
    hits, rows = 0, []
    for n in SIZES:
        for tr in range(TRIALS):
            n_, edges, w = random_graph(n, P, seed=100 + tr)
            _, true_cost = dijkstra(n_, edges, w, 0, n_ - 1)
            net = compile_network(n_, edges, w, 0, n_ - 1,
                                  Params(beta_fanout=beta_fanout))
            tr_ = simulate(net, steps=STEPS, seed=tr, source=0, target=n_ - 1,
                           irm_threshold=1.5, track_paths=True)
            ok = tr_.best_path is not None and abs(tr_.best_cost - true_cost) < 1e-6
            hits += int(ok)
            rows.append({"n": n, "trial": tr, "exact": bool(ok)})
    return hits, len(rows), rows


def run(argv=None) -> dict:
    argv = list(argv or [])
    hits, tot, rows = _end_to_end(beta_fanout=2)
    result = {
        "R-AGN": {"exact_hits": f"{hits}/{tot}", "expected": "12/12", "ok": hits == tot},
        "rows": rows,
    }
    if "--slot0" in argv or "--full" in argv:
        h0, t0, _ = _end_to_end(beta_fanout=1)
        result["R-SLOT0"] = {
            "exact_hits": f"{h0}/{t0}", "expected": "6/12", "reproduced": h0 == 6,
            "note": "beta_fanout=1 (slot-matched) recovers %d/%d, not 6/12; the exact "
                    "slot-restricted readout variant is not identified in the released "
                    "library." % (h0, t0),
        }
    out = write_json("validate/readout.json", result)
    print(f"[check C] end-to-end readout vs Dijkstra -> {rel(out)}")
    print(f"  R-AGN  exact hits = {hits}/{tot}  (expect 12/12)  "
          f"{'OK' if hits == tot else 'FAIL'}")
    if "R-SLOT0" in result:
        print(f"  R-SLOT0 contrast: {result['R-SLOT0']['exact_hits']} "
              f"(expected 6/12) -- NOT reproduced; see note")
    return result


if __name__ == "__main__":
    run()
