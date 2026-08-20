"""Experiment 5 -- constructed variant orderings (the boundary-form consequence).

Claims: VAR-REF, VAR-EQWT, VAR-INV, VAR-INV4 (CLAIMS.tsv, section IV-E).
Outputs: results/variant/reference.json    (VAR-REF)
         results/variant/constructed.json  (VAR-EQWT, VAR-INV, VAR-INV4)

On the VARIANT (unaugmented) construction the interior edges of a path state are
charged twice, so E_SPC(x_P) follows the boundary form (c/B)(2*sum w - w_first -
w_last) rather than the plain (c/B)*sum w. The consequence is that the network's
energy minimum over the path sector is NOT the least-weight path: an edge's charge
depends on its POSITION, so weight moved onto interior edges is penalised.

Everything here is hand-built and read back through the compiler's own
``sim.energy`` -- nothing is sampled. The two constructed 4-hop pairs live on one
8-vertex graph with two vertex-disjoint s->t routes of L=4 (0-1-2-3-7 and
0-4-5-6-7), so E_NC = -lambda*L + c is identical across the pair and the entire
energy difference is E_SPC:

  VAR-EQWT  equal total weight (10.0), terminals (1,4,4,1) vs (4,1,1,4):
            E_SPC 2.250000 / 1.500000, E_N -0.750000 / -1.500000 -- a 0.75 gap
            where the plain form predicts zero.
  VAR-INV   the lighter path loses: (0.1,4,4,0.1) w=8.2 (Dijkstra's answer) vs
            (5,0.1,0.1,5) w=10.2, E_N -1.380000 vs -1.960000 -- the strictly
            HEAVIER path is preferred by 0.58.
  VAR-INV4  the same inversion at the c=4 operating point: it scales linearly in c,
            so the gap is 2.32 (the abstract's 24 %).

VAR-REF is the reference instance on the same variant construction: c=1, lambda=+1,
E_NC = -7.000000, E_SPC = 0.510616, Etilde = 0.276267.

All constructed energies are taken at c=1, lambda=+1 (b_k = (lambda - D - c)/2 = -2,
D=4), which fixes E_NC = -lambda*L + c = -3 on the L=4 pairs; VAR-INV4 rescales c
to 4 (b_k = -3.5), holding lambda = +1.
"""

from __future__ import annotations

import numpy as np

from spnn.compile import Params, compile_network
from spnn.graphs import dijkstra, random_graph
from spnn.sim import energy, ideal_state

from experiments._base import rel, write_json

REF_N, REF_P, REF_SEED = 60, 0.05, 19
REF_S, REF_T = 0, 59
D = 4.0  # beta / drive strength (spnn default)


def two_path_graph(w1, w2):
    """8 vertices, two vertex-disjoint s->t routes of L=4.

    P1 = 0-1-2-3-7 with weights w1, P2 = 0-4-5-6-7 with weights w2.
    """
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 7],
                      [0, 4], [4, 5], [5, 6], [6, 7]], dtype=np.int32)
    weights = np.array(list(w1) + list(w2), dtype=np.float32)
    return 8, edges, weights, [0, 1, 2, 3, 7], [0, 4, 5, 6, 7]


def _energy_of(net, path):
    x = ideal_state(net, net.augmented_path(path))
    e_nc, e_spc = energy(net, x)
    return float(e_nc), float(e_spc), float(e_nc + e_spc)


def _pair(name, w1, w2, c, lam=1.0):
    """Energies of both routes of a constructed pair on the unaugmented network."""
    n, edges, weights, P1, P2 = two_path_graph(w1, w2)
    b_k = (lam - D - c) / 2.0
    net = compile_network(n, edges, weights, 0, 7,
                          params=Params(D=D, alpha_scale=c, bias=b_k),
                          augment_st=False)
    e1_nc, e1_spc, e1 = _energy_of(net, P1)
    e2_nc, e2_spc, e2 = _energy_of(net, P2)
    return {
        "name": name,
        "params": {"c": c, "lambda": lam, "D": D, "b_k": b_k},
        "B": float(net.params.B),
        "P1": {"weights": list(w1), "sum_w": float(sum(w1)),
               "E_NC": e1_nc, "E_SPC": e1_spc, "E_N": e1},
        "P2": {"weights": list(w2), "sum_w": float(sum(w2)),
               "E_NC": e2_nc, "E_SPC": e2_spc, "E_N": e2},
        "gap_E_N_P1_minus_P2": e1 - e2,
    }


def reference():
    """VAR-REF: the reference instance on the variant construction (c=1, lambda=+1)."""
    n, edges, w = random_graph(REF_N, REF_P, seed=REF_SEED)
    path, cost = dijkstra(n, edges, w, REF_S, REF_T)
    L = len(path) - 1
    B = 2.0 * float(np.max(w))
    eidx = {}
    for i in range(len(edges)):
        u, v = int(edges[i][0]), int(edges[i][1])
        eidx[(u, v)] = eidx[(v, u)] = float(w[i])
    wsum = sum(eidx[(path[i], path[i + 1])] for i in range(L))
    Etilde = wsum / B

    c, lam = 1.0, 1.0
    b_k = (lam - D - c) / 2.0
    net = compile_network(n, edges, w, REF_S, REF_T,
                          params=Params(D=D, alpha_scale=c, bias=b_k), augment_st=False)
    e_nc, e_spc = energy(net, ideal_state(net, net.augmented_path(path)))
    return {
        "instance": {
            "generator": f"random_graph({REF_N}, {REF_P}, seed={REF_SEED})",
            "source": REF_S, "target": REF_T, "L": L, "path_weight": wsum,
            "B": B, "Etilde": Etilde,
        },
        "params": {"c": c, "lambda": lam, "D": D, "b_k": b_k, "augment_st": False},
        "E_NC": float(e_nc),
        "E_SPC": float(e_spc),
        "E_N": float(e_nc + e_spc),
    }


def run(argv=None) -> dict:
    # ---- VAR-REF ----
    ref = reference()
    ref_out = write_json("variant/reference.json", ref)

    # ---- VAR-EQWT / VAR-INV / VAR-INV4 ----
    eqwt = _pair("equal_weight", (1.0, 4.0, 4.0, 1.0), (4.0, 1.0, 1.0, 4.0), c=1.0)
    inv = _pair("inversion", (0.1, 4.0, 4.0, 0.1), (5.0, 0.1, 0.1, 5.0), c=1.0)
    inv4 = _pair("inversion_c4", (0.1, 4.0, 4.0, 0.1), (5.0, 0.1, 0.1, 5.0), c=4.0)
    constructed = {
        "equal_weight": eqwt,        # VAR-EQWT
        "inversion": inv,            # VAR-INV
        "inversion_c4": inv4,        # VAR-INV4
        "note": ("P1 charges its weight on interior edges (charged twice under the "
                 "boundary form); equal-weight pairs split by 0.75 and the lighter "
                 "path loses the inversion by 0.58 (2.32 at c=4)."),
    }
    con_out = write_json("variant/constructed.json", constructed)

    heavier_pref_c1 = inv["P1"]["E_N"] - inv["P2"]["E_N"]
    heavier_pref_c4 = inv4["P1"]["E_N"] - inv4["P2"]["E_N"]
    print(f"[5] variant_ordering -> {rel(ref_out)}, {rel(con_out)}")
    print(f"    VAR-REF  E_NC = {ref['E_NC']:.6f}  E_SPC = {ref['E_SPC']:.6f}  "
          f"Etilde = {ref['instance']['Etilde']:.6f}")
    print(f"    VAR-EQWT E_SPC {eqwt['P1']['E_SPC']:.6f} / {eqwt['P2']['E_SPC']:.6f}; "
          f"E_N {eqwt['P1']['E_N']:.6f} / {eqwt['P2']['E_N']:.6f}")
    print(f"    VAR-INV  E_N {inv['P1']['E_N']:.6f} vs {inv['P2']['E_N']:.6f}; "
          f"heavier preferred by {heavier_pref_c1:.6f}")
    print(f"    VAR-INV4 heavier preferred by {heavier_pref_c4:.6f} at c=4")
    return {
        "VAR_REF": {"E_NC": ref["E_NC"], "E_SPC": ref["E_SPC"],
                    "Etilde": ref["instance"]["Etilde"]},
        "VAR_EQWT": {"E_SPC": [eqwt["P1"]["E_SPC"], eqwt["P2"]["E_SPC"]],
                     "E_N": [eqwt["P1"]["E_N"], eqwt["P2"]["E_N"]]},
        "VAR_INV": {"E_N": [inv["P1"]["E_N"], inv["P2"]["E_N"]],
                    "heavier_preferred_by": heavier_pref_c1},
        "VAR_INV4": {"heavier_preferred_by": heavier_pref_c4},
    }


if __name__ == "__main__":
    run()
