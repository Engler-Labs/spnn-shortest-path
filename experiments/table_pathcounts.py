"""Experiment 1 -- tab:pathcounts (one path configuration's energy decomposition).

Claims: T-PATHCOUNTS (CLAIMS.tsv, section IV-A, Tier 1, tolerance ``exact``).
Output: results/tables/pathcounts.tsv

============================================================================
WHAT THIS TABLE IS (and the red herring it is NOT).
============================================================================
tab:pathcounts does **NOT** count s-t paths.  An earlier build read it as a
path-enumeration table, noticed the pinned instance is a simple path with only
ONE s-t path, and flagged it ``needs-diff`` on the worry that a one-path instance
was wrong.  That was a red herring.  The table reports, for **ONE realized path
configuration**:

  * the Lemma-1 structural counts on the path state -- active neurons, beta pairs,
    alpha pairs, gamma pairs;
  * the energy decomposition (E_NC, E_SPC) of that state, both UNAUGMENTED and
    under the Ch.4 S/T AUGMENTATION; and
  * the alpha weight pair-sum  sum-sum_alpha (w_j + w_k).

A single-path instance is exactly what such a table wants -- there is nothing to
enumerate, so "only one s-t path" is not a defect.

The instance (paper caption), pinned exactly
---------------------------------------------
Path 0-1-2-3-4-5 with edge weights 1,2,3,4,5; a pendant at each terminal so
deg(s)=deg(t)=2; B = 2*phi_max = 12; energies at c=1, D=4, b_k=-2.

  * s = 0, t = 5.  A weight-6 pendant is added at each terminal (vertex 6 on 0,
    vertex 7 on 5), making deg(0)=deg(5)=2 and phi_max = 6 so B = 2*phi_max = 12.
    The pendant neurons are INACTIVE in the s-t path state, so the pendant weight
    only sets B (which is also passed explicitly, =12) and never a path energy.
  * Params: D=4, b_k=bias=-2 (== the library default -(D/4+1) at D=4), c=1
    (alpha_scale), B=12.  Then lambda = 2*b_k + D + c = 1, and L = 5.

The five cells verified here
--------------------------------------------------
  E_SPC augmented    2c*Etilde = 2*(15/12)      -> 2.500000
  E_SPC unaugmented  (c/B)*(2*15 - 1 - 5) = 24/12 -> 2.000000
  E_NC  augmented    -lambda*(L+2) + c = -7 + 1  -> -6.000000
  E_NC  unaugmented  -lambda*L + c = -5 + 1       -> -4.000000
  sum-sum_alpha (w_j+w_k)  2*sum(phi) = 2*15      -> 30

plus the Lemma-1 path-state counts on the UNAUGMENTED realized path:
  active = 2L = 10,  beta = L = 5,  alpha = L-1 = 4,  gamma = 0.

*** TERMINAL-BIAS CONVENTION -- READ THIS (a convention dependence). ***
The AUGMENTED E_NC depends on how the two auxiliary terminals S,T are biased,
because E_NC sums the bias over active neurons and the augmented path activates
S and T:

    E_NC_aug(b_st) = -lambda*(L+2) + c + 2*(b_k - b_st)

The target -lambda*(L+2)+c = -6.000000 is the "all-real"-style value in which
every one of the 2(L+2) active neurons carries b_k -- i.e. the terminals INHERIT
b_k (Params(b_st=None)).  Under the ANCHORED convention Params(b_st=0.0) the two
terminals carry 0 instead of b_k=-2, so at this instance E_NC_aug is 2*(b_k-0)=
-4.0 lower: **-10.000000, not -6.000000.**  That 4.0 gap is *exactly* the
discrepancy between the two conventions (2*b_S - 2*b_k = 0 - (-4) = 4.0 at
b_k=-2, b_S=0).  So the earlier note's parenthetical ("Params(b_st=0.0) ... gives
E_NC augmented = -6.0") is a mislabel: the -6.0 derivation is the b_st=None arm.

This module therefore computes the VERIFIED augmented E_NC from the convention
that reproduces the paper's derivation (b_st inherits b_k), and ALSO emits the
anchored b_st=0.0 value (-10.000000) as a labeled companion row, so the table
surfaces the convention dependence instead of burying it.  E_SPC and the
alpha weight-sum are bias-free, hence identical under either convention.

All cells are exact and deterministic on the pinned instance.
"""

from __future__ import annotations

import numpy as np

from spnn.compile import Params, compile_network
from spnn.graphs import dijkstra
from spnn.sim import energy, ideal_state, mass_class, state_counts

from experiments._base import rel, write_tsv

# --- the pinned instance (caption) --------------------------------------------
S, T = 0, 5
N = 8
EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],   # main path, weights 1..5
                  [0, 6], [5, 7]], dtype=np.int32)           # pendant at each terminal
WEIGHTS = np.array([1, 2, 3, 4, 5, 6, 6], dtype=np.float32)  # pendants=6 -> phi_max=6, B=12
PATH = [0, 1, 2, 3, 4, 5]                                    # the realized s-t path
L = 5                                                        # hop length of PATH
W_PATH = 15.0                                                # total weight 1+2+3+4+5

TOL = 1e-6

# c=1, D=4, b_k=-2, B=12 -> lambda = 2*b_k + D + c = 1.
_BASE = dict(alpha_scale=1.0, D=4.0, bias=-2.0, B=12.0)


def _fmt(v, integer: bool) -> str:
    """Deterministic cell text: bare int, else 6-decimal float."""
    return str(int(round(v))) if integer else f"{v:.6f}"


def run(argv=None) -> dict:
    # --- compile the three networks from the SAME graph ----------------------
    un = compile_network(N, EDGES, WEIGHTS, S, T, Params(**_BASE), augment_st=False)
    # augmented, terminals inherit b_k (b_st=None) -> reproduces -lambda*(L+2)+c
    ai = compile_network(N, EDGES, WEIGHTS, S, T, Params(b_st=None, **_BASE),
                         augment_st=True)
    # augmented, anchored terminals (b_st=0.0) -> companion arm
    a0 = compile_network(N, EDGES, WEIGHTS, S, T, Params(b_st=0.0, **_BASE),
                         augment_st=True)

    B = float(un.params.B)
    c = float(un.params.alpha_scale)
    bk = float(un.params.bias)
    D = float(un.params.D)
    lam = 2.0 * bk + D + c

    # --- instance sanity: the caption's conditions actually hold -------------
    deg = np.zeros(N, dtype=int)
    for a, b in EDGES:
        deg[a] += 1
        deg[b] += 1
    assert deg[S] == 2 and deg[T] == 2, f"deg(s)={deg[S]} deg(t)={deg[T]} (want 2,2)"
    assert float(WEIGHTS.max()) == 6.0, "phi_max must be 6 so B = 2*phi_max = 12"
    assert B == 12.0 and lam == 1.0, f"B={B} lambda={lam} (want 12, 1)"
    dpath, dcost = dijkstra(N, EDGES, WEIGHTS, S, T)
    assert list(dpath) == PATH and abs(dcost - W_PATH) < TOL, "PATH is not the s-t optimum"

    # --- realized path states, measured from spnn.sim ------------------------
    x_un = ideal_state(un, PATH)
    active, alpha_p, beta_p, gamma_p = state_counts(un, x_un)
    e_nc_un, e_spc_un = energy(un, x_un)

    apath = ai.augmented_path(PATH)          # [S,0,1,2,3,4,5,T]
    Laug = len(apath) - 1                     # = L + 2 = 7
    x_ai = ideal_state(ai, apath)
    e_nc_ai, e_spc_ai = energy(ai, x_ai)
    # alpha weight pair-sum sum-sum_alpha (w_j+w_k) on the augmented path state.
    # mass_class returns it directly; = 2*sum(phi) since the augmented graph
    # charges every edge from both endpoints (the two 0-weight terminal edges
    # add nothing).  bias-free, so the b_st convention does not enter.
    _, _, _, alpha_wsum_aug = mass_class(ai, x_ai, Laug)

    x_a0 = ideal_state(a0, a0.augmented_path(PATH))
    e_nc_a0, _ = energy(a0, x_a0)             # anchored-terminal companion

    # --- the verified cells: (name, config, derivation, value, expected, int) --
    cells = [
        ("active_neurons", "path(unaug)", "2L",
         float(active), 2.0 * L, True),
        ("beta_pairs", "path(unaug)", "L",
         float(beta_p), float(L), True),
        ("alpha_pairs", "path(unaug)", "L-1",
         float(alpha_p), float(L - 1), True),
        ("gamma_pairs", "path(unaug)", "0",
         float(gamma_p), 0.0, True),
        ("E_NC", "unaugmented", "-lambda*L+c=-5+1",
         float(e_nc_un), -lam * L + c, False),
        ("E_SPC", "unaugmented", "(c/B)*(2*15-1-5)=24/12",
         float(e_spc_un), (c / B) * (2 * W_PATH - 1 - 5), False),
        ("E_NC", "augmented[b_st=b_k]", "-lambda*(L+2)+c=-7+1",
         float(e_nc_ai), -lam * (L + 2) + c, False),
        ("E_SPC", "augmented", "2c*Etilde=2*(15/12)",
         float(e_spc_ai), 2.0 * c * (W_PATH / B), False),
        ("sumsum_alpha_wj+wk", "augmented", "2*sum(phi)=2*15",
         float(alpha_wsum_aug), 2.0 * W_PATH, True),
        # companion -- anchored terminals; exact, informational (not a verified cell)
        ("E_NC", "augmented[b_st=0]", "-lambda*(L+2)+c+2*(b_k-0)=-6-4",
         float(e_nc_a0), -lam * (L + 2) + c + 2.0 * (bk - 0.0), False),
    ]

    # --- assemble + write the table ------------------------------------------
    header = ["cell", "config", "derivation", "value", "expected", "match"]
    rows = []
    checks = {}
    for name, cfg, deriv, val, exp, is_int in cells:
        ok = abs(val - exp) < TOL
        key = f"{name}[{cfg}]"
        checks[key] = ok
        rows.append([name, cfg, deriv, _fmt(val, is_int), _fmt(exp, is_int),
                     "MATCH" if ok else "FAIL"])
    out = write_tsv("tables/pathcounts.tsv", header, rows)

    # The five verified derivations + the four Lemma-1 counts are the deliverable; the
    # b_st=0 companion row is exact too but informational, so it is excluded from
    # the "all cells pass" gate (it has its own MATCH in the table).
    a4_keys = [k for k in checks if k != "E_NC[augmented[b_st=0]]"]
    all_ok = all(checks[k] for k in a4_keys)

    # ============================== summary ==================================
    print(f"[1] table_pathcounts -> {rel(out)}   (counts ONE configuration, "
          f"NOT s-t paths)")
    print(f"    instance: 6-0-1-2-3-4-5-7  weights[main]=1..5 pendants=6  "
          f"B={B:g} c={c:g} b_k={bk:g} D={D:g} lambda={lam:g}  L={L} W={W_PATH:g}")
    print("    --- MATCH/FAIL per cell ---")
    for name, cfg, deriv, val, exp, is_int in cells:
        ok = abs(val - exp) < TOL
        tag = "MATCH" if ok else "FAIL "
        note = "  (companion, informational)" if cfg == "augmented[b_st=0]" else ""
        print(f"      {tag}  {name:<18} {cfg:<20} = {_fmt(val, is_int):>10}  "
              f"(want {_fmt(exp, is_int)}){note}")
    print(f"    A4 cells (5 derivations + 4 Lemma-1 counts): "
          f"{'ALL MATCH' if all_ok else 'FAILURES: ' + str([k for k in a4_keys if not checks[k]])}")
    print(f"    note: augmented E_NC = -6.0 is the b_st=None (inherit b_k) arm; "
          f"anchored b_st=0.0 gives -10.0 (the 4.0 terminal-bias gap).")

    return {
        "status": "built" if all_ok else "FAIL",
        "claim": "T-PATHCOUNTS",
        "output": rel(out),
        "counts_one_configuration_not_st_paths": True,
        "instance": {
            "graph": "6-0-1-2-3-4-5-7", "source": S, "target": T,
            "main_weights": [1, 2, 3, 4, 5], "pendant_weights": [6, 6],
            "B": B, "c": c, "b_k": bk, "D": D, "lambda": lam, "L": L, "W": W_PATH,
        },
        "header": header,
        "rows": rows,
        "lemma1_counts": {"active": int(active), "beta": float(beta_p),
                          "alpha": float(alpha_p), "gamma": float(gamma_p)},
        "energies": {
            "E_NC_unaugmented": float(e_nc_un), "E_SPC_unaugmented": float(e_spc_un),
            "E_NC_augmented_b_st_inherit": float(e_nc_ai),
            "E_NC_augmented_b_st_anchored": float(e_nc_a0),
            "E_SPC_augmented": float(e_spc_ai),
            "sumsum_alpha_wjwk_augmented": float(alpha_wsum_aug),
        },
        "checks": checks,
        "all_a4_cells_match": all_ok,
    }


if __name__ == "__main__":
    run()
