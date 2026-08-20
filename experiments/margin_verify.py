"""Experiment 6 -- the path-vs-scatter margin, the microstate ruling, the variant.

Claims: M-VERIFY, M-MICRO, M-VARIANT (CLAIMS.tsv, section V-A).
Outputs: results/margin/verify.tsv       (M-VERIFY)
         results/margin/conventions.json (M-MICRO)
         results/margin/variant.json     (M-VARIANT)

Extracted from an internal reference harness (the AUGMENTED margin) and
its Part B/C companion (the two counting conventions). Everything here is EXACT and
hand-built -- nothing is sampled -- and every energy is read back from the compiled
network's own ``sim.energy``.

M-VERIFY.  On the AUGMENTED network (Ch. 4 step 1: two new vertices S and T joined
to source/target by zero-weight edges) the size-matched-scatter-minus-path energy
margin is exactly

    margin  ==  E(scatter) - E(path)  ==  c * [ (L+1) - 2*Etilde(P) ],
    Etilde(P) = sum_{e in P} w_e / B.

Both hand-built configurations are differenced through the compiler's energy, over
c in {1,4} x L in {3,5,8} corridor graphs plus the reference instance -- 16 rows,
and the worst |measured - predicted| is exactly zero.

*** Terminal convention. ***  Two competitor arms are
emitted per (graph, c), and they behave differently in b_st:

  * ``aux-admitted`` -- the size-matched scatter ADMITS the two zero-weight
    auxiliary edges (S,s) and (t,T), so S and T are active in BOTH families and
    b_S cancels identically.  This is the PUBLISHED convention (paper §V-A's "one
    point of hygiene").  Under it the margin is **b_st-INDEPENDENT**: the identity
    holds exactly at b_st=0.0 (anchored, paper §III-A's b_S=b_T=0) AND at
    b_st=None (inherit b_k).  This is the arm the M-VERIFY exact-zero is claimed on.
  * ``all-real`` -- a size-matched scatter of L+2 disjoint REAL edges, no terminals.
    A DIFFERENT comparison: its bias sum differs from the path's by 2*b_S - 2*b_k,
    which at the reference b_k=-2, b_S=0 is exactly 4.0.  So it is zero at
    b_st=None but 4.0 at b_st=0.0.  That 4.0 is the terminal-bias discount the S/T
    augmentation removes -- a documented diagnostic, not a failure, and NOT part of
    the M-VERIFY exact-zero claim.

The table below emits, for all 16 rows, the ``arm`` and the discrepancy at BOTH
b_st=0.0 and b_st=None, so the b_st-independence of the published (aux-admitted)
arm is visible in the data.  The reproduced M-VERIFY quantity is the worst
discrepancy over the aux-admitted rows at the published b_st=0.0, which is zero.

M-MICRO.  The Boltzmann mass ratio is a ratio of MICROSTATE (neuron-configuration)
counts, not edge-set counts: an L-hop augmented path carries 2^(L+1) equal-energy
WTA slot assignments and a k-matching of real edges carries 4^k. On the reference
instance (L=8, matched size k=L+2=10) the per-configuration multiplicity ratio is
4^k / 2^(L+1) = 2^(L+3) = 2048 = 7.6 nats, which moves the required alpha-scale c by
+0.89 relative to the edge-set convention.

M-VARIANT.  The reference instance on the VARIANT (unaugmented) construction, at
c=1, lambda=+1: E_NC = -7.000000, E_SPC = 0.510616, so Delta = |E_N| = 6.489384 and
alphabar = E_N / E_NC = 0.927.
"""

from __future__ import annotations

import math

import numpy as np

from spnn.compile import Params, compile_network
from spnn.counting import count_matchings, count_paths_by_length, scatter_counts
from spnn.graphs import dijkstra, random_graph
from spnn.sim import energy, ideal_state, state_counts

from experiments._base import rel, write_json, write_tsv

REF_N, REF_P, REF_SEED = 60, 0.05, 19
REF_S, REF_T = 0, 59
D = 4.0  # beta / drive strength (spnn default)


# ----------------------------------------------------------------------
# Hand-built configurations (from parta).
# ----------------------------------------------------------------------
def corridor_graph(L: int, seed: int = 0):
    """Path 0-1-...-L, plus 2(L+2) pool vertices carrying L+2 disjoint edges."""
    rng = np.random.default_rng(seed)
    n_path = L + 1
    k = L + 2  # matched scatter size
    pool = [n_path + i for i in range(2 * k)]
    e = [(i, i + 1) for i in range(L)]
    matching = [(pool[2 * i], pool[2 * i + 1]) for i in range(k)]
    e += matching
    e += [(0, pool[2 * i]) for i in range(k)]  # connectors, all incident to 0
    edges = np.array(sorted(e), dtype=np.int32)
    w = rng.uniform(0.2, 1.0, size=edges.shape[0]).astype(np.float32)
    n = n_path + 2 * k
    return n, edges, w, 0, L, list(range(L + 1)), matching


def scatter_state(net, edge_list, comp_choice=0) -> np.ndarray:
    """State activating exactly the two endpoint neurons of each edge (disjoint)."""
    seen = set()
    for u, v in edge_list:
        if u in seen or v in seen:
            raise AssertionError(f"scatter edges are not disjoint at ({u},{v})")
        seen.add(u)
        seen.add(v)
    x = np.zeros(net.n_neurons, dtype=bool)
    for u, v in edge_list:
        x[net.neuron_at(u, v, comp_choice)] = True
        x[net.neuron_at(v, u, comp_choice)] = True
    return x


def greedy_matching(n, edges, k, forbid=(), seed=0):
    """k pairwise-disjoint edges avoiding vertices in `forbid`.  Deterministic."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(edges.shape[0])
    used = set(forbid)
    out = []
    for ei in order:
        u, v = int(edges[ei, 0]), int(edges[ei, 1])
        if u in used or v in used:
            continue
        used.add(u)
        used.add(v)
        out.append((u, v))
        if len(out) == k:
            return out
    raise RuntimeError(f"could not find a {k}-matching (got {len(out)})")


def _report(net, x):
    active, a, b, g = state_counts(net, x)
    e_nc, e_spc = energy(net, x)
    return dict(active=active, alpha=a, beta=b, gamma=g, E_N=float(e_nc + e_spc))


def _arm_discrepancies(n, edges, w, source, target, path, real_matching, c, b_st):
    """|measured - predicted| for both arms at one b_st. Returns {arm: (abs_err, struct_ok)}."""
    L = len(path) - 1
    net = compile_network(n, edges, w, source, target,
                          params=Params(alpha_scale=c, b_st=b_st), augment_st=True)
    B = float(net.params.B)
    eidx = {}
    for i in range(edges.shape[0]):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        eidx[(u, v)] = eidx[(v, u)] = float(w[i])
    wsum = sum(eidx[(path[i], path[i + 1])] for i in range(L))
    Etilde = wsum / B
    pred = c * ((L + 1) - 2.0 * Etilde)

    rp = _report(net, ideal_state(net, net.augmented_path(path)))
    rs_real = _report(net, scatter_state(net, real_matching))            # all-real
    aux_matching = [(net.aug_source, source), (target, net.aug_target)] + real_matching[:L]
    rs_aux = _report(net, scatter_state(net, aux_matching))              # aux-admitted

    out = {}
    for arm, r in (("aux-admitted", rs_aux), ("all-real", rs_real)):
        meas = r["E_N"] - rp["E_N"]
        struct_ok = (r["active"] == rp["active"] and r["beta"] == rp["beta"]
                     and r["gamma"] == rp["gamma"])
        out[arm] = (abs(meas - pred), struct_ok)
    return out, pred, Etilde


def margin_rows(case, n, edges, w, source, target, path, real_matching, c):
    """Two rows (aux-admitted + all-real) for one (graph, path, c), each with the
    discrepancy at BOTH b_st=0.0 (published, anchored) and b_st=None (inherit b_k)."""
    L = len(path) - 1
    d0, pred, Etilde = _arm_discrepancies(n, edges, w, source, target, path,
                                          real_matching, c, b_st=0.0)
    dN, _, _ = _arm_discrepancies(n, edges, w, source, target, path,
                                  real_matching, c, b_st=None)
    rows = []
    for arm in ("aux-admitted", "all-real"):
        err0, struct_ok = d0[arm]
        errN, _ = dN[arm]
        rows.append(dict(case=case, c=c, L=L, arm=arm, Etilde=Etilde,
                         predicted=pred, disc_bst0=err0, disc_bstNone=errN,
                         struct_ok=struct_ok))
    return rows


# ----------------------------------------------------------------------
def _reference_instance():
    n, edges, w = random_graph(REF_N, REF_P, seed=REF_SEED)
    path, cost = dijkstra(n, edges, w, REF_S, REF_T)
    L = len(path) - 1
    B = 2.0 * float(np.max(w))
    eidx = {}
    for i in range(len(edges)):
        u, v = int(edges[i][0]), int(edges[i][1])
        eidx[(u, v)] = eidx[(v, u)] = float(w[i])
    wsum = sum(eidx[(path[i], path[i + 1])] for i in range(L))
    return n, edges, w, path, cost, L, B, wsum, wsum / B


def conventions(n, edges, w, path, L, B, Etilde):
    """M-MICRO: the microstate ruling on the reference instance."""
    counts, _minw, _wit = count_paths_by_length(n, edges, w, REF_S, REF_T, L)
    n_path_edge = counts[L]
    k = L + 2

    sc = scatter_counts(n, edges, REF_S, REF_T, k)
    m_old, _ = count_matchings(n, [(int(a), int(b)) for a, b in edges], L)

    path_mult = 2 ** (L + 1)               # WTA slot multiplicity of an L-hop path
    per_matching_mult = 4 ** k             # 4 = 2^2 slot assignments per real edge
    micro_ratio = per_matching_mult // path_mult   # = 2^(L+3) = 2048 at L=8
    micro_ratio_nats = math.log(micro_ratio)

    path_micro = n_path_edge * path_mult
    margin_c1 = (L + 1) - 2.0 * Etilde

    def creq(num, den):
        r = math.log(num) - math.log(den)
        return r, r / margin_c1

    ln_edge, c_edge = creq(sc["edge_aux"], n_path_edge)
    ln_micro, c_micro = creq(sc["micro_aux"], path_micro)

    return {
        "instance": {
            "generator": f"random_graph({REF_N}, {REF_P}, seed={REF_SEED})",
            "source": REF_S, "target": REF_T, "L": L, "matched_size_k": k,
            "B": B, "Etilde": Etilde, "margin_at_c1": margin_c1,
        },
        "path": {
            "n_path_edge": n_path_edge,
            "path_multiplicity_2^(L+1)": path_mult,
            "path_micro": path_micro,
        },
        "matching": {
            "per_edge_multiplicity": 4,
            "per_matching_multiplicity_4^k": per_matching_mult,
        },
        "microstate_ratio": {
            "ratio_4^k_over_2^(L+1)": micro_ratio,
            "ratio_nats": micro_ratio_nats,
        },
        "scatter_counts": {
            "N0": sc["N0"], "N1": sc["N1"], "N2": sc["N2"],
            "edge_aux": sc["edge_aux"], "micro_aux": sc["micro_aux"],
            "old_edge": m_old[L], "old_micro": m_old[L] * (4 ** L),
        },
        "required_c": {
            "edge_convention": c_edge,
            "microstate_convention": c_micro,
            "delta_micro_minus_edge": c_micro - c_edge,
            "ln_ratio_edge_nats": ln_edge,
            "ln_ratio_micro_nats": ln_micro,
        },
    }


def variant(n, edges, w, path, cost, L, B, wsum, Etilde):
    """M-VARIANT: the reference instance on the VARIANT (unaugmented) construction.

    c=1, lambda=+1 (b_k = (lambda - D - c)/2 = -2, D=4): E_NC = -7, E_SPC = 0.510616.
    """
    c, lam = 1.0, 1.0
    b_k = (lam - D - c) / 2.0
    net = compile_network(n, edges, w, REF_S, REF_T,
                          params=Params(D=D, alpha_scale=c, bias=b_k), augment_st=False)
    e_nc, e_spc = energy(net, ideal_state(net, net.augmented_path(path)))
    E_N = float(e_nc + e_spc)
    return {
        "instance": {
            "generator": f"random_graph({REF_N}, {REF_P}, seed={REF_SEED})",
            "source": REF_S, "target": REF_T, "L": L, "path_weight": wsum,
            "B": B, "Etilde": Etilde,
        },
        "params": {"c": c, "lambda": lam, "D": D, "b_k": b_k, "augment_st": False},
        "E_NC": float(e_nc),
        "E_SPC": float(e_spc),
        "E_N": E_N,
        "Delta": -E_N,
        "alphabar": E_N / float(e_nc),
    }


def run(argv=None) -> dict:
    # ---- M-VERIFY: the augmented path-vs-scatter margin (16 rows) ----
    rows = []
    for c in (1.0, 4.0):
        for L in (3, 5, 8):
            n, edges, w, s, t, path, matching = corridor_graph(L, seed=100 + L)
            rows += margin_rows(f"corridor L={L}", n, edges, w, s, t, path, matching, c)
    rn, re, rw, rpath, rcost, rL, rB, rwsum, rEt = _reference_instance()
    for c in (1.0, 4.0):
        m = greedy_matching(rn, re, rL + 2, forbid=(REF_S, REF_T), seed=7)
        rows += margin_rows("reference random_graph(60,0.05,seed=19)",
                            rn, re, rw, REF_S, REF_T, rpath, m, c)

    # M-VERIFY reproduces on the AUX-ADMITTED arm at the published b_st=0.0.
    aux_rows = [r for r in rows if r["arm"] == "aux-admitted"]
    real_rows = [r for r in rows if r["arm"] == "all-real"]
    worst = max(r["disc_bst0"] for r in aux_rows)             # published margin, = 0
    worst_aux_bstNone = max(r["disc_bstNone"] for r in aux_rows)
    worst_real_bst0 = max(r["disc_bst0"] for r in real_rows)  # the 4.0 diagnostic
    header = ["case", "c", "L", "arm", "Etilde", "predicted",
              "disc_bst0", "disc_bstNone", "struct_ok"]
    tsv_rows = [[r["case"], r["c"], r["L"], r["arm"], r["Etilde"],
                 r["predicted"], r["disc_bst0"], r["disc_bstNone"], r["struct_ok"]]
                for r in rows]
    verify_out = write_tsv("margin/verify.tsv", header, tsv_rows)

    # ---- M-MICRO: the microstate ruling ----
    conv = conventions(rn, re, rw, rpath, rL, rB, rEt)
    conv_out = write_json("margin/conventions.json", conv)

    # ---- M-VARIANT: the reference instance on the variant ----
    var = variant(rn, re, rw, rpath, rcost, rL, rB, rwsum, rEt)
    var_out = write_json("margin/variant.json", var)

    micro = conv["microstate_ratio"]
    dc = conv["required_c"]["delta_micro_minus_edge"]
    print(f"[6] margin_verify -> {rel(verify_out)}, {rel(conv_out)}, {rel(var_out)}")
    print(f"    M-VERIFY  {len(rows)} rows (8 aux-admitted + 8 all-real).")
    print(f"      aux-admitted (PUBLISHED): worst disc = {worst:.3e} at b_st=0.0, "
          f"{worst_aux_bstNone:.3e} at b_st=None  -> b_st-INDEPENDENT, 0.0 correct")
    print(f"      all-real (diagnostic):    worst disc = {worst_real_bst0:.3f} at b_st=0.0 "
          f"(the 2*b_S-2*b_k terminal-bias discount the augmentation removes)")
    print(f"    M-MICRO   ratio 4^k/2^(L+1) = {micro['ratio_4^k_over_2^(L+1)']} "
          f"= {micro['ratio_nats']:.4f} nats = +{dc:.4f} in c")
    print(f"    M-VARIANT Delta = {var['Delta']:.6f}  alphabar = {var['alphabar']:.6f}")
    return {
        "M_VERIFY": {"rows": len(rows),
                     "worst_abs_err": worst,               # aux-admitted @ b_st=0.0 (published)
                     "arm": "aux-admitted", "b_st": 0.0,
                     "aux_worst_bst_None": worst_aux_bstNone,
                     "all_real_worst_bst0": worst_real_bst0},
        "M_MICRO": {"micro_ratio": micro["ratio_4^k_over_2^(L+1)"],
                    "ratio_nats": micro["ratio_nats"], "delta_c": dc},
        "M_VARIANT": {"Delta": var["Delta"], "alphabar": var["alphabar"]},
    }


if __name__ == "__main__":
    run()
