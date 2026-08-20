"""Experiment 3 -- lambda-tolerance / the optimality gap.

Claims: TOL-REF, TOL-DENOM, TOL-POP, TOL-FAM, TOL-K (CLAIMS.tsv, section IV-C).
Outputs:
  results/tolerance/reference.json      (TOL-REF, TOL-DENOM)  -- default / --reference, tier 1
  results/tolerance/population.csv       (TOL-POP, TOL-FAM)    -- --population, tier 2
  results/tolerance/dfs_crosscheck.json  (TOL-K)               -- --crosscheck, tier 2

Origin (the build spec 3.2): PORT -- "72 instances, K=4000, DFS cross-check", using
Yen's loopless k-shortest-paths (spnn.yen).  The reference part is the stage-26 pilot
rider; the population part is the stage-28 rider (from an internal harness), whose
graph builders use only the public generators (spnn.graphs) so the population is
reconstructible here.

THE MEASURE.  For an instance with optimum P (weight W*, hop count L_P), the
lambda-robustness DENOMINATOR is

    D(G) = min over rival simple paths Q with L_Q != L_P of  (W_Q - W*) / |L_Q - L_P|,

i.e. how much weight-per-hop separates the optimum from the nearest length-CHANGING
rival.  D is found with Yen's algorithm (K shortest LOOPLESS paths).  Because Yen is
TRUNCATED at K, D is an UPPER bound: a rival outside the returned K has W_Q >= the last
returned weight but could carry a larger |L_Q - L_P| and hence a smaller ratio.
``k_exhausted`` records whether Yen ran the graph out of simple paths (D exact) or hit K
(D an upper bound); both are reported, nothing exact is presented that is not.

  * TOL-REF (tier 1) -- on the reference instance the optimum is W* = 0.548769 at L=8
    and the nearest length-changing rival is W_Q = 0.550628, a 4-hop path (L=4).  This
    corrects a single-instance misreading that took the rival at the optimum's own depth.
  * TOL-DENOM (tier 1) -- denominator = (0.550628 - 0.548769)/|4-8| = 4.648e-4.  At the
    c=4 operating point the corresponding length-bias tolerance is (2c/B)*D = 1.9e-3
    (from the microstate energy E_SPC = 2c*Etilde = (2c/B)*W: a length-changing rival
    crosses the optimum in energy when lambda*|dL| = (2c/B)*(W_Q - W*)).
  * TOL-POP / TOL-FAM (tier 1/2) -- the 72-instance population (9 cells x 8 seeds,
    depths 3-15): median denominator 5.2e-2; family medians 1.04 / 0.067 / 0.042 / 1.3e-3
    for lattice / sparse / dense / terrain.  The full K=4000 sweep is expensive (tens of
    minutes); the committed population.csv is reproduced by ``--population`` and VERIFIED
    (cheaply, from the committed file) on every default run.
  * TOL-K (tier 2) -- Yen reaches K=4000 on all 72 instances, and its per-hop minima
    agree with an exhaustive depth-limited DFS on 147 of 162 length strata (the other 15
    are strata where a lighter path at that hop count lies outside the first 4000).

Reference instance: ``random_graph(60, 0.05, seed=19)``, source 0, target 59.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys

from spnn.counting import count_paths_by_length
from spnn.graphs import (dijkstra, grid_graph, mountain_spatial_graph, random_graph)
from spnn.yen import k_shortest_paths

from experiments._base import rel, results_path, write_json

# --- the reference instance ------------------------------------------------
REF_N, REF_P, REF_SEED = 60, 0.05, 19
REF_S, REF_T = 0, 59
REF_C = 4.0                          # the c=4 operating point (TOL-DENOM)
# Yen K for the reference: the nearest length-changing rival is the global 2nd-lightest
# path, so the denominator is stable from K=2; 64 confirms it across the low-L strata in
# ~1 s.  The population uses the full K below.
REF_K = 64

# --- the population (stage-28 rider) ---------------------------------------
POP_K = 4000
PER_CELL = 8
MIN_OPT_DEPTH = 3
DEG_SPARSE, DEG_DENSE = 3.0, 6.0
GRID_JITTER, MTN_KNN_K = 0.25, 4
# 9 cells x 8 seeds = 72 instances, spanning all four families and depths 3-15.
CELLS = [("sparse", 30), ("sparse", 60), ("sparse", 100),
         ("dense", 40), ("dense", 80),
         ("grid", 5), ("grid", 8),
         ("mtn_knn", 40), ("mtn_knn", 80)]
# family -> paper label, in the TOL-FAM order (lattice / sparse / dense / terrain)
FAMILY_LABEL = {"grid": "lattice", "sparse": "sparse",
                "dense": "dense", "mtn_knn": "terrain"}
TOL_FAM_ORDER = ["grid", "sparse", "dense", "mtn_knn"]


# ---------------------------------------------------------------------------
def _denominator(A, LP, Wstar):
    """min over Yen paths Q (L_Q != L_P) of (W_Q - W*)/|L_Q - L_P|; returns
    (D, L_Q, W_Q) or (None, None, None).  A is [(path, weight), ...] ascending."""
    best = None
    for path, wt in A:
        LQ = len(path) - 1
        if LQ == LP:
            continue
        v = (wt - Wstar) / abs(LQ - LP)
        if best is None or v < best[0]:
            best = (v, LQ, wt)
    return best if best is not None else (None, None, None)


def reference() -> dict:
    """TOL-REF + TOL-DENOM on the reference instance.  Tier 1, fast."""
    n, edges, w = random_graph(REF_N, REF_P, seed=REF_SEED)
    A = k_shortest_paths(n, edges, w, REF_S, REF_T, K=REF_K)
    opt_path, Wstar = A[0]
    LP = len(opt_path) - 1
    D, LQ, WQ = _denominator(A, LP, Wstar)
    B = 2.0 * float(w.max())
    tolerance_c4 = (2.0 * REF_C / B) * D          # length-bias tolerance at c=4

    result = dict(
        instance=f"random_graph({REF_N}, {REF_P}, seed={REF_SEED})",
        source=REF_S, target=REF_T, K=REF_K, n_edges=int(edges.shape[0]),
        TOL_REF=dict(
            W_star=Wstar, L_star=LP,
            W_rival=WQ, L_rival=LQ,
            rival_is_length_changing=bool(LQ != LP),
            note="rival W_Q is a 4-hop path (L=4), not at the optimum's depth L=8"),
        TOL_DENOM=dict(
            denominator=D, attaining_L_Q=LQ,
            weight_gap=WQ - Wstar,
            c=REF_C, B=B,
            tolerance_at_c4=tolerance_c4,
            tolerance_formula="(2*c/B) * denominator  [E_SPC = 2c*Etilde = (2c/B)*W]",
            note="tolerance rounds to 1.9e-3 at c=4; raw weight gap is 1.859e-3"),
    )
    out = write_json("tolerance/reference.json", result)
    print(f"[3] lambda_tolerance --reference -> {rel(out)}")
    print(f"    TOL-REF   W*={Wstar:.6f} @ L={LP};  rival W_Q={WQ:.6f} @ L={LQ}")
    print(f"    TOL-DENOM denominator={D:.6e}  tolerance@c=4=(2c/B)*D={tolerance_c4:.4e}"
          f"  (~1.9e-3)")
    return result


# ---------------------------------------------------------------------------
def build_graph(family, size, seed):
    """(n, edges, w, label).  `size` is N except for grid, where it is the lattice k."""
    if family == "sparse":
        n, e, w = random_graph(size, DEG_SPARSE / (size - 1), seed=seed)
        return n, e, w, f"sparse_N{size}_seed{seed}"
    if family == "dense":
        n, e, w = random_graph(size, DEG_DENSE / (size - 1), seed=seed)
        return n, e, w, f"dense_N{size}_seed{seed}"
    if family == "grid":
        n, e, w = grid_graph(size, seed=seed, jitter=GRID_JITTER)
        return n, e, w, f"grid_k{size}_N{n}_seed{seed}"
    if family == "mtn_knn":
        n, e, w = mountain_spatial_graph(size, seed=seed, k=MTN_KNN_K)
        return n, e, w, f"mtnknn_N{size}_seed{seed}"
    raise ValueError(family)


def choose_terminals(n, edges, w):
    """s=0, t=n-1 if that pair's optimum is deep enough; else re-target t to the
    lowest-id vertex at the smallest optimal depth >= MIN_OPT_DEPTH.  Never rejects a
    graph.  Returns (s, t, path, used_fallback)."""
    s, t = 0, n - 1
    path, _ = dijkstra(n, edges, w, s, t)
    if path is not None and len(path) - 1 >= MIN_OPT_DEPTH:
        return s, t, path, False
    best = None
    for tv in range(1, n):
        p2, _ = dijkstra(n, edges, w, s, tv)
        if p2 is None:
            continue
        d = len(p2) - 1
        if d >= MIN_OPT_DEPTH and (best is None or d < best[0]):
            best = (d, tv, p2)
    if best is None:
        return s, t, path, False
    return s, best[1], best[2], True


def _one(family, size, seed):
    """One instance: Yen K=POP_K denominator + the exhaustive-DFS strata cross-check."""
    n, edges, w, label = build_graph(family, size, seed)
    s, t, path, fb = choose_terminals(n, edges, w)
    if path is None:
        return dict(label=label, status="unreachable")
    L = len(path) - 1
    A = k_shortest_paths(n, edges, w, s, t, K=POP_K)
    if not A:
        return dict(label=label, status="no_paths")
    opt_path, Wstar = A[0]
    LP = len(opt_path) - 1
    D, LQ, WQ = _denominator(A, LP, Wstar)
    # cross-check Yen's per-hop minima against exhaustive depth-limited DFS (exact)
    cnts, minw, _ = count_paths_by_length(n, edges, w, s, t, L)
    dfs_min = {j: float(minw[j]) for j in range(1, L + 1) if cnts[j]}
    yen_min: dict[int, float] = {}
    for p_, wt in A:
        lq = len(p_) - 1
        yen_min[lq] = min(yen_min.get(lq, math.inf), wt)
    agree = sum(1 for j, v in dfs_min.items()
                if abs(yen_min.get(j, math.inf) - v) < 1e-6)
    return dict(
        label=label, status="ok", family=family, size=size, seed=seed,
        N=n, E=int(edges.shape[0]), L=L, terminal_fallback=int(fb),
        W_star=Wstar, K_returned=len(A), k_exhausted=bool(len(A) < POP_K),
        denominator=D, exact=bool(len(A) < POP_K),
        attaining_L_Q=LQ, attaining_W_Q=WQ,
        rel_denominator=(None if D is None else D / Wstar),
        yen_vs_dfs_strata_agree=f"{agree}/{len(dfs_min)}")


def _run_sweep(cells=CELLS, per_cell=PER_CELL):
    """The full population sweep (expensive: Yen K=4000 x 72 instances)."""
    rows = []
    for fam, sz in cells:
        for sd in range(1, per_cell + 1):
            r = _one(fam, sz, sd)
            rows.append(r)
            print(f"    {r['label']:<24} L={r.get('L', '?'):<3} "
                  f"D={r.get('denominator')!r:<22} K={r.get('K_returned')} "
                  f"strata={r.get('yen_vs_dfs_strata_agree')}", flush=True)
    return rows


_POP_COLS = ["family", "size", "seed", "label", "N", "E", "L", "terminal_fallback",
             "W_star", "denominator", "attaining_L_Q", "attaining_W_Q",
             "rel_denominator", "K_returned", "k_exhausted", "exact",
             "yen_vs_dfs_strata_agree"]


def _write_population_csv(rows):
    path = results_path("tolerance/population.csv")
    with path.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=_POP_COLS, extrasaction="ignore")
        wtr.writeheader()
        for r in rows:
            if r.get("status") == "ok":
                wtr.writerow(r)
    return path


def _family_medians(ok_rows):
    out = {}
    for fam in TOL_FAM_ORDER:
        ds = [r["denominator"] for r in ok_rows if r["family"] == fam]
        if ds:
            out[FAMILY_LABEL[fam]] = statistics.median(ds)
    return out


def _summarise_population(ok_rows, source):
    Ds = [r["denominator"] for r in ok_rows]
    Ls = [int(r["L"]) for r in ok_rows]
    fam_med = _family_medians(ok_rows)
    median_D = statistics.median(Ds)
    return dict(
        source=source, n_instances=len(ok_rows),
        depth_min=min(Ls), depth_max=max(Ls),
        median_denominator=median_D,
        median_matches_5p2e_2=bool(abs(median_D - 5.2e-2) < 1e-3),
        family_medians=fam_med,
        family_median_order="lattice / sparse / dense / terrain",
        expected_family_medians=dict(lattice=1.04, sparse=0.067,
                                     dense=0.042, terrain=1.3e-3))


def population(rows=None) -> dict:
    """TOL-POP + TOL-FAM.  Writes population.csv (from a fresh sweep if `rows` given)
    and returns the medians."""
    if rows is not None:
        _write_population_csv(rows)
        ok = [r for r in rows if r.get("status") == "ok"
              and r.get("denominator") is not None]
        source = "recomputed (--population)"
    else:
        ok = _read_population_csv()
        source = "committed results/tolerance/population.csv"
    summ = _summarise_population(ok, source)
    print(f"[3] lambda_tolerance population ({source}) "
          f"-> {rel(results_path('tolerance/population.csv'))}")
    print(f"    TOL-POP median denominator {summ['median_denominator']:.4e} "
          f"(exp 5.2e-2), depths {summ['depth_min']}-{summ['depth_max']}, "
          f"n={summ['n_instances']}")
    fam = summ["family_medians"]
    print(f"    TOL-FAM lattice {fam['lattice']:.3g} / sparse {fam['sparse']:.3g} / "
          f"dense {fam['dense']:.3g} / terrain {fam['terrain']:.3g}  "
          f"(exp 1.04 / 0.067 / 0.042 / 1.3e-3)")
    return summ


def _read_population_csv():
    path = results_path("tolerance/population.csv")
    ok = []
    with path.open() as f:
        for r in csv.DictReader(f):
            r["denominator"] = float(r["denominator"])
            r["L"] = int(r["L"])
            ok.append(r)
    return ok


def crosscheck(rows=None) -> dict:
    """TOL-K.  Writes dfs_crosscheck.json (from a fresh sweep if `rows` given)."""
    if rows is not None:
        ok = [r for r in rows if r.get("status") == "ok"]
        source = "recomputed (--crosscheck)"
    else:
        ok = _read_population_csv()
        source = "committed results/tolerance/population.csv"
    agree_num = agree_den = 0
    per, disagree = [], []
    for r in ok:
        a, b = r["yen_vs_dfs_strata_agree"].split("/")
        a, b = int(a), int(b)
        agree_num += a
        agree_den += b
        kret = int(r["K_returned"])
        per.append(dict(label=r["label"], family=r["family"], L=int(r["L"]),
                        K_returned=kret, strata_agree=a, strata_total=b))
        if a < b:
            disagree.append(dict(label=r["label"], L=int(r["L"]), agree=a, total=b))
    k_all = all(int(r["K_returned"]) == POP_K for r in ok)
    result = dict(
        claim="TOL-K", tier=2, state="built", source=source, K=POP_K,
        n_instances=len(ok), K_reached_on_all=k_all,
        strata_agree=agree_num, strata_total=agree_den,
        expected_agree=147, expected_total=162,
        matches=bool(agree_num == 147 and agree_den == 162 and k_all),
        n_disagreeing_instances=len(disagree), disagreeing=disagree,
        note=("every D is an UPPER bound (Yen truncated at K); a stratum disagrees when "
              "a lighter path at that hop count lies outside the first K returned."),
        per_instance=per)
    out = write_json("tolerance/dfs_crosscheck.json", result)
    print(f"[3] lambda_tolerance --crosscheck ({source}) -> {rel(out)}")
    print(f"    TOL-K K={POP_K} on all {len(ok)}: {k_all};  DFS strata "
          f"{agree_num} of {agree_den} (exp 147 of 162; MATCH={result['matches']})")
    return result


# ---------------------------------------------------------------------------
def run(argv=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    do_pop = "--population" in argv or "--full" in argv
    do_xc = "--crosscheck" in argv
    out: dict = {}

    if do_pop or do_xc:
        # one expensive sweep feeds both population.csv and dfs_crosscheck.json
        rows = _run_sweep()
        out["TOL_POP_FAM"] = population(rows)
        out["TOL_K"] = crosscheck(rows)
        return out

    # default / --reference: the fast tier-1 reference, plus a cheap verification of
    # the committed population (no sweep).
    out["TOL_REF_DENOM"] = reference()
    if results_path("tolerance/population.csv").exists():
        out["TOL_POP_FAM"] = population(rows=None)
    if results_path("tolerance/dfs_crosscheck.json").exists():
        out["TOL_K"] = crosscheck(rows=None)
    return out


if __name__ == "__main__":
    run()
