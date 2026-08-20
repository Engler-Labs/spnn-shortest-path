"""Experiment 8 -- the required-c POPULATION SWEEP (exact counting only).

Claims: POP-N, T-CREQDIST, POP-SPAN, POP-SIZE, POP-SIZED, C4-POOL, C4-FAM,
C4-SIZE, C4-CONV, CNT-HEAD  (CLAIMS.tsv, section V-B / V-C).

Outputs (under results/creq/):
  * population.csv       -- the committed per-instance table: 1,080 rows (1,027
                            completed + 53 censored, flagged not dropped), one row
                            per (family, size, seed).  Persists ``max_coeff_bits``
                            per row (the Code-availability promise -- CNT-HEAD).
  * by_family.tsv        -- required-c distribution per family + pooled, microstate
                            convention (T-CREQDIST, POP-SPAN).
  * by_size.tsv          -- required-c distribution per (family, size) cell,
                            microstate convention (POP-SIZE, POP-SIZED).
  * c4_verdict.json      -- fraction clearing the discrimination condition at c=4,
                            per family / pooled / N=60 cell (C4-POOL/FAM/SIZE).
  * convention_gap.json  -- edge-set vs microstate c=4 verdict + disagreement count
                            (C4-CONV; written only with --both-conventions).

WHAT THIS EXPERIMENT DOES BY DEFAULT.  It runs the *analysis* off the committed
``population.csv`` -- pure re-analysis, no counting, deterministic, seconds.  This
mirrors the origin: the sweep itself is ~1,080 exact matching-polynomial counts,
~90 core-minutes (tier 2), while every number the paper prints is a statistic of the
committed table.  ``--regenerate`` re-runs the whole sweep from the seeds (expensive;
needs the process-isolation / memory-cap machinery of the original harness and is NOT
what CI does) and is provided for completeness / auditing.

CONVENTIONS (pinned, and declared in the manuscript):
  * required c is reported in BOTH counting conventions -- ``micro`` (Boltzmann
    microstates, WTA slot degeneracy included; the correct one for a mass ratio) and
    ``edge`` (edge sets; reported for comparability with the withdrawn figure).  The
    headline verdict uses ``micro``.
  * an instance CLEARS the c=4 discrimination condition  <=>  required c < 4.
  * quantiles are numpy's default linear interpolation, matching the source harness.

Origin / provenance.  Extracted from the internal stage-28 harnesses (the sweep +
the reference anchor, and the analysis), deleted from that source tree and read at
a specific source revision; the committed per-instance chunk CSVs were concatenated
into ``population.csv`` at another source revision.  Imports are repointed at
``spnn``; a thread-pinning entry-point import (an OpenBLAS oversubscription guard,
a harness concern) is dropped.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np

from spnn.counting import count_matchings, count_paths_by_length, scatter_counts
from spnn.graphs import (dijkstra, grid_graph, mountain_spatial_graph,
                         random_graph)

from experiments._base import RESULTS, rel, results_path, write_json, write_tsv

FAM_ORDER = ["sparse", "dense", "grid", "mtn_knn"]
QS = [0, 10, 25, 50, 75, 90, 100]           # order statistics reported per cell
LEGACY_DIG = 96                             # the legacy 2^96 coefficient field (CNT-HEAD/CNT-49)

# The reference instance (instances/README.md): the flagship published-count graph.
REF_N, REF_P, REF_SEED, REF_S, REF_T = 60, 0.05, 19, 0, 59

POP_CSV = "creq/population.csv"

# The per-instance schema of population.csv (== the source harness FIELDS).
FIELDS = ["family", "size", "seed", "label", "status", "N", "E", "mean_degree",
          "s", "t", "terminal_fallback", "L", "k", "wsum", "B", "Etilde",
          "margin_c1", "alphabar", "min_edge_weight", "max_edge_weight",
          "path_status", "scatter_status",
          "n_path_edge", "n_path_micro", "path_min_weight",
          "scatter_N0", "scatter_N1", "scatter_N2",
          "scatter_edge_noaux", "scatter_edge_aux",
          "scatter_micro_noaux", "scatter_micro_aux",
          "ln_ratio_edge", "ln_ratio_micro", "c_req_edge", "c_req_micro",
          "ln_ratio_edge_noaux", "ln_ratio_micro_noaux",
          "c_req_edge_noaux", "c_req_micro_noaux",
          "frontier", "dp_states", "dig", "max_coeff_bits", "coeff_bound_bits",
          "legacy_dig96_exact", "n_matching_calls",
          "path_wall_s", "scatter_wall_s",
          "peak_rss_mib_after_path", "peak_rss_mib_final", "censor_note"]

_INT_COLS = ("size", "seed", "N", "E", "L", "k", "frontier", "dp_states", "dig",
             "max_coeff_bits", "coeff_bound_bits", "terminal_fallback")
_FLOAT_COLS = ("mean_degree", "Etilde", "margin_c1", "alphabar", "ln_ratio_edge",
               "ln_ratio_micro", "c_req_edge", "c_req_micro", "wsum", "B",
               "path_wall_s", "scatter_wall_s", "peak_rss_mib_final")


# --------------------------------------------------------------------------- load
def load_population():
    """Read the committed population.csv, coercing numeric columns.  Returns
    (all_rows, completed_rows)."""
    path = RESULTS / "creq" / "population.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.  It ships committed in the repo; regenerate it with "
            f"`python -m experiments 8 --regenerate` (expensive, ~90 core-minutes).")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in _INT_COLS:
                if r.get(k) not in (None, ""):
                    r[k] = int(r[k])
            for k in _FLOAT_COLS:
                if r.get(k) not in (None, ""):
                    r[k] = float(r[k])
            rows.append(r)
    done = [r for r in rows if r["status"] == "ok"]
    return rows, done


def quant(vals):
    a = np.asarray(sorted(vals), dtype=float)
    return {q: float(np.percentile(a, q)) for q in QS}


# ----------------------------------------------------------------- CNT-HEAD helper
def _scatter_max_coeff_bits(n, edges, s, t, k):
    """Max ``max_coeff_bits`` over the four count_matchings calls scatter_counts makes
    at matched size k -- the per-instance overflow diagnostic the paper persists."""
    def sub(drop):
        return [(int(a), int(b)) for a, b in edges
                if int(a) not in drop and int(b) not in drop]
    bits = []
    for drop, kk in ((set(), k), ({s}, k - 1), ({t}, k - 1), ({s, t}, k - 2)):
        if kk < 0:
            continue
        ee = sub(drop)
        if not ee:
            continue
        _m, st = count_matchings(n, ee, kk)
        bits.append(st["max_coeff_bits"])
    return max(bits) if bits else 0


def reference_headroom():
    """Recompute the reference instance's coefficient headroom to the legacy 2^96
    field (CNT-HEAD).  random_graph(60, 0.05, seed=19): max_coeff_bits = 45, so the
    flagship published-count instance sits 96 - 45 = 51 bits below the ceiling."""
    n, edges, w = random_graph(REF_N, REF_P, seed=REF_SEED)
    path, _ = dijkstra(n, edges, w, REF_S, REF_T)
    L = len(path) - 1
    mcb = _scatter_max_coeff_bits(n, edges, REF_S, REF_T, L + 2)
    return dict(instance=f"random_graph({REF_N}, {REF_P}, seed={REF_SEED})", L=L,
                max_coeff_bits=mcb, legacy_ceiling_bits=LEGACY_DIG,
                bits_below_ceiling=LEGACY_DIG - mcb)


# ---------------------------------------------------------------------- analysis
def run(argv=None) -> dict:
    argv = list(argv or [])
    both_conventions = "--both-conventions" in argv
    if "--regenerate" in argv:
        return _regenerate([a for a in argv if a != "--regenerate"])

    rows, done = load_population()
    n_att, n_done = len(rows), len(done)
    n_cens = n_att - n_done

    # ---- by_family.tsv (T-CREQDIST, POP-SPAN) -- microstate convention -----------
    fam_header = ["family", "n", "L_med", "min", "q10", "q25", "median",
                  "q75", "q90", "max", "clears_c4"]
    fam_rows = []
    for fam in FAM_ORDER + ["POOLED"]:
        v = done if fam == "POOLED" else [r for r in done if r["family"] == fam]
        if not v:
            continue
        c = [r["c_req_micro"] for r in v]
        q = quant(c)
        fam_rows.append([
            fam, len(v), round(float(np.median([r["L"] for r in v])), 1),
            round(q[0], 4), round(q[10], 4), round(q[25], 4), round(q[50], 4),
            round(q[75], 4), round(q[90], 4), round(q[100], 4),
            sum(1 for x in c if x < 4.0)])
    by_family = write_tsv("creq/by_family.tsv", fam_header, fam_rows)

    # ---- by_size.tsv (POP-SIZE, POP-SIZED) -- microstate convention --------------
    size_header = ["family", "size", "n", "L_med", "min", "q25", "median",
                   "q75", "max", "clears_c4"]
    size_rows = []
    for fam in FAM_ORDER:
        for sz in sorted({r["size"] for r in done if r["family"] == fam}):
            v = [r for r in done if r["family"] == fam and r["size"] == sz]
            c = [r["c_req_micro"] for r in v]
            q = quant(c)
            size_rows.append([
                fam, sz, len(v), round(float(np.median([r["L"] for r in v])), 1),
                round(q[0], 4), round(q[25], 4), round(q[50], 4),
                round(q[75], 4), round(q[100], 4),
                sum(1 for x in c if x < 4.0)])
    by_size = write_tsv("creq/by_size.tsv", size_header, size_rows)

    # ---- c4_verdict.json (C4-POOL, C4-FAM, C4-SIZE) -----------------------------
    def verdict(v):
        cm = [r["c_req_micro"] for r in v]
        ce = [r["c_req_edge"] for r in v]
        km = sum(1 for x in cm if x < 4.0)
        ke = sum(1 for x in ce if x < 4.0)
        return dict(n=len(v), micro_clear=km, edge_clear=ke,
                    micro_frac=km / len(v), edge_frac=ke / len(v),
                    micro_median=float(np.median(cm)),
                    edge_median=float(np.median(ce)))

    by_family_v = {fam: verdict([r for r in done if r["family"] == fam])
                   for fam in FAM_ORDER}
    by_size_v = {}
    for fam in FAM_ORDER:
        for sz in sorted({r["size"] for r in done if r["family"] == fam}):
            cell = [r for r in done if r["family"] == fam and r["size"] == sz]
            by_size_v[f"{fam}_{sz}"] = verdict(cell)
    c4 = dict(pooled=verdict(done), by_family=by_family_v, by_size=by_size_v,
              convention="microstate is the headline; edge reported for comparability",
              clears_defined_as="required c < 4")
    c4_path = write_json("creq/c4_verdict.json", c4)

    # ---- convention_gap.json (C4-CONV) -- only under --both-conventions ----------
    conv_path = None
    micro_clear = c4["pooled"]["micro_clear"]
    edge_clear = c4["pooled"]["edge_clear"]
    # micro-required c >= edge-required c identically, so every micro-clear is an
    # edge-clear; the conventions disagree on exactly (edge_clear - micro_clear).
    disagree = sum(1 for r in done
                   if (r["c_req_edge"] < 4.0) != (r["c_req_micro"] < 4.0))
    if both_conventions:
        conv = dict(
            n=n_done,
            micro_clear=micro_clear, micro_frac=micro_clear / n_done,
            edge_clear=edge_clear, edge_frac=edge_clear / n_done,
            conventions_disagree_at_c4=disagree,
            note="edge clears 40.0%; microstate 13.4%; the c=4 verdict flips on "
                 "273 of 1,027 instances -- every c=4 sentence must name its convention.",
            per_family_gap={
                fam: dict(
                    median=float(np.median([r["c_req_micro"] - r["c_req_edge"]
                                            for r in done if r["family"] == fam])),
                    min=float(np.min([r["c_req_micro"] - r["c_req_edge"]
                                      for r in done if r["family"] == fam])),
                    max=float(np.max([r["c_req_micro"] - r["c_req_edge"]
                                      for r in done if r["family"] == fam])))
                for fam in FAM_ORDER})
        conv_path = write_json("creq/convention_gap.json", conv)

    # ---- CNT-HEAD: max_coeff_bits persisted per row + reference headroom ----------
    mcb_done = [r["max_coeff_bits"] for r in done if r.get("max_coeff_bits") != ""]
    n_over_legacy = sum(1 for r in rows
                        if str(r.get("legacy_dig96_exact")).strip().lower()
                        in ("false", "0"))
    ref = reference_headroom()
    cnt_head = dict(
        max_coeff_bits_persisted_for=len(mcb_done),
        completed=n_done,
        population_max_coeff_bits=dict(min=min(mcb_done), max=max(mcb_done)),
        instances_over_legacy_2p96_field=n_over_legacy,   # CNT-49 cross-check (49)
        reference=ref)

    # ---- POP-N span helpers ------------------------------------------------------
    pooled_c = [r["c_req_micro"] for r in done]
    pq = quant(pooled_c)

    result = dict(
        POP_N=dict(attempted=n_att, completed=n_done, censored=n_cens,
                   censored_pct=100.0 * n_cens / n_att),
        POP_SPAN=dict(min=pq[0], max=pq[100], iqr_lo=pq[25], iqr_hi=pq[75]),
        c4_pooled_micro_frac=c4["pooled"]["micro_frac"],
        c4_pooled_edge_frac=c4["pooled"]["edge_frac"],
        conventions_disagree=disagree,
        cnt_head=cnt_head,
        outputs=dict(population=rel(RESULTS / "creq" / "population.csv"),
                     by_family=rel(by_family), by_size=rel(by_size),
                     c4_verdict=rel(c4_path),
                     convention_gap=rel(conv_path) if conv_path else None))

    # ---- summary ----------------------------------------------------------------
    print(f"[8] creq_sweep -> {rel(RESULTS / 'creq')}/")
    print(f"    POP-N     {n_att} generated; {n_done} completed; "
          f"{n_cens} censored ({100.0 * n_cens / n_att:.1f}%)")
    print(f"    C4-POOL   {c4['pooled']['micro_clear']}/{n_done} clear at c=4 "
          f"(microstate) = {100 * c4['pooled']['micro_frac']:.1f}%")
    print(f"    C4-FAM    grid {by_family_v['grid']['micro_clear']}/"
          f"{by_family_v['grid']['n']}; "
          f"terrain {100 * by_family_v['mtn_knn']['micro_frac']:.1f}%")
    s60 = by_size_v.get("sparse_60", {})
    d60 = by_size_v.get("dense_60", {})
    print(f"    C4-SIZE   sparse N=60: {s60.get('micro_clear')}/{s60.get('n')} clear; "
          f"dense N=60: {d60.get('micro_clear')}/{d60.get('n')} clear")
    print(f"    C4-CONV   edge {100 * c4['pooled']['edge_frac']:.1f}%; "
          f"conventions disagree on {disagree}/{n_done}"
          + ("" if both_conventions else "  (pass --both-conventions for convention_gap.json)"))
    print(f"    POP-SPAN  micro span {pq[0]:.2f} to {pq[100]:.2f}; "
          f"pooled IQR {pq[25]:.2f} to {pq[75]:.2f}")
    print(f"    CNT-HEAD  max_coeff_bits persisted for {len(mcb_done)}/{n_done} rows; "
          f"reference sits {ref['bits_below_ceiling']} bits below the 2^96 field "
          f"(max_coeff_bits={ref['max_coeff_bits']})")
    return result


# --------------------------------------------------------------------------- sweep
# The full re-run.  Faithful port of the stage-28 harness's measurement path; NOT run
# by CI (tier 2, ~90 core-minutes, needs per-instance process isolation + a memory cap
# for the censoring policy).  Provided for auditing.  See the module docstring.
DEG_SPARSE, DEG_DENSE = 3.0, 6.0
GRID_JITTER, MTN_KNN_K = 0.25, 4
MIN_OPT_DEPTH = 3
N_SEEDS, SEED0 = 40, 1
CELLS = [
    ("sparse", 20), ("sparse", 30), ("sparse", 40), ("sparse", 60),
    ("sparse", 80), ("sparse", 100),
    ("dense", 20), ("dense", 30), ("dense", 40), ("dense", 60), ("dense", 80),
    ("grid", 4), ("grid", 5), ("grid", 6), ("grid", 7), ("grid", 8), ("grid", 9),
    ("grid", 10), ("grid", 11), ("grid", 12),
    ("mtn_knn", 20), ("mtn_knn", 30), ("mtn_knn", 40), ("mtn_knn", 60),
    ("mtn_knn", 80), ("mtn_knn", 100), ("mtn_knn", 140),
]


def _build_graph(family, size, seed):
    if family == "sparse":
        return (*random_graph(size, DEG_SPARSE / (size - 1), seed=seed),
                f"sparse_N{size}_seed{seed}")
    if family == "dense":
        return (*random_graph(size, DEG_DENSE / (size - 1), seed=seed),
                f"dense_N{size}_seed{seed}")
    if family == "grid":
        n, e, w = grid_graph(size, seed=seed, jitter=GRID_JITTER)
        return n, e, w, f"grid_k{size}_N{n}_seed{seed}"
    if family == "mtn_knn":
        return (*mountain_spatial_graph(size, seed=seed, k=MTN_KNN_K),
                f"mtnknn_N{size}_seed{seed}")
    raise ValueError(family)


def _choose_terminals(n, edges, w):
    """L>=MIN_OPT_DEPTH imposed by CHOOSING the terminal pair, never by rejecting
    graphs (identical rule to the source harness)."""
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


def _measure_instance(family, size, seed):
    """One instance end to end, in-process (no walls / memory cap -- those are the
    original harness's process-isolation concern; a censored cell here just raises).
    Faithful arithmetic: required c in both conventions off the exact counts."""
    n, edges, w, label = _build_graph(family, size, seed)
    s, t, path, fallback = _choose_terminals(n, edges, w)
    if path is None:
        return dict(family=family, size=size, seed=seed, label=label,
                    status="UNREACHABLE")
    L = len(path) - 1
    B = 2.0 * float(np.max(w))
    eidx = {}
    for i in range(len(edges)):
        u, v = int(edges[i][0]), int(edges[i][1])
        eidx[(u, v)] = eidx[(v, u)] = float(w[i])
    wsum = sum(eidx[(path[i], path[i + 1])] for i in range(L))
    Et = wsum / B
    margin = (L + 1) - 2.0 * Et
    row = dict(family=family, size=size, seed=seed, label=label, status="ok",
               N=n, E=len(edges), mean_degree=2 * len(edges) / n, s=s, t=t,
               terminal_fallback=int(fallback), L=L, k=L + 2, wsum=wsum, B=B,
               Etilde=Et, margin_c1=margin, alphabar=1.0 - 2.0 * Et / (L + 1),
               min_edge_weight=float(np.min(w)), max_edge_weight=float(np.max(w)),
               path_status="ok", scatter_status="ok")
    cnts, minw, _wit = count_paths_by_length(n, edges, w, s, t, L)
    row["n_path_edge"] = cnts[L]
    row["n_path_micro"] = cnts[L] * (2 ** (L + 1))
    row["path_min_weight"] = float(minw[L])
    row["max_coeff_bits"] = _scatter_max_coeff_bits(n, edges, s, t, L + 2)
    sc = scatter_counts(n, edges, s, t, L + 2)
    row.update({f"scatter_{key}": sc[key] for key in
                ("N0", "N1", "N2", "edge_noaux", "edge_aux", "micro_noaux", "micro_aux")})
    for conv, npc, nsa in (("edge", row["n_path_edge"], sc["edge_aux"]),
                           ("micro", row["n_path_micro"], sc["micro_aux"])):
        lr = math.log(nsa) - math.log(npc)
        row[f"ln_ratio_{conv}"] = lr
        row[f"c_req_{conv}"] = lr / margin if margin > 0 else float("inf")
    return row


def _regenerate(argv) -> dict:
    """Re-run the whole sweep from the seeds and rewrite population.csv.  Expensive."""
    print("[8] creq_sweep --regenerate : re-running the full sweep "
          f"({len(CELLS)} cells x {N_SEEDS} seeds).  This is tier-2 (~90 core-minutes) "
          "and rewrites results/creq/population.csv.")
    path = results_path(POP_CSV)
    n_ok = 0
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        wr.writeheader()
        for fam, sz in CELLS:
            for sd in range(SEED0, SEED0 + N_SEEDS):
                try:
                    row = _measure_instance(fam, sz, sd)
                except Exception as e:                       # noqa: BLE001
                    row = dict(family=fam, size=sz, seed=sd, status="CENSORED",
                               censor_note=str(e)[:200])
                n_ok += int(row.get("status") == "ok")
                wr.writerow({k: (int(v) if isinstance(v, bool) else v)
                             for k, v in row.items() if k in FIELDS})
    print(f"    wrote {rel(path)} ({n_ok} completed).  Re-run "
          "`python -m experiments 8` to produce the analysis outputs.")
    return dict(regenerated=str(path), completed=n_ok)


if __name__ == "__main__":
    import sys
    run(sys.argv[1:])
