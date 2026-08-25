"""Experiment 21 -- the deficit PROFILE on a 12-instance stratified subsample (section V-D).

Claims: DP-ANCHOR, DP-SUBSAMPLE-BIND, DP-RATIO-RANGE, DP-COST.
Output: results/deficit_profile/s30_instances.jsonl  (the 12 committed per-instance rows)
        results/deficit_profile/s30_ladders.csv      (the long-form (label, d, N_d, term) table)
        results/deficit_profile/s30_anchor.json      (the reference-instance anchor record)
        results/deficit_profile/s30_verify.json      (the ORIGINATING lane's verification record)
        results/deficit_profile/s30_meta.json        (the declared plan + budgets, written FIRST)
        results/deficit_profile/verify.json          (THIS repo's own verification pass)

WHAT QUESTION THIS ANSWERS.  Experiment 18 (``deficit_counts``) established, ON THE REFERENCE
INSTANCE, that the discrimination condition's binding term is the d = 1 rung of the ladder

    term(d) = ln(N_d / #path) / (d * abar),          d = 1 .. m-1,   m = L+2              (33)

not the d = m-1 scatter term (29) the published condition uses -- 18.11 against 3.699, a factor
of 4.90.  Nothing in Exp 18 said whether that SHAPE (argmax at d = 1, ladder monotone
decreasing) is a property of the construction or an accident of one graph -- and the reference
instance is exactly the wrong instance to generalise from, because Exp 19/20 both found it to be
the MINORITY case on the population (its c = 4 verdict and its delta_min > 0 are both atypical
of its own cell).

This experiment answers it on a declared 12-instance stratified subsample: sizes spanning each
family's range at one seed apiece, plus a second seed at the customary sparse N = 60 cell.

  RESULT.  11 of 11 COMPLETED instances bind at d = 1 with a STRICTLY MONOTONE DECREASING
  ladder -- 4/4 sparse, 2/2 dense, 2/2 grid, 3/3 mtn_knn, over L = 3..19.  The reference's
  shape is NOT a reference-instance property.  One instance CENSORED (grid k = 12, m = 24) on
  the declared 4.0 GB memory cap; it is reported, not dropped and not substituted, and it is
  the DEEPEST cell in the plan -- so the single coverage gap sits at the top of the depth
  range, where a shape change would most plausibly hide.

  The d=1/scatter ratio ranges 2.70 .. 8.42 (median 4.66; the reference is 4.90).  So the
  paper's "factor of five" is a MIDDLE, not a constant -- but it is ONE-SIDED: on every
  completed instance the published condition UNDERSTATES the requirement and never overstates
  it, which is the load-bearing half of the claim.

THE d = 2 RUNG -- the printed-digit history, recorded here rather than quietly fixed.
An earlier manuscript draft printed the d = 2 rung as 11.25.  The computed rung is
11.244618410987663, i.e. 11.24 at two places under any rounding convention.  The originating
lane could not settle it (it had no independent pin on N_2) and said so.  It was settled FROM
THE DEPOSIT SIDE, using THIS repository's own committed results/counts/deficit.json:
N_2 = 1,320,177,088,811,008 with #path = 898,048 reproduces all nine printed rungs at two
places WITH the corrected digit.  The printed 11.25 was a transcription slip on a NON-BINDING
rung (the argmax is d = 1) and the manuscript now prints 11.24.  The manuscript also now prints
the crossing margin as 8.4475 rather than 8.447 (CNT-CROSS deposits it exactly:
8.447466352389776) so that a reader reconstructing the ladder from the paper's own printed
inputs lands on 11.24 and not on 11.2452 -> 11.25.  ``s30_anchor.json`` is the ORIGINATING
lane's record and is committed UNMODIFIED, so its ``verdict`` field still reads
DOES_NOT_REPRODUCE against the superseded 11.25 draft; this experiment re-checks the ladder
against the CORRECTED printed values and reports 22/22.

PROVENANCE -- what is REUSED and what is PORTED.  The originating lane could not find the
corrected N_d frontier DP on its own box and declared, before measuring, that it had
re-implemented rather than reused it.  In THIS repository the DP is present and committed:
``spnn.counting.count_degree2_by_deficit`` (Exp 18's counter), validated against
``spnn.counting.brute_degree2_by_deficit``.  So the port is a genuine reuse -- and the
verification pass below RECOUNTS the committed profiles with it, exactly, on real instances.

The default ``run()`` is the VERIFICATION pass and takes ~1 minute:
  1. the reference anchor re-derived from this repo's OWN results/counts/deficit.json;
  2. every subsample instance re-anchored (N/|E|/s/t/L/W*/abar) against the committed
     campaign population results/creq/population.csv -- all 12, censored one included;
  3. the exact integer identity N_(m-1) == scatter_counts' micro_aux, read off that same
     committed campaign CSV -- independent code, no recount needed;
  4. ladder / argmax / monotonicity / ratio recomputed from the committed N_d, and the
     long-form s30_ladders.csv re-derived row for row;
  5. a NATIVE RECOUNT with this repo's own count_degree2_by_deficit on a declared cheap
     subset (``--recount-all`` does all 11 completed; the two biggest take ~20 s and ~4 min);
  6. GROUND TRUTH: the full profile re-derived by EXHAUSTIVE ENUMERATION of every m-edge
     degree-<=2 subgraph (brute_degree2_by_deficit) on the two smallest REAL instances --
     agreement at every d, including the d = -1 cycle sector and the d = 2 rung.
``run(["--regenerate"])`` re-runs the full 12-instance sweep natively (tier-2) under the
declared 240 s / 4.0 GB budgets and rewrites the committed rows.
"""

from __future__ import annotations

import csv
import json
import math
import os
import resource
import subprocess
import sys
import time

from spnn.counting import (brute_degree2_by_deficit, count_degree2_by_deficit,
                           count_paths_by_length)

from experiments._base import rel, results_path, write_json
from experiments.cycle_prevalence import _build, _terminals

# --- the DECLARED plan and budgets (stated in advance, not fitted afterwards) ---
SUBSAMPLE = [("sparse", 20, 1), ("sparse", 60, 1), ("sparse", 60, 2), ("sparse", 100, 1),
             ("dense", 20, 1), ("dense", 60, 1),
             ("grid", 4, 1), ("grid", 8, 1), ("grid", 12, 1),
             ("mtn_knn", 20, 1), ("mtn_knn", 60, 1), ("mtn_knn", 140, 1)]
WALL_INSTANCE_S = 240.0          # per-instance wall budget
MEM_CAP_GB = 4.0                 # RLIMIT_AS in the worker process; a breach is MEMCAP
FAMILIES = ["sparse", "dense", "grid", "mtn_knn"]

# instances the verification pass recounts natively by default (deposited wall < 5 s each)
RECOUNT_DEFAULT = ["sparse_N20_seed1", "dense_N20_seed1", "grid_k4_N16_seed1",
                   "grid_k8_N64_seed1", "mtnknn_N20_seed1", "mtnknn_N60_seed1"]
# instances small enough for FULL-SPACE exhaustive enumeration (C(|E|+2, m) subsets)
BRUTE = ["grid_k4_N16_seed1", "sparse_N20_seed1"]

# the reference ladder AS THE MANUSCRIPT NOW PRINTS IT (d = 2 corrected 11.25 -> 11.24)
PRINTED_LADDER = {1: 18.11, 2: 11.24, 3: 8.68, 4: 7.22, 5: 6.23,
                  6: 5.46, 7: 4.82, 8: 4.25, 9: 3.70}
PRINTED_D2_SUPERSEDED = 11.25    # the earlier draft's transcription slip, kept on the record
REFERENCE_CROSSING = 3.699428677298893      # == CNT-CROSS, the section V-B crossing c
REFERENCE_PATH_MICRO = 898048               # 1754 s-t paths x 2^(L+1)
# the five N_d transcribed in the originating exchange, to the precision they were given
TRANSCRIBED = {-1: 1.040e8, 0: 1.239e11, 1: 2.165e13, 8: 6.646e19, 9: 3.352e19}

TOL = 1e-9


# ---------------------------------------------------------------------------
# deposits
# ---------------------------------------------------------------------------
def _instances():
    path = results_path("deficit_profile/s30_instances.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    meta = json.load(open(results_path("deficit_profile/s30_meta.json"), encoding="utf-8"))
    # the chunk guard: _meta.json is written LAST and its n_instances must match
    if meta["n_instances"] != len(rows):
        raise AssertionError(f"chunk guard: meta says {meta['n_instances']}, "
                             f"{len(rows)} rows on disk")
    declared = [tuple(x) for x in meta["declared_subsample"]]
    if declared != [(f, s, sd) for f, s, sd in SUBSAMPLE]:
        raise AssertionError("the committed plan is not the declared subsample")
    return rows, meta


def _population():
    """The committed campaign population, keyed by (family, size, seed)."""
    out = {}
    for r in csv.DictReader(open(results_path("creq/population.csv"), encoding="utf-8")):
        out[(r["family"], int(r["size"]), int(r["seed"]))] = r
    return out


def _f(x):
    return float(x) if x not in ("", None, "nan") else float("nan")


def _ln_big(x: int) -> float:
    """log of an arbitrarily large Python int, without float overflow."""
    b = x.bit_length()
    if b < 900:
        return math.log(x)
    shift = b - 900
    return math.log(x >> shift) + shift * math.log(2.0)


def ladder_from_profile(Nd, path_micro: int, alphabar: float, m: int):
    """term(d) = ln(N_d / #path) / (d * abar) for d = 1 .. m-1 -- the (33) grading."""
    out = {}
    lnp = _ln_big(int(path_micro))
    for d in range(1, m):
        v = int(Nd[str(d)]) if isinstance(Nd, dict) and str(d) in Nd else int(Nd[d])
        out[d] = (_ln_big(v) - lnp) / (d * alphabar)
    return out


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ---------------------------------------------------------------------------
# THE REFERENCE ANCHOR -- re-derived from THIS repo's own Exp 18 deposit
# ---------------------------------------------------------------------------
def anchor():
    """Re-derive the reference ladder from results/counts/deficit.json and check it against
    the manuscript's printed rungs (d = 2 CORRECTED to 11.24), the section V-B crossing, the
    five transcribed counts, and the committed s30_anchor.json from the originating lane."""
    d18 = json.load(open(results_path("counts/deficit.json"), encoding="utf-8"))
    s30 = json.load(open(results_path("deficit_profile/s30_anchor.json"), encoding="utf-8"))
    out = dict(instance=d18["instance"], checks=[],
               source="results/counts/deficit.json (experiment 18, this repository)")

    def chk(name, got, want, kind="float", tol=5e-3):
        ok = (got == want) if kind == "int" else (abs(float(got) - float(want)) <= tol)
        out["checks"].append(dict(name=name, got=str(got), want=str(want), agree=bool(ok)))
        return ok

    Nd = d18["N_d"]
    m, L, abar = int(d18["m"]), int(d18["L"]), float(d18["abar"])
    path_micro = int(d18["path_micro"])
    chk("L", L, 8, "int")
    chk("m = L+2", m, 10, "int")
    chk("#path microstates", path_micro, REFERENCE_PATH_MICRO, "int")
    chk("N_(m-1) == scatter_counts micro_aux", 1 if d18["crosscheck_Nm1_eq_scatter_micro_aux"]
        else 0, 1, "int")

    # the N_d themselves must be the same integers the originating lane measured
    same_Nd = all(str(int(Nd[str(d)])) == s30["N_d"][str(d)] for d in range(-1, m))
    chk("N_d identical to the originating lane, all 11 (EXACT)", 1 if same_Nd else 0, 1, "int")

    # the five counts transcribed in the originating exchange (to their printed precision)
    for d, want in TRANSCRIBED.items():
        chk(f"N_{d} vs the transcribed {want:.4g} (ln)", _ln_big(int(Nd[str(d)])),
            math.log(want), tol=5e-4)

    ladder = ladder_from_profile(Nd, path_micro, abar, m)
    for d in range(1, m):
        chk(f"ladder d={d} (printed {PRINTED_LADDER[d]})", round(ladder[d], 2),
            PRINTED_LADDER[d], tol=5e-3)
    chk("ladder d=m-1 == the section V-B crossing c", ladder[m - 1], REFERENCE_CROSSING,
        tol=1e-9)

    argmax = max(range(1, m), key=lambda d: ladder[d])
    monotone = all(ladder[d] > ladder[d + 1] + TOL for d in range(1, m - 1))
    chk("argmax d", argmax, 1, "int")
    chk("ladder strictly monotone decreasing", 1 if monotone else 0, 1, "int")
    ratio = ladder[1] / ladder[m - 1]
    chk("d=1/scatter ratio", ratio, 4.895363158564106, tol=1e-9)

    out["ladder"] = {str(d): ladder[d] for d in range(1, m)}
    out["argmax_d"] = argmax
    out["monotone_decreasing"] = monotone
    out["ratio_d1_scatter"] = ratio
    out["alphabar"] = abar
    out["d2_rung"] = dict(
        computed=ladder[2], printed_now=PRINTED_LADDER[2],
        printed_superseded=PRINTED_D2_SUPERSEDED,
        N_2=str(int(Nd["2"])), path_micro=str(path_micro),
        note="an earlier manuscript draft printed 11.25; the computed rung is 11.244618410987663 "
             "-> 11.24 at two places under any rounding convention. Settled from the deposit "
             "side with this repository's own results/counts/deficit.json: N_2 = 1,320,177,088,"
             "811,008 and #path = 898,048 reproduce all nine rungs at 2 dp with the corrected "
             "digit. The rung is NON-BINDING (the argmax is d = 1), so nothing the paper "
             "concludes depended on it. The manuscript now prints 11.24, and prints the "
             "crossing margin as 8.4475 (CNT-CROSS deposits 8.447466352389776) so the ladder "
             "reconstructs from the paper's own printed inputs.")
    out["originating_lane_anchor"] = dict(
        file="results/deficit_profile/s30_anchor.json",
        verdict=s30.get("verdict"),
        selftest_dp_vs_brute=f"{s30['selftest']['dp_vs_brute_pass']}/"
                             f"{s30['selftest']['dp_vs_brute_total']}",
        selftest_weight=f"{s30['selftest']['weight_pass']}/{s30['selftest']['weight_total']}",
        n_checks=len(s30["checks"]),
        n_agree=sum(1 for c in s30["checks"] if c["ok"]),
        note="committed UNMODIFIED. Its single disagreement, and hence its "
             "verdict=DOES_NOT_REPRODUCE, is the SUPERSEDED printed 11.25; against the "
             "corrected printed ladder the anchor agrees on all 22 checks (below).")
    out["n_checks"] = len(out["checks"])
    out["all_agree"] = all(c["agree"] for c in out["checks"])
    return out


# ---------------------------------------------------------------------------
# per-instance re-derivation from the committed profile
# ---------------------------------------------------------------------------
def _derive(row):
    """Recompute ladder / argmax / monotonicity / ratio from a committed row's N_d."""
    m, abar = int(row["m"]), float(row["alphabar"])
    ladder = ladder_from_profile(row["N_d"], int(row["n_path_micro"]), abar, m)
    argmax = max(range(1, m), key=lambda d: ladder[d])
    monotone = all(ladder[d] > ladder[d + 1] + TOL for d in range(1, m - 1))
    return dict(ladder=ladder, argmax_d=argmax, monotone_decreasing=monotone,
                term_d1=ladder[1], term_scatter=ladder[m - 1],
                ratio_d1_scatter=ladder[1] / ladder[m - 1])


def _ladders_csv_rows(rows):
    """The long-form (label, family, size, seed, L, m, d, N_d, term) table."""
    out = []
    for r in rows:
        if r["status"] != "ok":
            continue
        m = int(r["m"])
        der = _derive(r)
        for d in range(-1, m):
            term = "" if d < 1 else repr(der["ladder"][d])
            out.append([r["label"], r["family"], r["size"], r["seed"], r["L"], m, d,
                        r["N_d"][str(d)], term])
    return out


# ---------------------------------------------------------------------------
# THE VERIFICATION PASS
# ---------------------------------------------------------------------------
def run(argv=None) -> dict:
    argv = list(argv or [])
    if "--regenerate" in argv:
        return _regenerate(argv)
    recount_all = "--recount-all" in argv
    no_brute = "--no-brute" in argv

    rows, meta = _instances()
    pop = _population()
    checks, failures = [], []

    def chk(label, name, ok, got=None, want=None):
        checks.append(dict(label=label, name=name, ok=bool(ok), got=str(got), want=str(want)))
        if not ok:
            failures.append(f"{label}: {name} (got {got}, want {want})")
        return ok

    # ---- 0. the reference anchor, from THIS repo's own Exp 18 deposit
    anc = anchor()
    for c in anc["checks"]:
        chk("reference", c["name"], c["agree"], c["got"], c["want"])

    # ---- 1. every subsample instance re-anchored against the committed campaign CSV
    for r in rows:
        lab = r["label"]
        p = pop.get((r["family"], int(r["size"]), int(r["seed"])))
        if p is None:
            chk(lab, "present in the campaign population", False)
            continue
        chk(lab, "N/|E| vs campaign", int(p["N"]) == r["N"] and int(p["E"]) == r["E"],
            (r["N"], r["E"]), (p["N"], p["E"]))
        chk(lab, "(s,t) vs campaign", int(p["s"]) == r["s"] and int(p["t"]) == r["t"],
            (r["s"], r["t"]), (p["s"], p["t"]))
        chk(lab, "L vs campaign", int(p["L"]) == r["L"], r["L"], p["L"])
        chk(lab, "W* vs campaign", abs(_f(p["wsum"]) - r["wsum"]) <= 1e-12,
            r["wsum"], p["wsum"])
        chk(lab, "abar vs campaign", abs(_f(p["alphabar"]) - r["alphabar"]) <= 1e-12,
            r["alphabar"], p["alphabar"])
        chk(lab, "m == L+2", int(r["m"]) == int(r["L"]) + 2, r["m"], r["L"] + 2)

    # ---- 2. the EXACT integer identity, off the committed campaign CSV (independent code)
    id_ok = 0
    for r in rows:
        if r["status"] != "ok":
            continue
        p = pop[(r["family"], int(r["size"]), int(r["seed"]))]
        want = int(p["scatter_micro_aux"])
        got = int(r["N_d"][str(int(r["m"]) - 1)])
        if chk(r["label"], "N_(m-1) == scatter_counts micro_aux (EXACT)", got == want,
               got, want):
            id_ok += 1

    # ---- 3. ladder / argmax / monotonicity / ratio recomputed from the committed N_d
    derived, binds, monotones, ratios = {}, 0, 0, []
    for r in rows:
        lab = r["label"]
        if r["status"] != "ok":
            chk(lab, "censored row carries its resource + note",
                bool(r.get("censor_note")) and r.get("peak_rss_mib") is not None,
                r.get("censor_note"), "a censor note")
            continue
        der = _derive(r)
        derived[lab] = der
        m = int(r["m"])
        chk(lab, "N_d covers d = -1 .. m-1, every entry a positive integer",
            sorted(int(k) for k in r["N_d"]) == list(range(-1, m))
            and all(int(v) > 0 for v in r["N_d"].values()), len(r["N_d"]), m + 1)
        chk(lab, "N_0 >= #path (the path is a d = 0 member)",
            int(r["N_d"]["0"]) >= int(r["n_path_micro"]),
            r["N_d"]["0"], r["n_path_micro"])
        chk(lab, "ladder recomputes from N_d",
            all(abs(der["ladder"][int(d)] - v) <= 1e-9 for d, v in r["ladder"].items()))
        chk(lab, "argmax d == committed", der["argmax_d"] == int(r["argmax_d"]),
            der["argmax_d"], r["argmax_d"])
        chk(lab, "monotone flag == committed",
            der["monotone_decreasing"] == bool(r["monotone_decreasing"]))
        chk(lab, "ratio == committed",
            abs(der["ratio_d1_scatter"] - r["ratio_d1_scatter"]) <= 1e-9,
            der["ratio_d1_scatter"], r["ratio_d1_scatter"])
        binds += int(der["argmax_d"] == 1)
        monotones += int(der["monotone_decreasing"])
        ratios.append(der["ratio_d1_scatter"])

    # ---- 4. the long-form ladders CSV re-derived row for row
    want_csv = list(csv.reader(open(results_path("deficit_profile/s30_ladders.csv"),
                                    encoding="utf-8")))
    got_csv = [[str(x) for x in row] for row in _ladders_csv_rows(rows)]
    chk("deposit", "s30_ladders.csv re-derives row for row",
        [r for r in want_csv[1:]] == got_csv, len(got_csv), len(want_csv) - 1)

    # ---- 5. NATIVE RECOUNT with this repo's own committed DP
    recount = []
    targets = [r for r in rows if r["status"] == "ok"
               and (recount_all or r["label"] in RECOUNT_DEFAULT)]
    for r in targets:
        n, edges, w = _build(r["family"], int(r["size"]), int(r["seed"]))
        s, t, opt = _terminals(n, edges, w)
        t0 = time.perf_counter()
        Nd, stats = count_degree2_by_deficit(n, edges, s, t, len(opt) - 1)
        wall = time.perf_counter() - t0
        got = {str(k): str(v) for k, v in sorted(Nd.items())}
        ok = chk(r["label"], "N_d RECOUNTED natively with spnn.counting (EXACT, all d)",
                 got == r["N_d"], "match" if got == r["N_d"] else "MISMATCH", "match")
        # NOTE: the recount wall is printed, not written -- results/ files are committed
        # byte-identical across re-runs, so no timing goes into verify.json.
        print(f"      recount {r['label']:<22} {'OK' if ok else 'MISMATCH'}  "
              f"{stats['max_states']:>9,} states  {wall:7.1f}s", flush=True)
        recount.append(dict(label=r["label"], ok=ok, max_states=stats["max_states"]))

    # ---- 6. GROUND TRUTH: exhaustive enumeration of the full space, on real instances
    brute = []
    if not no_brute:
        for lab in BRUTE:
            r = next(x for x in rows if x["label"] == lab)
            n, edges, w = _build(r["family"], int(r["size"]), int(r["seed"]))
            s, t, opt = _terminals(n, edges, w)
            t0 = time.perf_counter()
            res = brute_degree2_by_deficit(n, edges, s, t, len(opt) - 1)
            Nd = res[0] if isinstance(res, tuple) else res
            got = {str(k): str(v) for k, v in sorted(Nd.items())}
            ok = chk(lab, "full profile vs EXHAUSTIVE ENUMERATION of every edge subset, "
                          "at every d", got == r["N_d"],
                     "match" if got == r["N_d"] else "MISMATCH", "match")
            print(f"      brute   {lab:<22} {'OK' if ok else 'MISMATCH'}  "
                  f"{time.perf_counter() - t0:7.1f}s", flush=True)
            brute.append(dict(label=lab, ok=ok,
                              subsets=math.comb(int(r["E"]) + 2, int(r["m"]))))

    # ---- the summary
    completed = [r for r in rows if r["status"] == "ok"]
    censored = [r for r in rows if r["status"] != "ok"]
    by_family = {}
    for fam in FAMILIES:
        fr = [r for r in rows if r["family"] == fam]
        fc = [r for r in fr if r["status"] == "ok"]
        by_family[fam] = dict(
            n=len(fr), completed=len(fc), censored=len(fr) - len(fc),
            binds_at_d1=sum(1 for r in fc if derived[r["label"]]["argmax_d"] == 1),
            monotone=sum(1 for r in fc if derived[r["label"]]["monotone_decreasing"]))

    summ = dict(
        n_rows=len(rows), n_completed=len(completed), n_censored=len(censored),
        n_checks=len(checks), n_pass=sum(1 for c in checks if c["ok"]),
        n_fail=len(failures), failures=failures,
        declared_budgets=dict(wall_instance_s=WALL_INSTANCE_S, mem_cap_gb=MEM_CAP_GB,
                              max_concurrent=meta["max_concurrent"]),
        binds_at_d1=f"{binds}/{len(completed)}",
        monotone_decreasing=f"{monotones}/{len(completed)}",
        by_family=by_family,
        ratio=dict(min=min(ratios), max=max(ratios), median=_median(ratios),
                   reference=anc["ratio_d1_scatter"],
                   one_sided_all_above_1=all(x > 1.0 for x in ratios),
                   per_instance={r["label"]: derived[r["label"]]["ratio_d1_scatter"]
                                 for r in completed}),
        depth_range=[min(int(r["L"]) for r in completed),
                     max(int(r["L"]) for r in completed)],
        censored=[dict(label=r["label"], L=r["L"], m=r["m"], resource="memory",
                       note=r.get("censor_note"), wall_s=r.get("wall_s"),
                       peak_rss_mib=r.get("peak_rss_mib"),
                       deepest_in_plan=int(r["L"]) == max(int(x["L"]) for x in rows))
                   for r in censored],
        exact_identity=f"{id_ok}/{len(completed)}",
        native_recount=recount, exhaustive_enumeration=brute,
        cost=dict(
            state_bound=dict(
                note="the random families: cost is 3^frontier and MEMCAPs before it walls",
                worst=max(((r["label"], r["dp_max_states"], r["peak_rss_mib"])
                           for r in completed), key=lambda x: x[1])),
            field_bound=dict(
                note="the deep, low-frontier families: each state carries an "
                     "(m+1)(m+3)-field polynomial, so it walls before it MEMCAPs",
                worst=max(((r["label"], r["dp_max_states"], r["wall_s"])
                           for r in completed), key=lambda x: x[2])),
            per_instance={r["label"]: dict(L=r["L"], m=r["m"], frontier=r["dp_frontier"],
                                           states=r["dp_max_states"],
                                           coeff_bits=r["dp_max_coeff_bits"],
                                           wall_s=r["wall_s"],
                                           peak_rss_mib=r["peak_rss_mib"])
                          for r in completed},
            vs_stage28={r["label"]: dict(
                s28_states=int(pop[(r["family"], int(r["size"]), int(r["seed"]))]["dp_states"]),
                s30_states=r["dp_max_states"],
                s28_scatter_wall_s=_f(pop[(r["family"], int(r["size"]),
                                           int(r["seed"]))]["scatter_wall_s"]),
                s30_wall_s=r["wall_s"])
                for r in completed}),
        d2_rung=anc["d2_rung"],
        reference_anchor=dict(n_checks=anc["n_checks"], all_agree=anc["all_agree"],
                              argmax_d=anc["argmax_d"],
                              ratio=anc["ratio_d1_scatter"],
                              originating_lane=anc["originating_lane_anchor"]),
        checks=checks,
    )
    out = write_json("deficit_profile/verify.json", summ)

    print(f"[21] deficit_profile -> {rel(out)}")
    print(f"    reference anchor (from results/counts/deficit.json): {anc['n_checks']} checks, "
          f"all agree = {anc['all_agree']}; argmax d={anc['argmax_d']}, "
          f"ratio {anc['ratio_d1_scatter']:.4f}")
    print(f"    d=2 rung: computed {anc['d2_rung']['computed']:.9f} -> printed "
          f"{anc['d2_rung']['printed_now']} (an earlier draft printed "
          f"{PRINTED_D2_SUPERSEDED}; superseded, non-binding rung)")
    print(f"    SHAPE: binds at d=1 {summ['binds_at_d1']}, strictly monotone decreasing "
          f"{summ['monotone_decreasing']}, over L = {summ['depth_range'][0]}..{summ['depth_range'][1]}")
    for fam in FAMILIES:
        f2 = by_family[fam]
        print(f"      {fam:<8} completed {f2['completed']}/{f2['n']}  binds "
              f"{f2['binds_at_d1']}  monotone {f2['monotone']}  censored {f2['censored']}")
    for c in summ["censored"]:
        print(f"    CENSORED (reported, not dropped, nothing substituted): {c['label']} "
              f"(L={c['L']}, m={c['m']}) -- {c['note']}; deepest cell in the plan: "
              f"{c['deepest_in_plan']}")
    rt = summ["ratio"]
    print(f"    d=1/scatter ratio: {rt['min']:.2f} .. {rt['max']:.2f}, median "
          f"{rt['median']:.2f} (reference {rt['reference']:.2f}); one-sided "
          f"(>1 on every completed instance): {rt['one_sided_all_above_1']}")
    print(f"    exact identity N_(m-1) == scatter_counts micro_aux: {summ['exact_identity']}")
    print(f"    native recount with spnn.counting: "
          f"{sum(1 for x in recount if x['ok'])}/{len(recount)}"
          + ("" if recount_all else "  (declared subset; --recount-all does all 11)"))
    detail = "; ".join("{} ({:,} subsets)".format(x["label"], x["subsets"]) for x in brute)
    print(f"    exhaustive enumeration (full space, real instances): "
          f"{sum(1 for x in brute if x['ok'])}/{len(brute)}"
          + (f"  [{detail}]" if brute else ""))
    print(f"    CHECKS: {summ['n_pass']}/{summ['n_checks']} pass, {summ['n_fail']} fail")

    # ---- ASSERT the headline figures the manuscript prints
    assert anc["all_agree"], f"reference anchor FAILED: {anc['checks']}"
    assert not failures, f"{len(failures)} verification failures: {failures[:5]}"
    assert (len(completed), len(censored)) == (11, 1), (len(completed), len(censored))
    assert binds == 11 and monotones == 11, (binds, monotones)
    assert by_family["sparse"]["binds_at_d1"] == 4 and by_family["sparse"]["completed"] == 4
    assert by_family["dense"]["binds_at_d1"] == 2 and by_family["dense"]["completed"] == 2
    assert by_family["grid"]["binds_at_d1"] == 2 and by_family["grid"]["censored"] == 1
    assert by_family["mtn_knn"]["binds_at_d1"] == 3 and by_family["mtn_knn"]["completed"] == 3
    assert id_ok == 11, id_ok
    assert round(rt["min"], 2) == 2.70 and round(rt["max"], 2) == 8.42, rt
    assert round(rt["median"], 2) == 4.66, rt["median"]
    assert rt["one_sided_all_above_1"], rt
    assert summ["depth_range"] == [3, 19], summ["depth_range"]
    assert summ["censored"][0]["label"] == "grid_k12_N144_seed1", summ["censored"]
    assert summ["censored"][0]["deepest_in_plan"], summ["censored"]
    assert all(x["ok"] for x in recount), recount
    assert all(x["ok"] for x in brute), brute
    return summ


# ---------------------------------------------------------------------------
# THE NATIVE SWEEP (tier-2; --regenerate)
# ---------------------------------------------------------------------------
def measure_instance(family, size, seed):
    """One instance end to end, with this repo's own committed counters."""
    n, edges, w = _build(family, size, seed)
    s, t, opt = _terminals(n, edges, w)
    row = dict(family=family, size=size, seed=seed,
               label=_label(family, size, seed, n), N=n, E=len(edges),
               mean_degree=2 * len(edges) / n, s=s, t=t, terminal_fallback=0)
    if opt is None:
        row["status"] = "UNREACHABLE"
        return row
    L = len(opt) - 1
    m = L + 2
    B = 2.0 * max(float(x) for x in w)
    lut = {}
    for (a, b), ww in zip(edges, w):
        lut[(int(a), int(b))] = lut[(int(b), int(a))] = float(ww)
    wsum = sum(lut[(opt[i], opt[i + 1])] for i in range(L))
    Et = wsum / B
    row.update(L=L, m=m, wsum=wsum, B=B, Etilde=Et, margin_c1=(L + 1) - 2.0 * Et,
               alphabar=1.0 - 2.0 * Et / (L + 1),
               max_edge_weight=max(float(x) for x in w),
               min_edge_weight=min(float(x) for x in w), status="ok")
    t0 = time.perf_counter()
    cnts, minw, _wit = count_paths_by_length(n, edges, w, s, t, L)
    row["n_path_edge"] = int(cnts[L])
    row["n_path_micro"] = str(int(cnts[L]) * (1 << (L + 1)))
    row["path_min_weight"] = float(minw[L])
    Nd, stats = count_degree2_by_deficit(n, edges, s, t, L)
    row["wall_s"] = time.perf_counter() - t0
    row["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    row["N_d"] = {str(d): str(v) for d, v in sorted(Nd.items())}
    row["dp_max_states"] = stats["max_states"]
    row["dp_frontier"] = stats["order_max_frontier"]
    der = _derive(row)
    row["ladder"] = {str(d): v for d, v in der["ladder"].items()}
    row.update(argmax_d=der["argmax_d"], ladder_max=der["ladder"][der["argmax_d"]],
               term_d1=der["term_d1"], term_scatter=der["term_scatter"],
               ratio_d1_scatter=der["ratio_d1_scatter"],
               monotone_decreasing=der["monotone_decreasing"],
               binds_at_d1=der["argmax_d"] == 1)
    return row


def _label(family, size, seed, n):
    if family == "grid":
        return f"grid_k{size}_N{n}_seed{seed}"
    if family == "mtn_knn":
        return f"mtnknn_N{size}_seed{seed}"
    return f"{family}_N{size}_seed{seed}"


def _worker():
    """Child process: RLIMIT_AS enforced, one JSON row to stdout. A breach is MEMCAP."""
    family, size, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    soft = int(MEM_CAP_GB * (1 << 30))
    resource.setrlimit(resource.RLIMIT_AS, (soft, soft))
    try:
        row = measure_instance(family, size, seed)
    except MemoryError:
        row = dict(family=family, size=size, seed=seed, status="MEMCAP",
                   censor_note=f"RLIMIT_AS {MEM_CAP_GB} GB breached")
    print("@@ROW@@" + json.dumps(row))


def _regenerate(argv) -> dict:
    """Re-run the declared 12-instance sweep natively under the declared budgets.

    Each instance runs in a child process with RLIMIT_AS = 4.0 GB and a 240 s wall, so a
    breach CENSORS that instance and is recorded -- never dropped, never substituted, never
    retried at a raised budget.  _meta.json is written LAST so a cut leaves no analysable
    chunk (the chunk guard in _instances()).
    """
    rows, t0 = [], time.perf_counter()
    for family, size, seed in SUBSAMPLE:
        st = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, "-m", "experiments.deficit_profile",
                                "worker", family, str(size), str(seed)],
                               capture_output=True, text=True,
                               timeout=WALL_INSTANCE_S + 90,
                               env=dict(os.environ, PYTHONPATH=os.getcwd()))
            tag = [ln for ln in p.stdout.splitlines() if ln.startswith("@@ROW@@")]
            if tag:
                row = json.loads(tag[-1][len("@@ROW@@"):])
            else:
                row = dict(family=family, size=size, seed=seed, status="MEMCAP",
                           censor_note=f"RLIMIT_AS {MEM_CAP_GB} GB breached")
        except subprocess.TimeoutExpired:
            row = dict(family=family, size=size, seed=seed, status="WALL",
                       censor_note=f"exceeded the declared {WALL_INSTANCE_S} s wall")
        row["driver_wall_s"] = time.perf_counter() - st
        rows.append(row)
        print(f"  {row.get('label', f'{family}/{size}/{seed}')}: {row['status']} "
              f"({row['driver_wall_s']:.1f}s)", flush=True)

    path = results_path("deficit_profile/s30_instances.jsonl")
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    lp = results_path("deficit_profile/s30_ladders.csv")
    with open(lp, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["label", "family", "size", "seed", "L", "m", "d", "N_d", "term"])
        wr.writerows(_ladders_csv_rows(rows))
    # _meta.json LAST -- the chunk guard
    write_json("deficit_profile/s30_meta.json", dict(
        slug="spnn-stage30-deficit-profile-subsample", n_instances=len(SUBSAMPLE),
        n_rows=len(rows), declared_subsample=[list(x) for x in SUBSAMPLE],
        wall_instance_s=WALL_INSTANCE_S, mem_cap_gb=MEM_CAP_GB, max_concurrent=1,
        total_wall_s=time.perf_counter() - t0,
        note="regenerated natively in this repository by experiments/deficit_profile.py "
             "--regenerate; counters are spnn.counting.count_degree2_by_deficit and "
             "count_paths_by_length"))
    print(f"[21] regenerated {len(rows)} rows -> {rel(path)} "
          f"({time.perf_counter() - t0:.0f}s)")
    return run([a for a in argv if a != "--regenerate"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        _worker()
    else:
        run(sys.argv[1:])
