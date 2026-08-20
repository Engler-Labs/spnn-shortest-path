"""Experiment 13 -- the mass-ratio null pilot.

Claims: MR-NULL, MR-ACT (CLAIMS.tsv, section V-D, Tier 2).
Output: results/mass/null.json
        (derived from the committed results/mass/mass_ratio_pilot_by_unit.csv)

This is **the only consumer of ``spnn.sim.mass_class``** in the repo (A11).  The
Ch. 4 discrimination condition

    c * (L - 1) * alpha_bar  >  ln( #scatter / #path )

compares the AGGREGATE Boltzmann mass on *chained* path-class states against the
mass on equal-activity *scatter* states -- two configurations that share the
matched shell (2L active, L beta pairs, no gamma conflict) and are separated
ONLY by their alpha count (L-1 for a chain, 0 for L disjoint edges), which is why
alpha is the lever.  ``mass_class`` is the pure predicate that names those classes
from the three pair counts, with no reference to the optimal path.

The pilot's job was to SIZE that measurement, not to make it: one operating point
(N=12, c=2, the bare sampler with the IRM off, run on the **variant**
i.e. unaugmented network), a short run per unit, and a read of how often the
sampler even visits the matched activity band.  The headline is an informative
FAILURE -- the classifiable classes never populate, so the mass ratio is
undefined at this operating point, and the sampler lives at Theta(N) activity far
above the 2L shell it would have to sit on.

  * MR-NULL: 72 units, 143,280,000 scored steps; the shell band (2L +/- 2) was
             occupied 2,116 steps (1.5e-5 of the run); path- and scatter-class
             mass are identically zero.
  * MR-ACT:  mean activity spans 25.0 to 43.5 -- above 2N = 24 -- the Theta(N)
             obstruction.

The full 143M-step run is Tier 2 and is NOT re-executed by default: ``run()``
does a cheap live smoke (a short ``simulate`` with the mass classifier engaged,
plus a direct ``mass_class`` self-check) and then derives + verifies the headline
from the committed by-unit CSV.  ``run(["--full"])`` re-executes the pilot design
serially into a separate ``mass_ratio_pilot_rerun.csv`` (seeded/expensive; it does
NOT overwrite the committed artefact).
"""

from __future__ import annotations

import argparse
import csv
import math
import time

import numpy as np

from spnn.compile import Params, compile_network
from spnn.graphs import dijkstra, random_graph
from spnn.sim import ideal_state, mass_class, simulate, state_counts

from experiments._base import RESULTS, rel, write_json

# ---------------------------------------------------------------- operating point
# lambda = 2*b_k + D + c = 0 is the corrected operating point; b_k is
# therefore c-dependent and is derived, never hardcoded.
D = 4.0
LAM = 0.0
ST_MODE = "always_on"
TAU = 10
IRM = None                 # disabled: the IRM distorts the stationary distribution
RECORD_EVERY = 200         # energy() is O(n_neurons); keep it off the hot path
DEG_TARGET = 3.0           # p = DEG_TARGET/(N-1); the REALISED degree is reported
BAND = 2                   # pre-filter halfwidth in active neurons around 2L

PILOT_C = 2.0              # the contested value: enrichment is already strong here
BURNIN = 10_000            # the band visits from empty are a TRANSIENT; drop them
NEFF_MIN_EVENTS = 30       # below this an indicator cannot estimate its own tau_int

# The committed pilot (MR-NULL / MR-ACT) and the file it lives in.
COMMITTED_CSV = RESULTS / "mass" / "mass_ratio_pilot_by_unit.csv"
RERUN_CSV = RESULTS / "mass" / "mass_ratio_pilot_rerun.csv"

# Full-run design, as the committed pilot was invoked (N=12, 10 graphs x 8 seeds;
# the committed CSV is a 72-unit prefix, cut at the wall by the per-unit writer).
FULL_N = 12
FULL_GRAPHS = 10
FULL_SEEDS = 8
FULL_STEPS = 2_000_000


def bias_for(c: float) -> float:
    return (LAM - D - c) / 2.0


# --------------------------------------------------------------- n_eff (inlined)
# ``n_eff``/``n_eff_indicator`` are the two tiny Sokal helpers the pilot's per-unit
# row schema carries; inlined here so the extraction stays self-contained.
def _n_eff(series) -> float:
    """Effective sample size via the integrated autocorrelation time (Sokal)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n == 0:
        return 0.0
    x = x - x.mean()
    v = float(np.dot(x, x) / n)
    if v <= 0:
        return float(n)
    tau_int = 0.5
    for lag in range(1, min(n // 3, 3000)):
        c = float(np.dot(x[:n - lag], x[lag:]) / ((n - lag) * v))
        if c <= 0:
            break
        tau_int += c
    return n / (2.0 * tau_int)


def _n_eff_indicator(series) -> float:
    """``n_eff`` for a 0/1 indicator, or NaN when the indicator is degenerate."""
    x = np.asarray(series, dtype=float)
    if len(x) == 0 or x.std() == 0:
        return float("nan")
    return _n_eff(x)


def _neff(indicator: np.ndarray, fallback: float) -> float:
    """n_eff on an indicator, but only when it fired often enough to estimate one."""
    fired = int(indicator.sum())
    if fired < NEFF_MIN_EVENTS:
        return float("nan")
    v = _n_eff_indicator(indicator)
    return v if math.isfinite(v) else fallback


# ------------------------------------------------------------- graph population
def build_graph(N: int, gseed: int):
    """A graph seed with a non-degenerate shortest path (L >= 2; L=1 is a direct edge)."""
    n, e, w = random_graph(N, DEG_TARGET / (N - 1), seed=gseed)
    path, cost = dijkstra(n, e, w, 0, N - 1)
    return n, e, w, path, cost


def graph_population(N: int, n_graphs: int, seed0: int = 1):
    """Walk seeds upward, skipping L=1, until ``n_graphs`` graphs are collected."""
    out, excluded, gs = [], 0, seed0
    while len(out) < n_graphs:
        n, e, w, path, cost = build_graph(N, gs)
        L = len(path) - 1
        if L < 2:
            excluded += 1
        else:
            out.append(dict(gseed=gs, n=n, e=e, w=w, path=path, cost=cost, L=L,
                            mean_deg=2.0 * len(e) / n))
        gs += 1
    return out, excluded


def make_net(g, c: float):
    # inhib off, no augmentation -> the "variant" (unaugmented) network MR-NULL ran on
    p = Params(D=D, bias=bias_for(c), alpha_scale=c, tau=TAU, st_mode=ST_MODE,
               inhib_kind="off")
    return compile_network(g["n"], g["e"], g["w"], 0, g["n"] - 1, p)


# ----------------------------------------------------------------- one work unit
def run_unit(job) -> dict:
    """One (graph x sampler seed) unit.  Returns a fully-populated row dict.

    The stationary rates are recomputed from the PER-STEP series with the first
    ``burnin`` steps (the ascent from the empty state) dropped -- the band is
    visited transiently on the way up, and the mass ratio is a stationary object.
    """
    g, c, steps, sseed = job["g"], job["c"], job["steps"], job["sseed"]
    band = job.get("band", BAND)
    burn = min(job.get("burnin", BURNIN), steps // 2)
    net = make_net(g, c)
    L = g["L"]
    t0 = time.time()
    tr = simulate(net, steps=steps, seed=sseed, irm_threshold=IRM,
                  record_every=RECORD_EVERY, track_paths=False, track_realised=0,
                  source=0, target=g["n"] - 1, mass_L=L, mass_band=band)
    wall = time.time() - t0

    ms_all, act_all = tr.mass_series, tr.active_series
    ms, act = ms_all[burn:], act_all[burn:]
    n = len(ms)
    band_ind = (ms > 0).astype(np.float64)
    path_ind = (ms == 2).astype(np.float64)
    scat_ind = (ms == 3).astype(np.float64)
    cons_ind = (ms >= 2).astype(np.float64)

    band_steps = int(band_ind.sum())
    matched_steps = int((act == 2 * L).sum())
    n_path = int(path_ind.sum())
    n_scat = int(scat_ind.sum())
    n_cons = int(cons_ind.sum())
    n_mixed = int((ms == 4).sum())
    n_closed = int((ms == 5).sum())
    transient_band = int((ms_all[:burn] > 0).sum())

    hist = np.bincount(act, minlength=net.n_neurons + 1)
    mean_act = float((np.arange(len(hist)) * hist).sum() / max(int(hist.sum()), 1))
    mode_act = int(np.argmax(hist))
    reached = np.nonzero(hist)[0]
    ne_act = _n_eff(act)

    return dict(
        stage=job["stage"], N=job["N"], c=c, bias=bias_for(c), lam=LAM,
        graph_seed=g["gseed"], L=L, opt_cost=float(g["cost"]), mean_deg=g["mean_deg"],
        n_edges=len(g["e"]), n_neurons=int(net.n_neurons),
        n_segments=int(net.n_segments), sampler_seed=sseed, steps=steps, burnin=burn,
        steps_scored=n, wall_s=round(wall, 3),
        steps_per_s=round(steps / wall, 1) if wall > 0 else "",
        band_lo=max(0, 2 * L - band), band_hi=2 * L + band, band_halfwidth=band,
        matched_shell=2 * L, band_steps=band_steps,
        transient_band_steps=transient_band, matched_steps=matched_steps,
        consistent_steps=n_cons, mass_path=n_path, mass_scatter=n_scat,
        mass_mixed=n_mixed, mass_closed=n_closed,
        band_frac=band_steps / n, matched_frac=matched_steps / n,
        classified_per_1e6=1e6 * n_cons / n, path_per_1e6=1e6 * n_path / n,
        scatter_per_1e6=1e6 * n_scat / n,
        n_eff_band=round(_neff(band_ind, ne_act), 2),
        n_eff_classified=round(_neff(cons_ind, ne_act), 2),
        n_eff_path=round(_neff(path_ind, ne_act), 2),
        n_eff_scatter=round(_neff(scat_ind, ne_act), 2),
        n_eff_active=round(ne_act, 2), n_eff_frac=round(ne_act / n, 6),
        mean_active=round(mean_act, 3), mode_active=mode_act,
        min_active_reached=int(reached.min()) if reached.size else "",
        max_active_reached=int(reached.max()) if reached.size else "",
        act_over_shell=round(mean_act / max(2 * L, 1), 3),
        st_mode=ST_MODE, tau=TAU, irm_threshold="none", record_every=RECORD_EVERY,
    )


# ------------------------------------------------------- the classifier, asserted
def classifier_selfcheck(N: int = 12, c: float = PILOT_C) -> dict:
    """*** The whole result rests on the classifier, so it is asserted, not assumed. ***

    A pilot whose headline is "zero classifiable states" is worthless if the
    classifier cannot recognise one.  Build the two classes BY HAND and check that
    ``mass_class`` names them, against Lemma 1's exact counts (2L / L / L-1 for a
    path; 2L / L / 0 for a scatter).  This is the direct ``spnn.sim.mass_class``
    consumer the smoke exercises before trusting the sampled zeros.
    """
    import collections
    g = graph_population(N, 1)[0][0]
    net, L, path = make_net(g, c), g["L"], g["path"]

    x = ideal_state(net, path)
    k, _a, _e, _aw = mass_class(net, x, L)
    assert k == "path", k
    assert state_counts(net, x) == (2 * L, float(L - 1), float(L), 0.0)

    # a chain ELSEWHERE in the graph is also a path-class state
    adj = collections.defaultdict(list)
    for (u, v) in g["e"]:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    seq = [v for v in range(g["n"]) if v not in path][:1]
    while seq and len(seq) < L + 1:
        nxt = [v for v in adj[seq[-1]] if v not in seq]
        if not nxt:
            break
        seq.append(nxt[0])
    chain_ok = None
    if seq and len(seq) == L + 1:
        chain_ok = mass_class(net, ideal_state(net, seq), L)[0]
        assert chain_ok == "path", chain_ok

    # L pairwise-disjoint consistent edges = a scatter-class state
    used, chosen = set(), []
    for (u, v) in g["e"]:
        u, v = int(u), int(v)
        if u not in used and v not in used:
            chosen.append((u, v))
            used |= {u, v}
        if len(chosen) == L:
            break
    xs = np.zeros(net.n_neurons, dtype=bool)
    for (u, v) in chosen:
        xs[net.neuron_at(u, v, 0)] = True
        xs[net.neuron_at(v, u, 1)] = True
    k3 = mass_class(net, xs, L)[0]
    assert k3 == "scatter", k3
    assert state_counts(net, xs) == (2 * L, 0.0, float(L), 0.0)

    # a gamma conflict, and a state off the matched shell, are both "other"
    xg = xs.copy()
    xg[net.neuron_at(chosen[0][0], chosen[0][1], 1)] = True
    assert mass_class(net, xg, L)[0] == "other"
    xw = xs.copy()
    xw[np.nonzero(xw)[0][0]] = False
    assert mass_class(net, xw, L)[0] == "other"

    return dict(graph_seed=g["gseed"], L=L, n_neurons=int(net.n_neurons),
                path=k, chain_elsewhere=chain_ok, scatter=k3, ok=True)


# ------------------------------------------------------------------ the smoke run
def smoke(n_graphs: int = 2, steps: int = 4000, N: int = 12, c: float = PILOT_C):
    """A few seconds of real sampling with the mass classifier engaged.

    This is the cheap default: it proves the ``simulate(mass_L=...)`` -> ``mass_class``
    hot path runs and produces the per-unit row the pilot ships -- it does NOT
    reproduce MR-NULL/MR-ACT (that is the committed 143M-step run).
    """
    pop, _ = graph_population(N, n_graphs)
    rows = []
    for g in pop:
        r = run_unit(dict(stage="smoke", N=N, g=g, c=c, steps=steps, sseed=1,
                          burnin=min(1000, steps // 2)))
        rows.append(r)
        print(f"    smoke gseed={r['graph_seed']:>2} L={r['L']} shell={r['matched_shell']:>2} "
              f"mean_act={r['mean_active']:>6.2f} band={r['band_steps']} "
              f"classifiable={r['consistent_steps']} ({r['steps_per_s']:.0f} steps/s)")
    return rows


# ----------------------------------------------------- the full pilot (--full only)
COLS = [
    "stage", "N", "c", "bias", "lam", "graph_seed", "L", "opt_cost", "mean_deg",
    "n_edges", "n_neurons", "n_segments", "sampler_seed", "steps", "burnin",
    "steps_scored", "wall_s", "steps_per_s", "band_lo", "band_hi", "band_halfwidth",
    "matched_shell", "band_steps", "transient_band_steps", "matched_steps",
    "consistent_steps", "mass_path", "mass_scatter", "mass_mixed", "mass_closed",
    "band_frac", "matched_frac", "classified_per_1e6", "path_per_1e6",
    "scatter_per_1e6", "n_eff_band", "n_eff_classified", "n_eff_path",
    "n_eff_scatter", "n_eff_active", "n_eff_frac", "mean_active", "mode_active",
    "min_active_reached", "max_active_reached", "act_over_shell", "st_mode", "tau",
    "irm_threshold", "record_every",
]


def run_full_pilot(N=FULL_N, n_graphs=FULL_GRAPHS, seeds=FULL_SEEDS, steps=FULL_STEPS):
    """Re-execute the pilot design SERIALLY into RERUN_CSV (seeded/expensive).

    Faithful re-run of the operating point; it does not overwrite the committed
    artefact and, being seeded and single-process, need not land the exact 72-unit
    prefix the original wall-cut run shipped.
    """
    pop, excluded = graph_population(N, n_graphs)
    print(f"  full pilot: N={N}, {len(pop)} graphs ({excluded} L=1 excluded), "
          f"{seeds} seeds x {steps:,} steps")
    RERUN_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(RERUN_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        wr.writeheader()
        for g in pop:
            for s in range(seeds):
                r = run_unit(dict(stage="P", N=N, g=g, c=PILOT_C, steps=steps,
                                  sseed=1000 + s))
                rows.append(r)
                wr.writerow(r)
                fh.flush()
    print(f"  full pilot -> {rel(RERUN_CSV)} ({len(rows)} units)")
    return rows


# ------------------------------------------------------- derive MR-NULL / MR-ACT
def _to_int(v):
    return int(v) if v not in ("", None) else 0


def derive_null(csv_path=COMMITTED_CSV) -> dict:
    """Aggregate the committed by-unit CSV into the MR-NULL / MR-ACT headline."""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    n_units = len(rows)
    scored = sum(_to_int(r["steps_scored"]) for r in rows)
    band = sum(_to_int(r["band_steps"]) for r in rows)
    matched = sum(_to_int(r["matched_steps"]) for r in rows)
    m_path = sum(_to_int(r["mass_path"]) for r in rows)
    m_scat = sum(_to_int(r["mass_scatter"]) for r in rows)
    m_mixed = sum(_to_int(r["mass_mixed"]) for r in rows)
    m_closed = sum(_to_int(r["mass_closed"]) for r in rows)
    mean_acts = [float(r["mean_active"]) for r in rows]
    Ns = sorted({_to_int(r["N"]) for r in rows})
    N = Ns[0] if len(Ns) == 1 else Ns

    return {
        "MR_NULL": {
            "n_units": n_units,
            "steps_scored_total": scored,
            # "shell occupied" in the manuscript = steps inside the 2L +/- BAND band
            "band_steps_total": band,
            "band_frac": band / scored if scored else float("nan"),
            # the stricter on-shell count (active == 2L exactly), reported alongside
            "matched_shell_steps_total": matched,
            "mass_path_total": m_path,
            "mass_scatter_total": m_scat,
            "mass_mixed_total": m_mixed,
            "mass_closed_total": m_closed,
            # MR-NULL: the mass ratio's two classes never populate
            "mass_identically_zero": (m_path == 0 and m_scat == 0),
            "network": "variant (unaugmented)",
        },
        "MR_ACT": {
            "mean_activity_min": min(mean_acts),
            "mean_activity_max": max(mean_acts),
            "two_N": 2 * N if isinstance(N, int) else None,
            "N": N,
        },
        "design": {
            "N": N, "c": PILOT_C, "lam": LAM, "D": D, "st_mode": ST_MODE, "tau": TAU,
            "irm": "off", "inhib": "off", "band_halfwidth": BAND,
            "source_csv": rel(csv_path),
        },
    }


# ------------------------------------------------------------------------ verify
def _check(label, got, expected, tol):
    ok = abs(got - expected) <= tol
    print(f"    {'OK ' if ok else 'XX '}{label:<34} got {got:<16.6g} "
          f"expect {expected:<12g} (tol {tol:g})")
    return ok


def run(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="mass_ratio_pilot")
    ap.add_argument("--full", action="store_true",
                    help="re-execute the full 143M-step pilot (Tier 2, expensive)")
    ap.add_argument("--smoke-steps", type=int, default=4000)
    ap.add_argument("--smoke-graphs", type=int, default=2)
    args = ap.parse_args([] if argv is None else argv)

    # 1. classifier self-check -- the direct spnn.sim.mass_class consumer
    chk = classifier_selfcheck()
    print(f"[13] mass_ratio_pilot: classifier OK "
          f"(gseed={chk['graph_seed']} L={chk['L']}: path/chain/scatter named)")

    # 2. cheap live smoke through simulate(mass_L=...) -> mass_class
    print(f"    live smoke ({args.smoke_graphs} graphs x {args.smoke_steps} steps):")
    smoke(n_graphs=args.smoke_graphs, steps=args.smoke_steps)

    # 3. optional full re-run (seeded/expensive, separate file)
    if args.full:
        run_full_pilot()

    # 4. derive + ship the MR-NULL / MR-ACT headline from the committed pilot
    result = derive_null()
    out = write_json("mass/null.json", result)
    nul, act = result["MR_NULL"], result["MR_ACT"]
    print(f"    derived {rel(out)} from {result['design']['source_csv']}")
    print(f"    MR-NULL {nul['n_units']} units, {nul['steps_scored_total']:,} steps; "
          f"band {nul['band_steps_total']:,} ({nul['band_frac']:.2e}); "
          f"mass zero = {nul['mass_identically_zero']}")
    print(f"    MR-ACT  mean activity {act['mean_activity_min']:.1f} to "
          f"{act['mean_activity_max']:.1f} against 2N={act['two_N']}")

    # verify against CLAIMS.tsv (MR-NULL exact-ish; MR-ACT tol 1e-1)
    print("    verify vs CLAIMS:")
    oks = [
        _check("MR-NULL n_units", nul["n_units"], 72, 0),
        _check("MR-NULL scored steps", nul["steps_scored_total"], 143_280_000, 0),
        _check("MR-NULL band steps", nul["band_steps_total"], 2116, 0),
        _check("MR-NULL band frac", nul["band_frac"], 1.5e-5, 5e-7),
        _check("MR-NULL path+scatter mass", nul["mass_path_total"] + nul["mass_scatter_total"], 0, 0),
        _check("MR-ACT activity min", act["mean_activity_min"], 25.0, 1e-1),
        _check("MR-ACT activity max", act["mean_activity_max"], 43.5, 1e-1),
        _check("MR-ACT 2N", act["two_N"], 24, 0),
    ]
    print(f"    {sum(oks)}/{len(oks)} checks pass"
          + ("" if all(oks) else "  *** MISMATCH ***"))
    result["_verified"] = bool(all(oks))
    return result


if __name__ == "__main__":
    run()
