"""Experiment 11 -- counting limits, the top-coefficient bit ladder, and the
legacy-width overflow audit.

Claims: CNT-BITS, CNT-49, CNT-BIND (CLAIMS.tsv, section V-C).
Outputs:
  results/counts/bit_ladder.tsv       (CNT-BITS)  -- default / --bit-ladder, tier 1
  results/counts/limits.json          (CNT-BIND)  -- default, PINNED (hardware)
  results/counts/overflow_audit.json  (CNT-49)    -- --legacy-width, tier 2

Origin (the build spec 3.2): EXTRACT -- "stage 26 pilot + stage 27 fix
verification".  The three deliverables are the three faces of the same fact: the
packed matching-polynomial counter (spnn.counting.count_matchings) carries the
polynomial in ONE Python integer with fixed-width coefficient fields, and the
historical fixed width DIG = 96 is exact only while every coefficient stays below
2^96.  Above it a coefficient carries into the next field and is masked -- silently,
and always DOWNWARD, which makes the required continuity scale c look SMALLER (easier)
than it truly is.  Stage 27 sized the field from the provable bound
m(G,k) <= C(|E|,k), turned a would-be silent mask into a loud CoefficientOverflow,
and made every call report ``max_coeff_bits`` in ``stats`` so the overflow condition
is visible in the DATA (the paper's Code-availability promise), not only in an
assertion.

  * CNT-BITS  (tier 1, EXACT) -- the top-coefficient bit ladder over the grid family
    grid_graph(k, seed=7, jitter=0.25) for the lattice size k = 4..12.  For each grid
    the matching polynomial is counted at the matched scatter size K = L+2 (L = the
    corner-to-corner optimum hop count) and the ladder value is the LARGEST
    coefficient's bit length (``max_coeff_bits``).  It is 12 20 30 41 53 65 78 90 103.
    The realised largest coefficient stays under the legacy 2^96 line through k=11
    (90 bits) and breaks it at k=12 (103 bits): the 96-bit field is exact through
    k=11 and would have been silently wrong at k=12.  Independently confirmed by the
    stage-26 packed-vs-unpacked exactness audit (grid 10/11/12 -> 78/90/103 bits).

  * CNT-49 (tier 2) -- the legacy-width audit over the 1,080-instance required-c
    population (the stage-28 sweep, experiment 8's ``creq_sweep``): 49 completed
    instances would have carried a silently-wrong #scatter count under the old 96-bit
    field.  The population is expensive to regenerate, so this reads the committed
    per-instance legacy verdict (``results/counts/legacy_width_population.tsv``,
    transcribed from the stage-28 population sweep) and RE-DERIVES the count:
    ``legacy_dig96_exact == False AND status == ok`` -> 49, all 49 in the two cells
    stage 27 unblocked (grid k=12: 40, mtn_knn N=140: 9).

  * CNT-BIND (tier 2, PINNED) -- the scatter DP's memory ceiling.  Under a 6 GB cap
    the exact frontier DP dies past sparse N~140 and dense N~100 (the frontier grows
    Theta(N) on G(n,p)).  This boundary is HARDWARE-dependent; it is pinned, not run.

Reference instance: ``random_graph(60, 0.05, seed=19)`` (used only as a headroom
control here).
"""

from __future__ import annotations

import csv
import sys

from spnn.counting import (LEGACY_DIG, CoefficientOverflow, count_matchings)
from spnn.graphs import dijkstra, grid_graph

from experiments._base import rel, results_path, write_json, write_tsv

# CNT-BITS: the grid family and its expected top-coefficient (max_coeff_bits) ladder.
GRID_SEED, GRID_JITTER = 7, 0.25
GRID_SIZES = list(range(4, 13))                       # lattice size k = 4..12
EXPECTED_LADDER = [12, 20, 30, 41, 53, 65, 78, 90, 103]

# CNT-49: the 1,080-instance population with a per-row legacy-width verdict.  Prefer the
# canonical population file (experiment 8 / creq_sweep, which persists max_coeff_bits per
# row -- the Code-availability promise, claim CNT-HEAD); fall back to a self-contained
# copy so the audit stands alone if that experiment's output is absent.
POPULATION_SOURCES = [
    ("creq/population.csv", ","),                      # canonical, if creq_sweep is built
    ("counts/legacy_width_population.tsv", "\t"),      # self-contained fallback
]


# ---------------------------------------------------------------------------
def _grid_ladder():
    """The CNT-BITS ladder: (grid_k, N, E, L, matched_k, max_coeff_bits,
    top_coeff_bits, legacy_dig96_exact, dig_autosized, coeff_bound_bits) per grid."""
    rows = []
    for k in GRID_SIZES:
        n, edges, w = grid_graph(k, seed=GRID_SEED, jitter=GRID_JITTER)
        path, _ = dijkstra(n, edges, w, 0, n - 1)
        L = len(path) - 1
        matched_k = L + 2                              # matched scatter size
        _m, st = count_matchings(n, edges, matched_k, dig=None)
        rows.append(dict(
            grid_k=k, N=n, E=int(edges.shape[0]), L=L, matched_k=matched_k,
            max_coeff_bits=st["max_coeff_bits"],
            top_coeff_bits=st["top_coeff_bits"],
            argmax_coeff_k=st["argmax_coeff_k"],
            legacy_dig96_exact=st["legacy_dig96_exact"],
            dig_autosized=st["dig"],
            coeff_bound_bits=st["coeff_bound_bits"]))
    return rows


def bit_ladder() -> dict:
    """CNT-BITS -- the top-coefficient bit ladder, EXACT, tier 1."""
    rows = _grid_ladder()
    ladder = [r["max_coeff_bits"] for r in rows]
    matches = ladder == EXPECTED_LADDER

    # The overflow verdict: the realised largest coefficient stays under 2^96 through
    # k=11 and breaks it at k=12.  Demonstrate the guard on the k=12 grid directly:
    # forcing the historical width dig=96 now RAISES rather than silently masking.
    last = rows[-1]                                    # grid k=12
    n, edges, w = grid_graph(last["grid_k"], seed=GRID_SEED, jitter=GRID_JITTER)
    path, _ = dijkstra(n, edges, w, 0, n - 1)
    matched_k = len(path) - 1 + 2
    legacy_raises = False
    try:
        count_matchings(n, edges, matched_k, dig=LEGACY_DIG)
    except CoefficientOverflow:
        legacy_raises = True

    exact_through = [r["grid_k"] for r in rows if r["legacy_dig96_exact"]]
    fails_at = [r["grid_k"] for r in rows if not r["legacy_dig96_exact"]]

    header = ["grid_k", "N", "E", "L", "matched_k", "max_coeff_bits",
              "top_coeff_bits", "argmax_coeff_k", "legacy_dig96_exact",
              "dig_autosized", "coeff_bound_bits", "expected_max_coeff_bits"]
    tsv_rows = [[r["grid_k"], r["N"], r["E"], r["L"], r["matched_k"],
                 r["max_coeff_bits"], r["top_coeff_bits"], r["argmax_coeff_k"],
                 r["legacy_dig96_exact"], r["dig_autosized"], r["coeff_bound_bits"],
                 EXPECTED_LADDER[i]]
                for i, r in enumerate(rows)]
    out = write_tsv("counts/bit_ladder.tsv", header, tsv_rows)

    result = dict(
        claim="CNT-BITS", tier=1, state="built",
        grid="grid_graph(k, seed=%d, jitter=%s), matched scatter size k=L+2"
             % (GRID_SEED, GRID_JITTER),
        grid_sizes=GRID_SIZES, ladder=ladder, expected=EXPECTED_LADDER,
        ladder_matches=matches, legacy_dig=LEGACY_DIG,
        legacy_exact_through_k=max(exact_through) if exact_through else None,
        legacy_fails_at_k=min(fails_at) if fails_at else None,
        legacy_dig96_forced_raises_at_k12=legacy_raises,
        rows=rows, output=rel(out))
    print(f"[11] counting_limits --bit-ladder -> {rel(out)}")
    print(f"    ladder (max_coeff_bits, grid k={GRID_SIZES[0]}..{GRID_SIZES[-1]}): "
          f"{ladder}")
    print(f"    expected                                       : {EXPECTED_LADDER}")
    print(f"    MATCH={matches}; legacy 96-bit exact through k="
          f"{result['legacy_exact_through_k']}, fails at k="
          f"{result['legacy_fails_at_k']} "
          f"(forced dig=96 raises at k=12: {legacy_raises})")
    return result


# ---------------------------------------------------------------------------
def limits() -> dict:
    """CNT-BIND -- the scatter-DP memory ceiling.  PINNED (hardware-dependent).

    Pinned from the stage-26 ceiling pilot (6 GB RLIMIT_AS per count): the exact
    scatter frontier DP completes for sparse through N=100 and MEMCAPs at N=140, and
    for dense through N=80 and MEMCAPs at N=100 -- i.e. it dies past sparse N~140 and
    dense N~100.  These boundaries move with the memory cap and the machine, so they
    are recorded, not measured here.
    """
    result = dict(
        claim="CNT-BIND", tier=2, state="pinned",
        note=("hardware-dependent: the scatter frontier DP's peak RSS grows with the "
              "elimination frontier (Theta(N) on G(n,p)); the boundary is where that "
              "exceeds the memory cap, so it is pinned to the cap and the machine, not "
              "reproduced.  Source: stage-26 scatter-count ceiling pilot "
              "(s26_curve.csv), 6 GB RLIMIT_AS per count."),
        memory_cap_gb=6.0,
        boundaries={
            "sparse": {"last_completed_N": 100, "memcap_at_N": 140,
                       "dies_past_N": 140},
            "dense":  {"last_completed_N": 80,  "memcap_at_N": 100,
                       "dies_past_N": 100},
            "grid":   {"completed_through_k": 12, "note": "treewidth ~ sqrt(N); the "
                       "affordable family -- no memcap over the swept ladder"},
            "mtn_knn": {"note": "scatter completes over the swept ladder; #path (DFS) "
                        "hits the wall first at N=180"},
        },
        claim_text="sparse dies past N~140, dense past N~100, at a 6 GB cap")
    out = write_json("counts/limits.json", result)
    print(f"[11] counting_limits (CNT-BIND, pinned) -> {rel(out)}")
    print(f"    PINNED @ 6 GB cap: sparse dies past N~140, dense past N~100")
    return result


# ---------------------------------------------------------------------------
def legacy_width() -> dict:
    """CNT-49 -- the legacy-96-bit overflow audit over the 1,080-instance population.

    Re-derives the headline count from the committed per-instance legacy verdict
    (``results/counts/legacy_width_population.tsv``, transcribed from the stage-28
    required-c population sweep, chunks A/B/C).  An instance would have been SILENTLY
    WRONG under the old fixed 96-bit field iff it (a) COMPLETED (status ok) and (b)
    its largest matching coefficient reached >= 2^96 (legacy_dig96_exact == False).
    """
    src = next((rel_ for rel_, _ in POPULATION_SOURCES
                if results_path(rel_).exists()), None)
    if src is None:
        missing = ", ".join(r for r, _ in POPULATION_SOURCES)
        result = dict(claim="CNT-49", tier=2, state="not-reproduced",
                      reason=f"no population backing file found (looked for {missing})")
        out = write_json("counts/overflow_audit.json", result)
        print(f"[11] counting_limits --legacy-width: NOT REPRODUCED "
              f"(no population file) -> {rel(out)}")
        return result
    delim = dict(POPULATION_SOURCES)[src]
    path = results_path(src)

    def _is_exact(v):     # legacy_dig96_exact is serialized as 1/0/'' or True/False/''
        return {"1": True, "true": True, "0": False, "false": False}.get(
            v.strip().lower(), None)

    total = completed = censored = legacy_exact = silently_wrong = 0
    overflow_but_censored = 0
    by_cell: dict[str, int] = {}
    with path.open() as f:
        for r in csv.DictReader(f, delimiter=delim):
            total += 1
            ok = r["status"].strip() == "ok"
            le = _is_exact(r["legacy_dig96_exact"])    # True exact, False overflow, None censored
            if ok:
                completed += 1
            else:
                censored += 1
            if le is True:
                legacy_exact += 1
            elif le is False and ok:
                silently_wrong += 1
                cell = f"{r['family']}_{r['size']}"
                by_cell[cell] = by_cell.get(cell, 0) + 1
            elif le is False and not ok:
                overflow_but_censored += 1

    result = dict(
        claim="CNT-49", tier=2, state="built",
        definition=("silently-wrong = legacy_dig96_exact False (largest matching "
                    "coefficient >= 2^96, so the old fixed 96-bit field would carry "
                    "and mask it DOWNWARD) AND status ok (the instance completed, so "
                    "the masked -- too-small -- #scatter WOULD have been returned)."),
        source=src,
        provenance=("stage-28 required-c population sweep, chunks A/B/C; per-row "
                    "max_coeff_bits + legacy verdict persisted per instance (CNT-HEAD). "
                    "Read from the canonical results/creq/population.csv when present, "
                    "else the self-contained results/counts/legacy_width_population.tsv"),
        population_total=total, completed=completed, censored=censored,
        legacy_exact_completed=legacy_exact,
        would_overflow_total=silently_wrong + overflow_but_censored,
        would_overflow_but_censored=overflow_but_censored,
        silently_wrong=silently_wrong,
        silently_wrong_by_cell=by_cell,
        expected_silently_wrong=49, matches=(silently_wrong == 49),
        claim_text="49 of 1,080 would have been silently wrong")
    out = write_json("counts/overflow_audit.json", result)
    print(f"[11] counting_limits --legacy-width -> {rel(out)}")
    print(f"    silently wrong: {silently_wrong} of {total}  "
          f"(expected 49; MATCH={result['matches']})")
    print(f"    by cell: {by_cell}  "
          f"[completed {completed}, censored {censored}, "
          f"legacy-exact {legacy_exact}]")
    return result


# ---------------------------------------------------------------------------
def run(argv=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    do_ladder = "--bit-ladder" in argv
    do_legacy = "--legacy-width" in argv
    # Default (no flag): the cheap tier-1 work + the pinned boundary.  The 1,080-
    # instance legacy audit is tier-2 and only runs behind --legacy-width.
    out: dict = {}
    if do_ladder or not (do_ladder or do_legacy):
        out["CNT_BITS"] = bit_ladder()
        out["CNT_BIND"] = limits()
    if do_legacy:
        out["CNT_49"] = legacy_width()
    return out


if __name__ == "__main__":
    run()
