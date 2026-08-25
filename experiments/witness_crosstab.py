"""Experiment 22 -- the clears-at-c=4 x below-path-sector cross-tab (section V-B).

Claims: XT-C4-DZERO.
Output: results/crosstab/witness_clears.json

WHY THIS EXISTS.  Section V-B reports how many campaign instances CLEAR the discrimination
condition at the operating point c = 4 -- the (29) scatter-term crossing computed under the
microstate convention (``c_req_micro <= 4``).  Section V-D/V-E then reports, separately, that a
d = 0 member sitting AT OR BELOW the optimal path is the majority case across the population.
Those two statements are about the same 1,027 instances, and the paper's qualified premise
needs their INTERSECTION: an instance that clears the published condition is not thereby
established to have the path at the bottom of its matched-activity landscape.

This experiment computes that intersection.  It runs no simulation, no counting and no search:
it is pure arithmetic over two files this repository already ships, so it is tier 1 and runs in
CI.  Neither input is written.

  results/creq/population.csv   (experiment 8)   -- c_req_micro per instance
  results/dzero/verdicts.csv    (experiment 20)  -- WITNESS / CERTIFIED / UNDETERMINED per
                                                    instance for the d = 0 below-path sector

RESULT.  Of the 1,027 instances the campaign completed, 138 clear at c = 4.  Of those 138:

  126  carry a WITNESS -- a concrete d = 0 member at or below the path (a proof, regenerable);
    7  are CERTIFIED free of the below-path sector -- both legs closed, so these are the only
       instances that clear AND are established to have nothing below the path;
    5  are UNDETERMINED -- a leg is open; a search miss is NOT absence and is never imputed.

So "clears at c = 4" and "is free of the below-path sector" are close to disjoint in practice:
91 % of the clearing instances carry a below-path witness, and only 7 of 1,027 pass both.  The
UNDETERMINED five are reported as their own cell and are NOT folded into either verdict -- if
they were all to resolve as certified the count would be at most 12, and that bound is stated
rather than the midpoint being guessed.

VERDICT CONVENTION.  ``TIE`` (a co-optimal member exactly AT the path, delta = 0) counts with
WITNESS -- the sector is "at or below" the path, which is what the premise turns on.  The
committed verdicts carry no TIE rows, so this is a convention statement, not a reclassification.
"""

from __future__ import annotations

import csv
from collections import Counter

from experiments._base import rel, results_path, write_json

C4 = 4.0
FAMILIES = ["sparse", "dense", "grid", "mtn_knn"]
BELOW = {"WITNESS", "TIE"}          # a member AT or BELOW the path
FREE = {"CERTIFIED"}                # both legs closed: nothing at or below the path
OPEN = {"UNDETERMINED"}             # a leg open; never imputed either way


def _key(r):
    return (r["family"], int(r["size"]), int(r["seed"]))


def run(argv=None) -> dict:
    pop = list(csv.DictReader(open(results_path("creq/population.csv"), encoding="utf-8")))
    dz = {_key(r): r for r in
          csv.DictReader(open(results_path("dzero/verdicts.csv"), encoding="utf-8"))}

    ok = [r for r in pop if r["status"] == "ok"]
    clears = [r for r in ok if r["c_req_micro"] not in ("", "nan")
              and float(r["c_req_micro"]) <= C4]

    missing = [_key(r) for r in clears if _key(r) not in dz]
    tally = Counter()
    by_family = {f: Counter() for f in FAMILIES}
    for r in clears:
        v = dz[_key(r)]["verdict"]
        cell = "below_path" if v in BELOW else "certified_free" if v in FREE else \
               "undetermined" if v in OPEN else f"UNEXPECTED:{v}"
        tally[cell] += 1
        by_family[r["family"]][cell] += 1

    n_clear = len(clears)
    n_below = tally["below_path"]
    n_free = tally["certified_free"]
    n_open = tally["undetermined"]

    out = dict(
        convention=dict(
            clears_at_c4="c_req_micro <= 4 in results/creq/population.csv (the microstate "
                         "convention, the one section V-B's operating point uses)",
            below_path="verdict in {WITNESS, TIE} in results/dzero/verdicts.csv -- a concrete "
                       "d = 0 member AT or BELOW the optimal path",
            certified_free="verdict == CERTIFIED -- both legs closed, nothing at or below",
            undetermined="verdict == UNDETERMINED -- a leg open; never imputed either way",
            c4=C4),
        population=dict(rows=len(pop), completed=len(ok),
                        censored=len(pop) - len(ok)),
        n_clear_at_c4=n_clear,
        clear_and_below_path=n_below,
        clear_and_certified_free=n_free,
        clear_and_undetermined=n_open,
        fraction_of_clearing_with_witness=n_below / n_clear,
        upper_bound_if_all_undetermined_resolve_free=n_free + n_open,
        by_family={f: dict(by_family[f]) for f in FAMILIES},
        dzero_rows=len(dz),
        missing_from_dzero=missing,
        note="pure arithmetic over two committed deposits; neither is written. The "
             "UNDETERMINED cell is reported in its own right -- at most "
             f"{n_free + n_open} of 1,027 could pass both, and exactly {n_free} are "
             "established to.",
    )
    path = write_json("crosstab/witness_clears.json", out)

    print(f"[22] witness_crosstab -> {rel(path)}")
    print(f"    campaign: {out['population']['completed']} completed "
          f"({out['population']['censored']} censored); d=0 verdicts: {len(dz)}")
    print(f"    CLEAR at c = {C4:g}: {n_clear}")
    print(f"      of those, carry a d=0 member at or below the path : {n_below} "
          f"({100 * n_below / n_clear:.0f} %)")
    print(f"      of those, CERTIFIED free of the below-path sector  : {n_free}")
    print(f"      of those, UNDETERMINED (never imputed)             : {n_open}")
    for f in FAMILIES:
        c = by_family[f]
        print(f"      {f:<8} clear {sum(c.values()):<4} below {c['below_path']:<4} "
              f"certified {c['certified_free']:<3} undetermined {c['undetermined']}")

    assert not missing, f"{len(missing)} clearing instances absent from the d=0 verdicts"
    assert n_below + n_free + n_open == n_clear, (n_below, n_free, n_open, n_clear)
    assert (n_clear, n_below, n_free, n_open) == (138, 126, 7, 5), \
        (n_clear, n_below, n_free, n_open)
    return out


if __name__ == "__main__":
    import sys
    run(sys.argv[1:])
