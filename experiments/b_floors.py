"""Experiment 15 -- B floors per instance (section VII-B).

Claims: B-FLOOR-FAM, B-FLOOR-UNREACH (CLAIMS.tsv, section VII-B).
Output: results/creq/b_floors.json

The required-c FLOOR as B -> infinity, per instance.  Required c is a ratio,
c_req = num / margin, with num = ln(#scatter/#path) (microstate convention) and
margin(B) = (L+1) - 2*Etilde(B).  Only the denominator carries B: Etilde(B) = W*/B,
so margin(B) -> (L+1) as B -> infinity and

    c_req -> num / (L+1)  =:  floor_i .

Because margin is bounded above by (L+1), NO admissible B can push an instance's
required c below its floor.  So an instance is cleared at c = 4 by SOME B iff
floor_i < 4; where floor_i >= 4 it is unreachable at c = 4 for every B.  The B that
reaches c_req = 4 (as a multiple of w_max, using mu/w_max = 2*rho and Etilde = rho*L
under B = 2*w_max) is

    B / w_max = 4 * Etilde / ((L+1) - num/4) ,   valid only where floor_i < 4 .

This upgrades section VII-B's B-floor statement from four family medians to a
population fact ("X% of instances have a floor above 4").  Pure arithmetic on the
committed results/creq/population.csv counts -- no enumeration -- so it re-derives
from scratch in CI.
"""

from __future__ import annotations

import csv
import statistics as st

from experiments._base import RESULTS, rel, write_json

FAM_ORDER = ["sparse", "dense", "grid", "mtn_knn"]
C4 = 4.0


def _load():
    rows = []
    with open(RESULTS / "creq" / "population.csv") as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            L = int(r["L"])
            num = float(r["ln_ratio_micro"])
            Et = float(r["Etilde"])
            floor = num / (L + 1)
            reachable = floor < C4
            # B (as a multiple of w_max) that would bring c_req down to 4
            b_over_wmax = (4.0 * Et / ((L + 1) - num / C4)) if reachable else None
            rows.append(dict(family=r["family"], L=L, floor=floor,
                             reachable=reachable, b_over_wmax=b_over_wmax))
    return rows


def _dist(xs):
    xs = sorted(xs)
    n = len(xs)
    q = lambda p: xs[min(n - 1, int(p * (n - 1) + 0.5))]
    return dict(n=n, min=xs[0], q25=q(0.25), median=st.median(xs), q75=q(0.75),
                max=xs[-1], mean=st.mean(xs))


def run(argv=None) -> dict:
    rows = _load()
    n = len(rows)
    floors = [r["floor"] for r in rows]
    n_unreach = sum(1 for r in rows if not r["reachable"])

    by_family = {}
    for fam in FAM_ORDER:
        v = [r for r in rows if r["family"] == fam]
        fv = [r["floor"] for r in v]
        by_family[fam] = dict(
            n=len(v), median_floor=st.median(fv),
            frac_floor_ge_4=sum(1 for r in v if not r["reachable"]) / len(v),
            median_B_over_wmax_to_c4=(
                st.median([r["b_over_wmax"] for r in v if r["reachable"]])
                if any(r["reachable"] for r in v) else None))

    reach_B = [r["b_over_wmax"] for r in rows if r["reachable"]]
    result = {
        "definition": "floor_i = ln(#scatter/#path)/(L+1) = c_req as B->inf; "
                      "unreachable at c=4 iff floor>=4; B/w_max to reach c=4 = "
                      "4*Etilde/((L+1)-num/4).",
        "n_instances": n,
        "floor_overall": _dist(floors),
        "n_floor_ge_4": n_unreach,
        "frac_floor_ge_4": n_unreach / n,
        "by_family": by_family,
        "B_over_wmax_to_c4_reachable": (_dist(reach_B) if reach_B else None),
        "n_reachable": len(reach_B),
    }
    out = write_json("creq/b_floors.json", result)

    print(f"[15] b_floors -> {rel(out)}  (arithmetic off committed population.csv)")
    fo = result["floor_overall"]
    print(f"    floor = num/(L+1) over {n}: median {fo['median']:.3f}, "
          f"range [{fo['min']:.3f}, {fo['max']:.3f}]")
    print(f"    UNREACHABLE at c=4 (floor>=4): {n_unreach}/{n} = "
          f"{100*n_unreach/n:.1f}% -- no admissible B clears these")
    print("    per family (median floor / % unreachable / median B/w_max to c=4):")
    for fam in FAM_ORDER:
        b = by_family[fam]
        bw = f"{b['median_B_over_wmax_to_c4']:.2f}" if b["median_B_over_wmax_to_c4"] else "n/a"
        print(f"      {fam:<8} {b['median_floor']:.3f}  "
              f"{100*b['frac_floor_ge_4']:.0f}%  {bw}")
    if reach_B:
        rb = result["B_over_wmax_to_c4_reachable"]
        print(f"    B/w_max to reach c=4 (reachable {len(reach_B)}): median {rb['median']:.2f}, "
              f"up to {rb['max']:.2f}")
    return result


if __name__ == "__main__":
    run()
