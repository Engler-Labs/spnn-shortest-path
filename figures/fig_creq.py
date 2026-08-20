"""Figure fig:creq -- the required-c population figure (both panels), as numbers.

Claims: F-CREQ-A, F-CREQ-B  (CLAIMS.tsv, section V-B fig).

Outputs (under results/creq/):
  * quantiles.tsv  -- panel A: the box-plot table, 5 groups x 7 order statistics
                      of required c (microstate convention) over completed
                      instances (F-CREQ-A).
  * slopes.tsv     -- panel B: the per-family OLS slope of required c against
                      depth L, with a 95% CI (F-CREQ-B) -- the OPPOSITE-SIGN
                      "informative failure".

WHAT THIS FIGURE MODULE DOES.  It emits the numbers the manuscript's TikZ
consumes for the two panels of fig:creq -- NOT a rendered image -- so the figure
and its data are provably the same object (the build spec sec 3.3).  It is a
pure re-analysis off the committed ``results/creq/population.csv`` (the same
table ``creq_sweep`` ships): no counting, no sampler, deterministic, seconds.

PANEL A (F-CREQ-A).  Box-plot of the required-c distribution for the four graph
families and the pooled population.  The seven order statistics per group are
min, q10, q25, median, q75, q90, max of ``c_req_micro`` (Boltzmann-microstate
convention -- the headline one) over the status=="ok" instances.  This is the
SAME statistic ``creq_sweep`` writes to ``by_family.tsv``; ``run`` recomputes it
from scratch and asserts it reproduces ``by_family.tsv``'s order-stat columns
exactly (printed PASS/FAIL).

PANEL B (F-CREQ-B).  A single per-family ordinary-least-squares regression of
each instance's required c on its optimal-path depth L,

    c_req_micro ~ 1 + L      (per family; 95% CI = Student-t at 0.975, dof=n-2),

reported as the slope dc/dL with its 95% interval.  The four family slopes have
OPPOSITE SIGNS -- sparse -0.61, dense -0.76, grid +0.81, terrain(mtn_knn) +0.02 --
which is exactly the point: required c does NOT track depth with a single sign
across families, so a naive "deeper => harder" reading fails.  The pooled slope
(+0.58) is emitted as a fifth row for reference; the panel itself is the four
families.

Origin / provenance.  The panel-A statistic and the panel-B slope definition are
extracted from an internal population-analysis harness.  Panel B is its
"c-VERSUS-L functional form" block: a per-family (and POOLED) ordinary-least-
squares fit ``ols_ci([r["L"] for r in v], [r["c_req_micro"] for r in v])``.  The
four slopes reproduce the committed fit values: sparse -0.6144834538633818,
dense -0.7584615738664676, grid +0.8069260920107081, mtn_knn +0.02195920157805372.
``ols_ci`` is preserved verbatim (numpy lstsq + scipy Student-t) so the slopes and
their intervals reproduce exactly; it reads the committed population CSV.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as _st

# figures/ is a script directory (not a package), so put the repo root on the
# path before importing the experiments._base result writers.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._base import RESULTS, rel, write_tsv  # noqa: E402

FAM_ORDER = ["sparse", "dense", "grid", "mtn_knn"]
GROUPS = FAM_ORDER + ["POOLED"]

# The seven order statistics of the box-plot (panel A), matching creq_sweep's
# by_family.tsv exactly.
QS = [0, 10, 25, 50, 75, 90, 100]
Q_LABEL = ["min", "q10", "q25", "median", "q75", "q90", "max"]

# The four family slopes the panel-B claim prints, for the MATCH/FAIL check.
EXPECTED_SLOPE = {"sparse": -0.61, "dense": -0.76, "grid": 0.81, "mtn_knn": 0.02}


# --------------------------------------------------------------------------- load
def load_completed():
    """Read the committed population.csv; keep status=="ok" rows with the two
    columns this figure needs (L, c_req_micro) plus the group key."""
    path = RESULTS / "creq" / "population.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- it ships committed; regenerate with "
            f"`python -m experiments 8 --regenerate` (expensive, ~90 core-minutes).")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            rows.append(dict(family=r["family"], size=int(r["size"]),
                             seed=int(r["seed"]), L=int(r["L"]),
                             c_req_micro=float(r["c_req_micro"])))
    return rows


def quant(vals):
    """The seven order statistics, numpy default linear interpolation -- the exact
    convention creq_sweep uses for by_family.tsv."""
    a = np.asarray(sorted(vals), dtype=float)
    return {q: float(np.percentile(a, q)) for q in QS}


# ---------------------------------------------------- OLS (verbatim from the harness)
def ols_ci(x, y):
    """Slope, intercept, 95% CI on the slope, R^2 -- plain OLS, n >= 3.

    Verbatim from the internal analysis harness so the panel-B
    slopes and their Student-t intervals reproduce exactly."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return None
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(cov[1, 1])
    tcrit = float(_st.t.ppf(0.975, dof))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return dict(n=n, slope=float(beta[1]), intercept=float(beta[0]),
                lo=float(beta[1] - tcrit * se), hi=float(beta[1] + tcrit * se),
                se=se, r2=r2)


# ------------------------------------------------------- panel-A self-verification
def _verify_against_by_family(computed):
    """Cross-check the panel-A order statistics against the committed
    by_family.tsv (creq_sweep's T-CREQDIST/POP-SPAN output).  Returns (ok, diffs)."""
    path = RESULTS / "creq" / "by_family.tsv"
    if not path.exists():
        return None, []
    ref = {}
    with open(path) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            ref[r["family"]] = r
    diffs = []
    for grp in GROUPS:
        if grp not in ref:
            diffs.append((grp, "n/a", "missing in by_family.tsv"))
            continue
        for lab in Q_LABEL:
            got = computed[grp][lab]
            want = float(ref[grp][lab])
            if abs(got - want) > 1e-9:
                diffs.append((grp, lab, f"{got} != {want}"))
    return (len(diffs) == 0), diffs


# ---------------------------------------------------------------------------- run
def run(argv=None) -> dict:
    rows = load_completed()
    by_fam = {fam: [r for r in rows if r["family"] == fam] for fam in FAM_ORDER}

    # ---- panel A: quantiles.tsv (F-CREQ-A) -- 5 groups x 7 order statistics -------
    a_header = ["family", "n"] + Q_LABEL
    a_rows, a_computed = [], {}
    for grp in GROUPS:
        v = rows if grp == "POOLED" else by_fam[grp]
        q = quant([r["c_req_micro"] for r in v])
        vals = {lab: round(q[qs], 4) for lab, qs in zip(Q_LABEL, QS)}
        a_computed[grp] = vals
        a_rows.append([grp, len(v)] + [vals[lab] for lab in Q_LABEL])
    a_path = write_tsv("creq/quantiles.tsv", a_header, a_rows)

    a_ok, a_diffs = _verify_against_by_family(a_computed)

    # ---- panel B: slopes.tsv (F-CREQ-B) -- per-family OLS of c_req_micro on L ------
    b_header = ["family", "n", "slope", "ci_lo", "ci_hi", "intercept", "r2"]
    b_rows, b_fits, b_match = [], {}, {}
    for grp in FAM_ORDER + ["POOLED"]:
        v = rows if grp == "POOLED" else by_fam[grp]
        f_ = ols_ci([r["L"] for r in v], [r["c_req_micro"] for r in v])
        if f_ is None:
            continue
        b_fits[grp] = f_
        b_rows.append([grp, f_["n"], round(f_["slope"], 6), round(f_["lo"], 6),
                       round(f_["hi"], 6), round(f_["intercept"], 6),
                       round(f_["r2"], 6)])
        if grp in EXPECTED_SLOPE:
            b_match[grp] = (round(f_["slope"], 2) == EXPECTED_SLOPE[grp])
    b_path = write_tsv("creq/slopes.tsv", b_header, b_rows)

    slopes_all_match = all(b_match.get(fam) for fam in FAM_ORDER)

    # ---- summary ----------------------------------------------------------------
    print(f"[fig:creq] fig_creq -> {rel(RESULTS / 'creq')}/  (off committed population.csv)")
    print(f"    F-CREQ-A  quantiles.tsv : {len(a_rows)} groups x {len(Q_LABEL)} "
          f"order statistics -> {rel(a_path)}")
    if a_ok is None:
        print("      (by_family.tsv absent -- skipped the cross-check)")
    elif a_ok:
        print("      PASS: order statistics reproduce by_family.tsv exactly")
    else:
        print(f"      FAIL: {len(a_diffs)} cell(s) differ from by_family.tsv: "
              + "; ".join(f"{g}.{lab}: {msg}" for g, lab, msg in a_diffs[:6]))
    print(f"    F-CREQ-B  slopes.tsv    : per-family dc/dL (c_req_micro ~ L) -> "
          f"{rel(b_path)}")
    for fam in FAM_ORDER:
        f_ = b_fits.get(fam)
        if f_ is None:
            continue
        tag = "MATCH" if b_match.get(fam) else "FAIL"
        print(f"      {fam:<8} slope {f_['slope']:+.4f} "
              f"[{f_['lo']:+.4f}, {f_['hi']:+.4f}]  "
              f"(expected {EXPECTED_SLOPE[fam]:+.2f})  {tag}")
    if "POOLED" in b_fits:
        pf = b_fits["POOLED"]
        print(f"      POOLED   slope {pf['slope']:+.4f} "
              f"[{pf['lo']:+.4f}, {pf['hi']:+.4f}]  (aggregate; reference only)")
    print("      => opposite signs across families: the informative failure "
          + ("(all 4 match CLAIMS)" if slopes_all_match else "(SLOPES DO NOT MATCH CLAIMS)"))

    return dict(
        F_CREQ_A=dict(groups=GROUPS, order_stats=Q_LABEL, quantiles=a_computed,
                      matches_by_family=a_ok, diffs=a_diffs),
        F_CREQ_B=dict(fits=b_fits, expected=EXPECTED_SLOPE,
                      per_family_match=b_match, all_match=slopes_all_match),
        outputs=dict(quantiles=rel(a_path), slopes=rel(b_path)))


if __name__ == "__main__":
    run(sys.argv[1:])
