"""Experiment 10 -- the U[1,2] weight-convention arm.

Claims: WT-PRED, WT-MEAS, WT-ERR (CLAIMS.tsv, section V-B).
Outputs: results/weights/prediction.json  (WT-PRED, WT-ERR -- Tier 1)
         results/weights/measured.csv      (WT-MEAS -- Tier 2, committed data)

A pre-registered test of the section V-B ratio law under a DIFFERENT weight
convention.  Required c is a ratio of two measured functions of the depth L:

    c_req(L, |E|) = [ a*L + b*ln|E| + c0 ] / [ (1 - 2*rho)*L + 1 ],
    rho = Etilde / L = the path's mean edge weight in units of B = 2*w_max,
    since  margin = (L+1) - 2*Etilde = (1 - 2*rho)*L + 1  identically.

Moving the sparse family's weights from U(0,1) to U[1,2] raises rho (the path
runs over heavier edges relative to a w_max that only doubled), which shrinks the
denominator and so RAISES required c.  The arm was run before the outcome was
known, and it lands where the law says:

  * WT-PRED  the sparse rho median rises 0.119 -> ~0.36, so (1 - 2*rho) falls
             0.76 -> ~0.29 -- confirmed by the committed U[1,2] instances.
  * WT-MEAS  119 completed U[1,2] instances; per-cell required-c is 1.95 / 1.95 /
             1.89 times the U(0,1) baseline (it is (1-2rho), not U[1,2] as such);
             0 of 119 clear at c = 4.
  * WT-ERR   the MAIN-POOL numerator fit (which never saw this weight convention),
             applied to each U[1,2] instance's own margin, predicts required c to a
             median relative error of 7.5 % -- the law transfers to an unseen
             convention.

Provenance note: an internal first pass analysed this arm off a still-streaming
CSV and read only 103 of the 119 rows (a self-found defect, later corrected).
``results/weights/measured.csv`` is the CORRECTED 119-instance file (120 attempted,
1 CENSORED), and the figures here are the corrected ones.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st

from experiments._base import RESULTS, rel, write_json

MEASURED_CSV = RESULTS / "weights" / "measured.csv"

# --- documented inputs from the other section V-B claims (not re-fit here) -------
# The main-pool numerator fit (CLAIMS LAW-FIT; s28 lnratio_model over 1,027 rows).
# It never saw the U[1,2] convention -- that is what makes WT-ERR a held-out test.
NUMERATOR_FIT = {"a": 2.288863661768877,
                 "b": 11.566492187192502,
                 "c0": -42.451262975160056,
                 "r2": 0.9659988428361291}
# The sparse family's U(0,1) baseline: median rho (CLAIMS LAW-RHO) and median
# required c per size (CLAIMS POP-SIZE), the denominators the U[1,2] arm is
# measured against.
BASELINE_RHO = 0.11876572041095769           # sparse median Etilde/L, U(0,1)
BASELINE_MED_C = {30: 5.01, 60: 6.15, 100: 7.77}   # sparse median c_req_micro, U(0,1)

# The pre-registered prediction the arm was a test of.
PRED_RHO_U12 = 0.36
PRED_ONE_MINUS_2RHO_U12 = 0.29

C4 = 4.0                                       # the "clears at c=4" threshold


def _median(xs):
    return float(st.median(xs))


def _read_ok(csv_path=MEASURED_CSV):
    """The completed U[1,2] instances (status == 'ok'); the 1 CENSORED row dropped."""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    ok = [r for r in rows if r["status"] == "ok"]
    return rows, ok


def _rho(r):
    return float(r["Etilde"]) / int(r["L"])


def _predict_creq(r):
    """The main-pool numerator fit over this instance's OWN measured margin."""
    a, b, c0 = NUMERATOR_FIT["a"], NUMERATOR_FIT["b"], NUMERATOR_FIT["c0"]
    return (a * int(r["L"]) + b * math.log(float(r["E"])) + c0) / float(r["margin_c1"])


def run(argv=None) -> dict:
    ap = argparse.ArgumentParser(prog="weight_convention")
    ap.parse_args([] if argv is None else argv)

    rows, ok = _read_ok()
    n_ok = len(ok)

    # ---- WT-MEAS: per-cell required c, rho, and the ratio to the U(0,1) baseline
    cells = []
    for size in (30, 60, 100):
        v = [r for r in ok if int(r["size"]) == size]
        med_c = _median([float(r["c_req_micro"]) for r in v])
        rho = _median([_rho(r) for r in v])
        base = BASELINE_MED_C[size]
        cells.append({"size": size, "n": len(v), "med_rho": rho,
                      "one_minus_2rho": 1 - 2 * rho, "med_c_req_micro": med_c,
                      "baseline_med_c": base, "ratio": med_c / base})
    clears_c4 = sum(1 for r in ok if float(r["c_req_micro"]) <= C4)

    # ---- WT-PRED: the sparse rho rise, confirmed on the committed U[1,2] instances
    u12_rho = _median([_rho(r) for r in ok])
    prediction = {
        "baseline_rho": BASELINE_RHO,
        "baseline_one_minus_2rho": 1 - 2 * BASELINE_RHO,
        "predicted_rho_u12": PRED_RHO_U12,
        "predicted_one_minus_2rho_u12": PRED_ONE_MINUS_2RHO_U12,
        "measured_rho_u12": u12_rho,
        "measured_one_minus_2rho_u12": 1 - 2 * u12_rho,
    }

    # ---- WT-ERR: the held-out numerator fit vs each instance's observed required c
    pred = [_predict_creq(r) for r in ok]
    obs = [float(r["c_req_micro"]) for r in ok]
    rel_err = [abs(o - p) / o for o, p in zip(obs, pred)]
    med_pred, med_obs = _median(pred), _median(obs)
    wt_err = {
        "n_instances": n_ok,
        "median_predicted": med_pred,
        "median_observed": med_obs,
        "ratio_pred_over_obs": med_pred / med_obs,
        "median_relative_error": _median(rel_err),
        "numerator_fit": NUMERATOR_FIT,
    }

    result = {
        "WT_PRED": prediction,
        "WT_MEAS": {"n_instances": n_ok, "n_attempted": len(rows),
                    "cells": cells, "clears_at_c4": clears_c4,
                    "measured_csv": rel(MEASURED_CSV)},
        "WT_ERR": wt_err,
    }
    out = write_json("weights/prediction.json", result)

    print(f"[10] weight_convention -> {rel(out)}  (+ {rel(MEASURED_CSV)})")
    print(f"    WT-PRED  rho {BASELINE_RHO:.3f} -> {u12_rho:.3f} (pred ~{PRED_RHO_U12}); "
          f"(1-2rho) {1 - 2 * BASELINE_RHO:.3f} -> {1 - 2 * u12_rho:.3f} "
          f"(pred ~{PRED_ONE_MINUS_2RHO_U12})")
    print(f"    WT-MEAS  {n_ok} instances; ratios "
          + " / ".join(f"{c['ratio']:.2f}" for c in cells)
          + f"; {clears_c4} of {n_ok} clear at c={C4:.0f}")
    print(f"    WT-ERR   median predicted {med_pred:.2f} vs observed {med_obs:.2f} "
          f"(ratio {med_pred / med_obs:.3f}); median rel err "
          f"{100 * _median(rel_err):.1f}%")

    # ---- verify vs CLAIMS.tsv
    print("    verify vs CLAIMS:")
    oks = [
        _check("WT-PRED baseline rho", BASELINE_RHO, 0.119, 1e-2),
        _check("WT-PRED baseline 1-2rho", 1 - 2 * BASELINE_RHO, 0.76, 1e-2),
        _check("WT-PRED u12 rho", u12_rho, 0.36, 1e-2),
        _check("WT-PRED u12 1-2rho", 1 - 2 * u12_rho, 0.29, 1e-2),
        _check("WT-MEAS n_instances", n_ok, 119, 0),
        _check("WT-MEAS ratio N=30", cells[0]["ratio"], 1.95, 1e-2),
        _check("WT-MEAS ratio N=60", cells[1]["ratio"], 1.95, 1e-2),
        _check("WT-MEAS ratio N=100", cells[2]["ratio"], 1.89, 1e-2),
        _check("WT-MEAS clears at c=4", clears_c4, 0, 0),
        _check("WT-ERR median rel err", _median(rel_err), 0.075, 1e-2),
    ]
    print(f"    {sum(oks)}/{len(oks)} checks pass"
          + ("" if all(oks) else "  *** MISMATCH ***"))
    result["_verified"] = bool(all(oks))
    return result


def _check(label, got, expected, tol):
    ok = abs(got - expected) <= tol
    print(f"    {'OK ' if ok else 'XX '}{label:<26} got {got:<16.6g} "
          f"expect {expected:<10g} (tol {tol:g})")
    return ok


if __name__ == "__main__":
    run()
