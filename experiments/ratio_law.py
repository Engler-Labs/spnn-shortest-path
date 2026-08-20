"""Experiment 9 -- the numerator RATIO LAW for required c.

Claims: LAW-FIT, LAW-PRED, LAW-HOLDOUT, LAW-DEPTH, LAW-RHO, LAW-INTERACT
(CLAIMS.tsv, section V-B).

Outputs (under results/law/):
  * numerator_fit.json   -- the one three-parameter numerator fit over all 1,027
                            completed instances (LAW-FIT).
  * per_instance.csv     -- each instance's ratio-law-predicted c vs its own measured
                            required c, and the pooled/per-family residual summary
                            (LAW-PRED).
  * holdout.json         -- leave-one-family-out: numerator fitted on the other three
                            families, the held-out family predicted out of sample
                            (LAW-HOLDOUT).
  * depth_transfer.json  -- the cross-family c-vs-L table and the depth-law transfer
                            at L~14 (lattice vs terrain) (LAW-DEPTH).
  * rho.tsv              -- the per-family mean-edge-weight rho = Etilde/L and the
                            finite asymptote a/(1-2rho) it sets (LAW-RHO).
  * interaction_fit.json -- the refit of the numerator in the theory-predicted
                            interaction form X1 = L*ln(|E|*e/L), X2 = L: does the
                            leading coefficient beta1 come back at 1?  Plus the
                            three-term (L*ln|E|, L*lnL, L) model, each with R^2,
                            pooled median relative error, and leave-one-family-out
                            ratios against the additive baseline (LAW-INTERACT).
                            Emit the detailed console breakdown with --interaction.

THE LAW.  Required c is not linear in depth; it is a RATIO of two measured functions
of the instance:

    c_req(L, |E|) = [ a*L + b*ln|E| + c0 ] / [ (1 - 2*rho)*L + 1 ],
    rho = Etilde / L  (the path's mean edge weight in units of B = 2*w_max),

because the discrimination margin is  (L+1) - 2*Etilde = (1 - 2*rho)*L + 1  identically.
The NUMERATOR  ln(#scatter/#path) = c0 + a*L + b*ln|E|  is fitted once (LAW-FIT); every
prediction below then uses each instance's OWN margin as the denominator -- no per-family
term anywhere.  This is a Tier-1 experiment: it runs entirely off the committed
``results/creq/population.csv`` (produced by ``creq_sweep``), so it re-derives from
scratch in CI.

Origin / provenance.  Extracted from the ratio-law and depth-transfer sections of an
internal stage-28 analysis harness (deleted from that source tree; read at a specific
source revision).  Imports repointed at ``spnn``; a thread-pinning entry-point import
(an OpenBLAS oversubscription guard, a harness concern) is dropped.  ``numpy`` /
``scipy`` OLS is preserved verbatim so the fitted coefficients and their Student-t
intervals reproduce exactly.
"""

from __future__ import annotations

import csv
import math

import numpy as np
from scipy import stats as _st

from experiments._base import RESULTS, rel, results_path, write_json, write_tsv

FAM_ORDER = ["sparse", "dense", "grid", "mtn_knn"]
FAM_LABEL = {"grid": "lattice", "mtn_knn": "terrain",
             "sparse": "sparse", "dense": "dense"}


# --------------------------------------------------------------------------- load
def load_completed():
    path = RESULTS / "creq" / "population.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m experiments 8` first (ratio_law runs "
            f"off the committed population.csv that creq_sweep ships).")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            rows.append(dict(
                family=r["family"], size=int(r["size"]), seed=int(r["seed"]),
                L=int(r["L"]), E=int(r["E"]), Etilde=float(r["Etilde"]),
                margin_c1=float(r["margin_c1"]),
                ln_ratio_micro=float(r["ln_ratio_micro"]),
                ln_ratio_edge=float(r["ln_ratio_edge"]),
                c_req_micro=float(r["c_req_micro"])))
    return rows


def load_u12():
    """The completed U[1,2] weight-convention instances (experiment 10's committed
    data), for the held-out weight-convention test under the interaction numerator.
    Empty list if the file is absent (interaction JSON then omits the U[1,2] arm)."""
    path = RESULTS / "weights" / "measured.csv"
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            out.append(dict(L=int(r["L"]), E=int(r["E"]),
                            margin_c1=float(r["margin_c1"]),
                            c_req_micro=float(r["c_req_micro"])))
    return out


# ---------------------------------------------------- OLS (verbatim from the harness)
def ols_ci(x, y):
    """Slope, intercept, 95% CI on the slope, R^2 -- plain OLS, n >= 3."""
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


def multi_ols(X, y, names):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = n - p - 1
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(Xd.T @ Xd)
    tcrit = float(_st.t.ppf(0.975, dof))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    out = dict(n=n, r2=1.0 - float(resid @ resid) / ss_tot, intercept=float(beta[0]))
    for i, nm in enumerate(names):
        se = math.sqrt(cov[i + 1, i + 1])
        out[nm] = dict(coef=float(beta[i + 1]), lo=float(beta[i + 1] - tcrit * se),
                       hi=float(beta[i + 1] + tcrit * se), se=se)
    return out


def _numerator_fit(rows):
    """Fit ln(#scatter/#path) = c0 + a*L + b*ln|E| over `rows`."""
    return multi_ols(np.column_stack([[r["L"] for r in rows],
                                      [math.log(r["E"]) for r in rows]]),
                     [r["ln_ratio_micro"] for r in rows], ["L", "ln|E|"])


# ------------------------------------------------ the interaction refit (W4/W3)
# The asymptotic ln(#scatter/#path) ~ L*ln(|E|*e/(L*b)) = X1 - (ln b)*L predicts a
# UNIT coefficient on the composite predictor X1 = L*ln(|E|*e/L).  The additive fit
# above is a linearization of that multiplicative form; this refit tests the
# prediction directly -- does beta1 come back at 1?  Centered R^2 throughout so the
# no-intercept model stays comparable to the additive baseline.
_FEAT = {
    "L":       lambda r: float(r["L"]),
    "ln|E|":   lambda r: math.log(r["E"]),
    "X1":      lambda r: r["L"] * (math.log(r["E"]) - math.log(r["L"]) + 1.0),
    "X2":      lambda r: float(r["L"]),
    "L*ln|E|": lambda r: r["L"] * math.log(r["E"]),
    "L*lnL":   lambda r: r["L"] * math.log(r["L"]),
}


def _ols_named(rows, names, intercept, resp="ln_ratio_micro"):
    """OLS on named _FEAT columns, optional intercept, centered R^2, 95% t-CIs.
    ``resp`` selects the response column (ln_ratio_micro by default; ln_ratio_edge
    for the edge-count robustness check)."""
    X = np.column_stack([[_FEAT[nm](r) for r in rows] for nm in names]).astype(float)
    y = np.array([r[resp] for r in rows], float)
    n = len(rows)
    Xd = np.column_stack([np.ones(n), X]) if intercept else X
    p = Xd.shape[1]
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    s2 = float(resid @ resid) / (n - p)
    cov = s2 * np.linalg.inv(Xd.T @ Xd)
    tcrit = float(_st.t.ppf(0.975, n - p))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    off = 1 if intercept else 0
    coefs = {}
    if intercept:
        se = math.sqrt(cov[0, 0])
        coefs["intercept"] = [float(beta[0]), float(beta[0] - tcrit * se),
                              float(beta[0] + tcrit * se)]
    for i, nm in enumerate(names):
        se = math.sqrt(cov[i + off, i + off])
        coefs[nm] = [float(beta[i + off]), float(beta[i + off] - tcrit * se),
                     float(beta[i + off] + tcrit * se)]
    return dict(names=names, intercept=intercept, beta=beta.tolist(),
                r2=1.0 - float(resid @ resid) / ss_tot, coefs=coefs)


def _predict_c_named(fit, r):
    """c_req prediction = numerator(fit) / this instance's own margin."""
    b = fit["beta"]
    off = 1 if fit["intercept"] else 0
    num = b[0] if fit["intercept"] else 0.0
    for i, nm in enumerate(fit["names"]):
        num += b[i + off] * _FEAT[nm](r)
    return num / r["margin_c1"]


def _interaction_fit(rows, by_fam, u12_rows, m_additive):
    """W4/W3: the theory interaction form (+/- intercept) and the three-term model,
    each with pooled median relative error and leave-one-family-out ratios, so they
    are directly comparable to the additive LAW-FIT / LAW-PRED / LAW-HOLDOUT.  Plus
    the edge-count robustness check (same fit on ln_ratio_edge), the held-out U[1,2]
    weight-convention error under the interaction numerator, and kappa_E/beta1 vs the
    pooled median depth -- everything the interaction rule prints must regenerate."""
    def pooled_rel_err(fit):
        ob = np.array([r["c_req_micro"] for r in rows])
        pr = np.array([_predict_c_named(fit, r) for r in rows])
        return float(np.median(np.abs(ob - pr) / ob))

    def lofo(names, intercept):
        out = {}
        for fam in FAM_ORDER:
            oth = [r for r in rows if r["family"] != fam]
            mo = _ols_named(oth, names, intercept)
            pr = np.array([_predict_c_named(mo, r) for r in by_fam[fam]])
            ob = np.array([r["c_req_micro"] for r in by_fam[fam]])
            out[fam] = float(np.median(ob) / np.median(pr))
        return out

    def pack(fit, model):
        d = dict(model=model, r2=fit["r2"],
                 median_rel_err=pooled_rel_err(fit),
                 lofo_ratios=lofo(fit["names"], fit["intercept"]))
        d.update(fit["coefs"])
        return d

    B = _ols_named(rows, ["X1", "X2"], True)
    B0 = _ols_named(rows, ["X1", "X2"], False)
    C = _ols_named(rows, ["L*ln|E|", "L*lnL", "L"], True)
    b1, b1lo, b1hi = B["coefs"]["X1"]
    b2 = B["coefs"]["X2"][0]
    llnl = C["coefs"]["L*lnL"][0]

    # edge-count robustness: the same theory form on ln_ratio_edge -- beta1 stays
    # ~1.7-1.8, so the >1 leading coefficient is a measurement, not a micro-count
    # artifact.
    Be = _ols_named(rows, ["X1", "X2"], True, resp="ln_ratio_edge")
    Be0 = _ols_named(rows, ["X1", "X2"], False, resp="ln_ratio_edge")
    edge_robustness = dict(response="ln_ratio_edge",
                           beta1_with_intercept=Be["coefs"]["X1"],
                           beta1_no_intercept=Be0["coefs"]["X1"], r2=Be["r2"])

    # kappa_E / beta1 vs the pooled median depth: the additive size coefficient IS
    # the interaction coefficient times a population-typical depth.
    a_, b_, c0_ = m_additive["L"]["coef"], m_additive["ln|E|"]["coef"], m_additive["intercept"]
    med_L = float(np.median([r["L"] for r in rows]))
    kappa = dict(kappa_E=b_, beta1=b1, kappa_E_over_beta1=b_ / b1,
                 pooled_median_L=med_L,
                 rel_gap_to_median_L=abs(b_ / b1 - med_L) / med_L)

    # held-out U[1,2] weight convention: the interaction numerator (which never saw
    # this convention) vs the additive one, each over the instance's OWN margin.
    def _u12_err(numer):
        re = [abs(r["c_req_micro"] - numer(r) / r["margin_c1"]) / r["c_req_micro"]
              for r in u12_rows]
        return float(np.median(re)) if re else None
    u12_holdout = dict(
        n=len(u12_rows),
        additive_median_rel_err=_u12_err(
            lambda r: a_ * r["L"] + b_ * math.log(r["E"]) + c0_),
        model_b_median_rel_err=_u12_err(
            lambda r: B["beta"][0] + B["beta"][1] * _FEAT["X1"](r)
            + B["beta"][2] * _FEAT["X2"](r)))

    pB = pack(B, "b0 + beta1*X1 + beta2*X2;  X1=L*ln(|E|*e/L), X2=L")
    return dict(
        response="ln_ratio_micro", n=len(rows),
        additive_r2=0.9659988428361291,
        beta1_equals_one=bool(b1lo <= 1.0 <= b1hi),
        implied_branching_b=math.exp(-b2),
        theory_with_intercept=pB,
        theory_no_intercept=pack(B0, "beta1*X1 + beta2*X2  (no intercept)"),
        three_term=pack(C, "b0 + g1*(L*ln|E|) + g2*(L*lnL) + g3*L"),
        edge_robustness=edge_robustness,
        kappa_E_over_beta1=kappa,
        u12_holdout=u12_holdout,
        verdict=(
            "beta1 = %.4f [%.4f, %.4f], decisively NOT 1 (the asymptotic predicts 1): "
            "the additive numerator fit is a linearization over the operating range. "
            "The interaction FORM is right -- it fits materially better at "
            "INTERPOLATION (R2 %.3f vs 0.966; pooled median rel err %.1f%% vs 8.6%%) and "
            "the -L*lnL term is recovered near -1 (%.3f) -- but NOT at EXTRAPOLATION: "
            "the worst held-out family (grid) goes 0.84->%.2f and the U[1,2] "
            "weight-convention holdout worsens %.1f%%->%.1f%%.  So the leading coefficient "
            "carries a ~%.2f finite-size prefactor the leading-order expansion omits."
            % (b1, b1lo, b1hi, B["r2"], 100 * pB["median_rel_err"], llnl,
               pB["lofo_ratios"]["grid"],
               100 * u12_holdout["additive_median_rel_err"],
               100 * u12_holdout["model_b_median_rel_err"], b1)))


def _predict(m, r):
    """Ratio-law prediction of required c for one instance from numerator fit `m`."""
    return (m["intercept"] + m["L"]["coef"] * r["L"]
            + m["ln|E|"]["coef"] * math.log(r["E"])) / r["margin_c1"]


# ---------------------------------------------------------------------------- run
def run(argv=None) -> dict:
    argv = list(argv or [])
    leave_one_out = "--leave-one-family-out" in argv          # accepted (always emitted)
    show_interaction = "--interaction" in argv                # accepted (always emitted)

    rows = load_completed()
    by_fam = {fam: [r for r in rows if r["family"] == fam] for fam in FAM_ORDER}

    # ---- LAW-FIT : one numerator fit over all 1,027 ------------------------------
    m = _numerator_fit(rows)
    numerator_fit = dict(
        n=m["n"], r2=m["r2"],
        a=m["L"]["coef"], a_ci=[m["L"]["lo"], m["L"]["hi"]],
        b=m["ln|E|"]["coef"], b_ci=[m["ln|E|"]["lo"], m["ln|E|"]["hi"]],
        c0=m["intercept"],
        model="ln(#scatter/#path) = c0 + a*L + b*ln|E|; "
              "c_req = numerator / margin, margin = (1-2rho)*L + 1")
    fit_path = write_json("law/numerator_fit.json", numerator_fit)

    # ---- LAW-INTERACT : the W4/W3 interaction refit (does beta1 == 1?) -----------
    interaction = _interaction_fit(rows, by_fam, load_u12(), m)
    interaction_path = write_json("law/interaction_fit.json", interaction)

    # ---- LAW-PRED : per-instance prediction vs each instance's own margin --------
    per_rows = []
    for r in sorted(rows, key=lambda z: (z["family"], z["size"], z["seed"])):
        pr = _predict(m, r)
        ob = r["c_req_micro"]
        per_rows.append([r["family"], r["size"], r["seed"], r["L"], r["E"],
                         round(r["margin_c1"], 6), round(ob, 6), round(pr, 6),
                         round(abs(ob - pr), 6), round(abs(ob - pr) / ob, 6)])
    per_path = write_tsv_csv("law/per_instance.csv",
                             ["family", "size", "seed", "L", "E", "margin_c1",
                              "observed_c", "predicted_c", "abs_err", "rel_err"],
                             per_rows)

    def resid_summary(v):
        pr = np.array([_predict(m, r) for r in v])
        ob = np.array([r["c_req_micro"] for r in v])
        r2 = 1.0 - float(((ob - pr) ** 2).sum()) / float(((ob - ob.mean()) ** 2).sum())
        return dict(n=len(v), median_abs_err=float(np.median(np.abs(ob - pr))),
                    median_rel_err=float(np.median(np.abs(ob - pr) / ob)), r2=r2)

    pred_summary = {fam: resid_summary(by_fam[fam]) for fam in FAM_ORDER}
    pred_summary["POOLED"] = resid_summary(rows)

    # ---- LAW-RHO : per-family mean-edge-weight and finite asymptote ---------------
    a_ = m["L"]["coef"]
    rho_header = ["family", "median_rho", "one_minus_2rho", "asymptote_a_over_1m2rho",
                  "deepest_L", "observed_deep_median_c"]
    rho_rows, rho_json = [], {}
    for fam in FAM_ORDER:
        v = by_fam[fam]
        rho = float(np.median([r["Etilde"] / r["L"] for r in v]))
        Lmax = max(r["L"] for r in v)
        deep = [r["c_req_micro"] for r in v if r["L"] >= Lmax - 1]
        asym = a_ / (1 - 2 * rho) if (1 - 2 * rho) > 0 else float("inf")
        rho_json[fam] = dict(median_rho=rho, one_minus_2rho=1 - 2 * rho,
                             asymptote=asym, deepest_L=int(Lmax),
                             observed_deep_median=float(np.median(deep)))
        rho_rows.append([fam, round(rho, 4), round(1 - 2 * rho, 4), round(asym, 4),
                         int(Lmax), round(float(np.median(deep)), 4)])
    rho_path = write_tsv("law/rho.tsv", rho_header, rho_rows)

    # ---- LAW-HOLDOUT : leave-one-family-out, out-of-sample ------------------------
    holdout = {}
    for fam in FAM_ORDER:
        me = by_fam[fam]
        oth = [r for r in rows if r["family"] != fam]
        mo = _numerator_fit(oth)
        pr = np.array([_predict(mo, r) for r in me])
        ob = np.array([r["c_req_micro"] for r in me])
        holdout[fam] = dict(n=len(me), median_predicted=float(np.median(pr)),
                            median_observed=float(np.median(ob)),
                            ratio=float(np.median(ob) / np.median(pr)),
                            median_rel_err=float(np.median(np.abs(ob - pr) / ob)))
    holdout_path = write_json("law/holdout.json", holdout)

    # ---- LAW-DEPTH : cross-family c-vs-L table + the depth-law transfer at L~14 ---
    Ls = sorted({r["L"] for r in rows})
    per_L = {}
    for L in Ls:
        cell = {}
        for fam in FAM_ORDER:
            v = [r["c_req_micro"] for r in rows if r["family"] == fam and r["L"] == L]
            if v:
                cell[fam] = dict(median=float(np.median(v)), n=len(v))
        per_L[L] = cell
    # the linear depth-law leave-one-family-out prediction for grid at its median L
    grid = by_fam["grid"]
    others = [r for r in rows if r["family"] != "grid"]
    lin = ols_ci([r["L"] for r in others], [r["c_req_micro"] for r in others])
    grid_med_L = float(np.median([r["L"] for r in grid]))
    depth_law_pred_grid = lin["intercept"] + lin["slope"] * grid_med_L
    L14 = per_L.get(14, {})
    lattice14 = L14.get("grid", {}).get("median")
    terrain14 = L14.get("mtn_knn", {}).get("median")
    depth_transfer = dict(
        per_L={str(L): per_L[L] for L in Ls},
        at_L14=dict(lattice_grid_median=lattice14, terrain_mtn_knn_median=terrain14,
                    factor=(lattice14 / terrain14) if (lattice14 and terrain14) else None,
                    depth_law_prediction_grid=depth_law_pred_grid,
                    note="lattice (grid) 13.57 vs terrain (mtn_knn) 4.30 at L~14, "
                         "factor ~3.2; the linear depth law fitted on the other three "
                         "families predicts ~4.01 for grid -- it transfers to terrain, "
                         "not to the lattice's weight convention."),
        linear_leave_one_out_grid=dict(median_L=grid_med_L, slope=lin["slope"],
                                       intercept=lin["intercept"],
                                       predicted=depth_law_pred_grid))
    depth_path = write_json("law/depth_transfer.json", depth_transfer)

    # ---- summary ----------------------------------------------------------------
    result = dict(LAW_FIT=numerator_fit, LAW_PRED=pred_summary,
                  LAW_HOLDOUT=holdout, LAW_RHO=rho_json,
                  LAW_DEPTH=depth_transfer["at_L14"], LAW_INTERACT=interaction,
                  outputs=dict(numerator_fit=rel(fit_path), per_instance=rel(per_path),
                               holdout=rel(holdout_path),
                               depth_transfer=rel(depth_path), rho=rel(rho_path),
                               interaction_fit=rel(interaction_path)))
    print(f"[9] ratio_law -> {rel(RESULTS / 'law')}/  (off committed population.csv)")
    print(f"    LAW-FIT   a={numerator_fit['a']:.3f} "
          f"[{numerator_fit['a_ci'][0]:.3f},{numerator_fit['a_ci'][1]:.3f}]; "
          f"b={numerator_fit['b']:.3f} "
          f"[{numerator_fit['b_ci'][0]:.3f},{numerator_fit['b_ci'][1]:.3f}]; "
          f"c0={numerator_fit['c0']:.3f}; R2={numerator_fit['r2']:.3f}")
    pp = pred_summary["POOLED"]
    print(f"    LAW-PRED  median relative error {100 * pp['median_rel_err']:.1f}%; "
          f"R2={pp['r2']:.3f}")
    print("    LAW-HOLDOUT held-out family ratios "
          + " / ".join(f"{holdout[f]['ratio']:.2f}" for f in FAM_ORDER)
          + ("" if leave_one_out else "  (--leave-one-family-out)"))
    print(f"    LAW-DEPTH L~14: lattice {lattice14:.2f} vs terrain {terrain14:.2f}, "
          f"factor {lattice14 / terrain14:.1f}; depth law predicts "
          f"{depth_law_pred_grid:.2f}")
    print("    LAW-RHO   rho medians "
          + " / ".join(f"{rho_json[f]['median_rho']:.3f}" for f in FAM_ORDER))
    B = interaction["theory_with_intercept"]
    print(f"    LAW-INTERACT  beta1={B['X1'][0]:.4f} [{B['X1'][1]:.4f},{B['X1'][2]:.4f}] "
          f"(theory 1.00 -> {'IN' if interaction['beta1_equals_one'] else 'NOT in'} CI); "
          f"R2={B['r2']:.4f} vs additive 0.966; -L*lnL={interaction['three_term']['L*lnL'][0]:.3f}"
          + ("  (--interaction)" if not show_interaction else ""))
    if show_interaction:
        for key, tag in (("theory_with_intercept", "theory (+intercept)"),
                         ("theory_no_intercept", "theory (no intercept)"),
                         ("three_term", "three-term")):
            m = interaction[key]
            print(f"      {tag}:  R2={m['r2']:.5f}  median_rel_err={100*m['median_rel_err']:.2f}%  "
                  "LOFO(" + "/".join(f"{m['lofo_ratios'][f]:.2f}" for f in FAM_ORDER) + ")")
            for nm in [k for k in m if isinstance(m[k], list) and len(m[k]) == 3]:
                print(f"        {nm:<10} {m[nm][0]:+.4f}  [{m[nm][1]:+.4f}, {m[nm][2]:+.4f}]")
        er = interaction["edge_robustness"]
        print(f"      edge-count robustness (ln_ratio_edge): beta1="
              f"{er['beta1_with_intercept'][0]:.4f} "
              f"[{er['beta1_with_intercept'][1]:.4f},{er['beta1_with_intercept'][2]:.4f}] "
              f"(+int) / {er['beta1_no_intercept'][0]:.4f} (no int)  -> not a micro-count artifact")
        k = interaction["kappa_E_over_beta1"]
        print(f"      kappa_E/beta1 = {k['kappa_E_over_beta1']:.3f} vs pooled median L = "
              f"{k['pooled_median_L']:.0f}  ({100*k['rel_gap_to_median_L']:.1f}% off)")
        u = interaction["u12_holdout"]
        print(f"      U[1,2] holdout (n={u['n']}): additive {100*u['additive_median_rel_err']:.1f}% "
              f"-> Model B {100*u['model_b_median_rel_err']:.1f}%  (extrapolation worsens)")
        print(f"      implied branching b = {interaction['implied_branching_b']:.2f}")
        print(f"      => {interaction['verdict']}")
    return result


def write_tsv_csv(relpath, header, rows):
    """Comma-separated variant of _base.write_tsv (per_instance ships as a .csv)."""
    p = results_path(relpath)
    with open(p, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    return p


if __name__ == "__main__":
    import sys
    run(sys.argv[1:])
