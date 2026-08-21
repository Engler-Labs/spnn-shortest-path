"""Experiment 16 -- ordering tolerance (20) checked against a compiled network (IV-C).

Claims: ORD-TOL (CLAIMS.tsv, section IV-C).
Output: results/tolerance/ordering.json

Removes the paper's last ``[DERIVED]`` tag.  Proposition 2's ordering tolerance --
that the energy minimum over the path sector stays the least-weight (Dijkstra) path
while the drive-bias offset lambda lies within

    tau = (2c/B) * min_{Q: L_Q != L_P} (W_Q - W*) / |L_Q - L_P|

-- was stated as derived but never checked against a compiled network (only the
right-hand quantity was measured).  Section V-A's margin was verified this way; this
mirrors it for the ordering tolerance, at L in {3, 5, 8}.

Method.  On small graphs where every simple s->t path is enumerable, compile the
AUGMENTED network (b_st = b_k) at alpha-scale c = 4 and read each path state's E_N
from ``sim.energy``.  E_N is exactly AFFINE in lambda -- E_SPC does not depend on
lambda and E_NC is affine in b_k = (lambda - D - c)/2 -- so two anchor compiles
(lambda = 0, 1) give the compiled network's E_N at every lambda exactly; the
ordering read off them IS the compiled network's ordering.  We report:

  * the measured inversion threshold (the nearest lambda at which the argmin over
    path states stops being Dijkstra's path), divided by tau.  >= 1 confirms the
    bound; how far above 1 says how conservative it is (the bound takes a min over
    rivals, so it is sufficient, not necessary -- ordering may survive past tau).
  * which rival Q inverts first, and whether it is the argmin of tau's min.
  * the sign of lambda-space the ordering holds over at 0, +/-0.5/0.9/1.1/2/5 * tau.
"""

from __future__ import annotations

import numpy as np

from spnn.compile import Params, compile_network
from spnn.graphs import dijkstra, random_graph
from spnn.sim import energy, ideal_state

from experiments._base import rel, write_json

D = 4.0
C = 4.0
SWEEP = [0.5, 0.9, 1.1, 2.0, 5.0]     # multiples of tau, applied at both signs
# Pinned enumerable instances, one per target depth (named seeds).
INSTANCES = [
    dict(L=3, n=10, p=0.12, seed=9),
    dict(L=5, n=10, p=0.12, seed=7),
    dict(L=8, n=14, p=0.12, seed=32),
]


def _enum_paths(n, edges, s, t, cap=200000):
    adj = {i: [] for i in range(n)}
    for a, b in edges:
        a, b = int(a), int(b)
        adj[a].append(b)
        adj[b].append(a)
    out, on, stack = [], [False] * n, [s]
    on[s] = True

    def rec(v):
        if len(out) >= cap:
            return
        if v == t:
            out.append(tuple(stack))
            return
        for u in adj[v]:
            if not on[u]:
                on[u] = True
                stack.append(u)
                rec(u)
                stack.pop()
                on[u] = False
    rec(s)
    return out


def _analyze(spec):
    n, edges, w = random_graph(spec["n"], spec["p"], seed=spec["seed"])
    t = n - 1
    idx = {}
    for i, (a, b) in enumerate(edges):
        idx[(int(a), int(b))] = idx[(int(b), int(a))] = float(w[i])
    wsum = lambda q: sum(idx[(q[i], q[i + 1])] for i in range(len(q) - 1))

    P, _ = dijkstra(n, edges, w, 0, t)
    P = tuple(P)
    Lp, Wstar = len(P) - 1, wsum(P)
    B = 2.0 * float(np.max(w))
    paths = _enum_paths(n, edges, 0, t)

    # tau: (2c/B) * min over different-length rivals of (W_Q - W*)/|L_Q - L_P|
    tau_terms = [((2 * C / B) * (wsum(q) - Wstar) / abs((len(q) - 1) - Lp), q)
                 for q in paths if (len(q) - 1) != Lp]
    tau, tau_arg = min(tau_terms, key=lambda z: z[0])

    # two anchor compiles -> E_N(path, lambda) = e0 + lambda * slope, exactly
    def en_at(lam):
        bk = (lam - D - C) / 2.0
        net = compile_network(n, edges, w, 0, t,
                              params=Params(D=D, alpha_scale=C, bias=bk, b_st=None),
                              augment_st=True)
        return {q: float(sum(energy(net, ideal_state(net, net.augmented_path(list(q))))))
                for q in paths}
    e0, e1 = en_at(0.0), en_at(1.0)
    slope = {q: e1[q] - e0[q] for q in paths}

    # self-check: at lambda=0 the compiled argmin must be Dijkstra's path
    argmin0 = min(paths, key=lambda q: e0[q])
    if argmin0 != P:
        raise AssertionError(f"lambda=0 argmin {argmin0} != Dijkstra {P}")

    # exact inversion lambda for each different-length rival
    inversions = []
    for q in paths:
        if (len(q) - 1) == Lp:
            continue                       # same length: parallel in lambda, never inverts
        denom = slope[q] - slope[P]
        if abs(denom) < 1e-12:
            continue
        lam_inv = (e0[P] - e0[q]) / denom
        inversions.append((abs(lam_inv), lam_inv, q))
    inversions.sort()
    measured_abs, measured_signed, first_q = inversions[0]

    # discrete sweep: does the argmin stay P at +/- m*tau ?
    sweep = {}
    for m in SWEEP:
        for sgn in (+1, -1):
            lam = sgn * m * tau
            am = min(paths, key=lambda q: e0[q] + lam * slope[q])
            sweep[f"{sgn*m:+g}tau"] = (am == P)

    return dict(
        instance=f"random_graph({spec['n']}, {spec['p']}, seed={spec['seed']})",
        L=Lp, n_paths=len(paths), B=B, opt_weight=Wstar,
        tau=tau, tau_binding_rival=dict(L=len(tau_arg) - 1, weight=wsum(tau_arg)),
        measured_threshold=measured_abs, measured_signed=measured_signed,
        measured_over_tau=measured_abs / tau,
        first_inverting_rival=dict(L=len(first_q) - 1, weight=wsum(first_q)),
        first_rival_is_tau_argmin=(first_q == tau_arg),
        holds_at=sweep)


def run(argv=None) -> dict:
    results = [_analyze(s) for s in INSTANCES]
    summary = dict(
        c=C, note="ordering tolerance (20) checked against the compiled augmented "
                  "network at b_st=b_k; E_N affine in lambda so two anchor compiles "
                  "give the exact compiled ordering at every lambda.",
        instances=results,
        min_measured_over_tau=min(r["measured_over_tau"] for r in results),
        all_hold_within_0p9tau=all(r["holds_at"]["+0.9tau"] and r["holds_at"]["-0.9tau"]
                                   for r in results),
        all_first_rival_is_tau_argmin=all(r["first_rival_is_tau_argmin"] for r in results))
    out = write_json("tolerance/ordering.json", summary)

    print(f"[16] ordering_tolerance -> {rel(out)}  (compiled network, c={C:.0f})")
    print(f"    {'L':>3} {'#paths':>6} {'tau':>9} {'measured':>9} {'meas/tau':>8} "
          f"{'holds<=0.9t':>11} {'firstQ=argmin':>13}")
    for r in results:
        h = r["holds_at"]["+0.9tau"] and r["holds_at"]["-0.9tau"]
        print(f"    {r['L']:>3} {r['n_paths']:>6} {r['tau']:>9.4f} "
              f"{r['measured_threshold']:>9.4f} {r['measured_over_tau']:>8.3f} "
              f"{str(h):>11} {str(r['first_rival_is_tau_argmin']):>13}")
    print(f"    min measured/tau = {summary['min_measured_over_tau']:.3f} "
          f"(>= 1 confirms the bound); ordering holds within +/-0.9*tau on all: "
          f"{summary['all_hold_within_0p9tau']}; first-inverting rival is the "
          f"tau-argmin on all: {summary['all_first_rival_is_tau_argmin']}")
    return summary


if __name__ == "__main__":
    run()
