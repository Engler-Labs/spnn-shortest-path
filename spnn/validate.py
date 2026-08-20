"""
Three independent correctness checks.

A. decomposition == dense.  The O(|E|) segment-reduction membrane update
   must agree exactly with b + W x for the explicitly built W.  This is
   what catches indexing and edge-class errors in the compiler.

B. sampler == Boltzmann.  On a small arbitrary network, the empirical
   distribution over network states must match P(x) ~ exp(-E(x)).  This
   is what catches discretisation errors in Eq. 4.1, which is exactly
   where a reimplementation silently breaks the theory.

C. end-to-end.  On small graphs, does the induced graph contain the
   Dijkstra shortest path?
"""

from __future__ import annotations

import itertools

import numpy as np

from .compile import Params, compile_network, dense_weights
from .graphs import dijkstra, random_graph
from .sim import energy, membrane, sampler_fire, simulate


def check_decomposition(n=8, p=0.45, seed=3, trials=25, verbose=True):
    n, edges, w = random_graph(n, p, seed)
    net = compile_network(n, edges, w, 0, n - 1, Params())
    W = dense_weights(net)
    rng = np.random.default_rng(seed)

    worst_u = 0.0
    worst_e = 0.0
    for _ in range(trials):
        x = rng.random(net.n_neurons) < 0.3
        u_fast = membrane(net, x)
        u_ref = net.b + W @ x.astype(np.float64)
        worst_u = max(worst_u, float(np.max(np.abs(u_fast - u_ref))))

        e_nc, e_spc = energy(net, x)
        xf = x.astype(np.float64)
        e_ref = -float(net.b @ xf) - 0.5 * float(xf @ W @ xf)
        # Eq. 4.10 adds the constant |alpha| into E_NC; compare totals
        # against the raw energy plus that constant.
        worst_e = max(worst_e, abs((e_nc + e_spc) - e_ref))

    if verbose:
        print(f"A. decomposition vs dense (N={net.n_neurons} neurons)")
        print(f"   max |u_fast - u_dense|  = {worst_u:.3e}")
        print(f"   max |E_fast - E_dense|  = {worst_e:.3e}")
    return worst_u, worst_e


def check_boltzmann(N=8, seed=1, steps=4_000_000, tau=3, verbose=True):
    """Sampler check on an arbitrary small network with enumerable states."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.6, size=(N, N))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    b = rng.normal(-1.0, 0.4, size=N)

    # enumerate the exact Boltzmann distribution
    states = np.array(list(itertools.product([0, 1], repeat=N)), dtype=np.float64)
    E = -(states @ b) - 0.5 * np.einsum("si,ij,sj->s", states, W, states)
    P = np.exp(-E - E.min())
    P /= P.sum()

    # sample
    refrac = np.zeros(N, dtype=np.int32)
    counts = np.zeros(2**N, dtype=np.int64)
    pw = (2 ** np.arange(N))[::-1]
    x = np.zeros(N)
    for _ in range(steps):
        # goes through the SAME sampler step as sim.simulate (single source of
        # truth) -- decrement-first + logistic q live in sampler_fire.
        u = b + W @ x
        x = sampler_fire(u, refrac, tau, rng).astype(np.float64)
        counts[int(x.astype(int) @ pw)] += 1

    Q = counts / counts.sum()
    tv = 0.5 * float(np.abs(P - Q).sum())
    kl = float(np.sum(np.where(Q > 0, Q * np.log(np.maximum(Q, 1e-300) / P), 0.0)))
    if verbose:
        print(f"B. sampler vs Boltzmann (N={N}, tau={tau}, {steps:,} steps)")
        print(f"   total variation = {tv:.4f}")
        print(f"   KL(emp || exact) = {kl:.5f}")
        top = np.argsort(-P)[:5]
        print("   top states   exact     empirical")
        for s in top:
            print(f"     {format(s, f'0{N}b')}    {P[s]:.4f}    {Q[s]:.4f}")
    return tv, kl


def check_end_to_end(sizes=(10, 15, 20), p=0.25, trials=3, steps=1500, verbose=True):
    rows = []
    for n in sizes:
        hit = 0
        gaps = []
        for t in range(trials):
            n_, edges, w = random_graph(n, p, seed=100 + t)
            src, tgt = 0, n_ - 1
            true_path, true_cost = dijkstra(n_, edges, w, src, tgt)
            net = compile_network(n_, edges, w, src, tgt, Params())
            tr = simulate(net, steps=steps, seed=t, source=src, target=tgt,
                          irm_threshold=1.5, track_paths=True)
            if tr.best_path is not None:
                gaps.append((tr.best_cost - true_cost) / true_cost)
                if abs(tr.best_cost - true_cost) < 1e-6:
                    hit += 1
        rows.append((n, hit / trials, float(np.mean(gaps)) if gaps else np.nan,
                     len(gaps) / trials))
    if verbose:
        print("C. end-to-end vs Dijkstra")
        print("    n   exact-hit   mean rel. gap   feasible-rate")
        for n, h, g, f in rows:
            print(f"  {n:3d}      {h:.2f}          {g:+.3f}           {f:.2f}")
    return rows


if __name__ == "__main__":
    check_decomposition()
    print()
    check_boltzmann(steps=600_000)
    print()
    check_end_to_end()
