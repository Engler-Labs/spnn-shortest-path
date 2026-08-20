"""Scaling benchmark for the decomposed update vs a sparse-spmv baseline."""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .compile import Params, compile_network
from .graphs import random_graph
from .sim import membrane


def sparse_from_net(net):
    """Explicit sparse W, the way a direct transcription of Eq. 4.2 would."""
    p = net.params
    rows, cols, vals = [], [], []
    order = np.argsort(net.seg, kind="stable")
    seg_sorted = net.seg[order]
    bounds = np.searchsorted(seg_sorted, np.arange(net.n_segments + 1))
    clus_of_seg = np.arange(net.n_segments) // 2

    for s in range(net.n_segments):
        mine = order[bounds[s]:bounds[s + 1]]
        i, c = clus_of_seg[s], s % 2
        opp = order[bounds[2 * i + (1 - c)]:bounds[2 * i + (1 - c) + 1]]
        # gamma (i): within component, all pairs
        if mine.size > 1:
            a, b = np.meshgrid(mine, mine, indexing="ij")
            m = a != b
            rows.append(a[m]); cols.append(b[m]); vals.append(np.full(m.sum(), p.I))
        # alpha / gamma(ii): across components
        if mine.size and opp.size:
            a, b = np.meshgrid(mine, opp, indexing="ij")
            same = net.nbr[a] == net.nbr[b]
            v = np.where(same, p.I, 1.0 - (net.w[a] + net.w[b]) / p.B)
            rows.append(a.ravel()); cols.append(b.ravel()); vals.append(v.ravel())

    bi = net.beta_idx
    for f in range(bi.shape[1]):
        idx = bi[:, f]; valid = idx >= 0
        rows.append(np.nonzero(valid)[0]); cols.append(idx[valid])
        vals.append(np.full(int(valid.sum()), p.D))

    W = sp.csr_matrix(
        (np.concatenate(vals).astype(np.float32),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(net.n_neurons, net.n_neurons))
    return W


def bench(sizes=(20, 40, 60, 90, 150, 250, 400), p=0.2, reps=20, dense_max=40000):
    print(f"{'n':>6} {'|E|':>8} {'neurons':>9} {'W nnz':>11} "
          f"{'W mem':>9} {'decomp':>9} {'spmv':>9} {'speedup':>8}")
    rng = np.random.default_rng(0)
    for n in sizes:
        n_, edges, w = random_graph(n, p, seed=1)
        net = compile_network(n_, edges, w, 0, n_ - 1, Params())
        x = rng.random(net.n_neurons) < 0.05

        t0 = time.perf_counter()
        for _ in range(reps):
            membrane(net, x)
        t_dec = (time.perf_counter() - t0) / reps

        nnz = mem = t_sp = float("nan")
        if net.n_neurons <= dense_max:
            W = sparse_from_net(net)
            nnz = W.nnz
            mem = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 2**20
            xf = x.astype(np.float32)
            t0 = time.perf_counter()
            for _ in range(reps):
                net.b + W @ xf
            t_sp = (time.perf_counter() - t0) / reps
            del W

        sp_s = f"{t_sp*1e3:8.2f}m" if t_sp == t_sp else "       --"
        nn_s = f"{nnz:11,.0f}" if nnz == nnz else "         --"
        mm_s = f"{mem:7.1f}MB" if mem == mem else "       --"
        su = f"{t_sp/t_dec:7.1f}x" if t_sp == t_sp else "      --"
        print(f"{n:6d} {edges.shape[0]:8,d} {net.n_neurons:9,d} {nn_s} "
              f"{mm_s} {t_dec*1e3:8.2f}m {sp_s} {su}")


if __name__ == "__main__":
    bench()
