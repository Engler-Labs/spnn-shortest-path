"""Experiment 14 -- tab:scale, the O(|E|) decomposition vs sparse-spmv scaling.

Claim: T-SCALE (CLAIMS.tsv, section VI-C, Tier 2, label=PINNED).
Output: results/bench/scaling.tsv

A thin driver over ``spnn.bench`` (the same decomposed-update-vs-explicit-sparse
measurement).  The STRUCTURAL columns (|E|, neurons, W nnz, W memory) are exact
and hardware-independent; the TIMING columns (decomp ms, spmv ms, speedup) are
measured on the running machine and are therefore hardware-dependent.

⚠️ PINNED, not reproduced.  CLAIMS labels T-SCALE ``pinned`` and the paper
disclaims tab:scale as *not a benchmark* (no repetitions, an oversubscribed
harness).  So this regenerates the table's STRUCTURE and reports fresh local
timings for context; it does not attempt to reproduce the manuscript's timing
cells, which are the authors' machine.  Only the structural columns should be
diffed against the paper.
"""

from __future__ import annotations

import time

import numpy as np

from spnn import Params, compile_network, random_graph
from spnn.bench import sparse_from_net
from spnn.sim import membrane

from experiments._base import rel, write_tsv

# The paper's tab:scale is 25 cells; these sizes give the structural ladder.
SIZES = (20, 40, 60, 90, 150, 250, 400)
P = 0.2
REPS = 20
DENSE_MAX = 40000          # skip the explicit sparse W above this many neurons


def run(argv=None) -> dict:
    rng = np.random.default_rng(0)
    header = ["n", "n_edges", "n_neurons", "W_nnz", "W_mem_MB",
              "decomp_ms", "spmv_ms", "speedup"]
    rows = []
    for n in SIZES:
        n_, edges, w = random_graph(n, P, seed=1)
        net = compile_network(n_, edges, w, 0, n_ - 1, Params())
        x = rng.random(net.n_neurons) < 0.05

        t0 = time.perf_counter()
        for _ in range(REPS):
            membrane(net, x)
        t_dec = (time.perf_counter() - t0) / REPS

        nnz = mem = t_sp = float("nan")
        if net.n_neurons <= DENSE_MAX:
            W = sparse_from_net(net)
            nnz = int(W.nnz)
            mem = (W.data.nbytes + W.indices.nbytes + W.indptr.nbytes) / 2 ** 20
            xf = x.astype(np.float32)
            t0 = time.perf_counter()
            for _ in range(REPS):
                net.b + W @ xf
            t_sp = (time.perf_counter() - t0) / REPS
            del W

        speedup = (t_sp / t_dec) if t_sp == t_sp else float("nan")
        rows.append([
            n, int(edges.shape[0]), int(net.n_neurons),
            nnz if nnz == nnz else "",
            f"{mem:.3f}" if mem == mem else "",
            f"{t_dec * 1e3:.4f}",
            f"{t_sp * 1e3:.4f}" if t_sp == t_sp else "",
            f"{speedup:.2f}" if speedup == speedup else "",
        ])

    out = write_tsv("bench/scaling.tsv", header, rows)
    print(f"[14] bench_scaling -> {rel(out)}   (PINNED: timings hardware-local)")
    print(f"     {len(rows)} sizes; structural columns exact, timing columns machine-dependent")
    return {"sizes": list(SIZES), "rows": len(rows), "note": "pinned; timings hardware-dependent"}


if __name__ == "__main__":
    run()
