"""SPNN Paper 1 experiments.

Each experiment is a module exposing ``run(argv=None) -> dict`` that computes its
result, writes its committed output under ``results/``, and returns a summary.

Run one by id from the repo root::

    python -m experiments 7           # run experiment 7
    python -m experiments list        # list the catalog
    python -m experiments all         # run every built tier-1 experiment
    python -m experiments all --full  # include tier-2 (seeded/expensive)

The id -> experiment mapping is the single table in ``experiments/registry.py``.
"""
