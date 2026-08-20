"""Verification layer for the SPNN shortest-path release.

Three independent correctness checks plus a claims walker:

  * ``check_a`` -- decomposition == dense (OE-VALID)
  * ``check_b`` -- sampler == Boltzmann, TV at tau=3/5/10 (V-TV*)
  * ``check_c`` -- end-to-end readout vs Dijkstra (R-AGN)
  * ``check_claims`` -- walk CLAIMS.tsv: every row must carry a label, and every
    ``reproduces`` row must have its committed output on disk.

Run from the repo root::

    python -m verify            # Tier-1 checks + the claims walk
    python -m verify --full     # also the (non-reproducible) contrast variants
    python -m verify claims     # just the CLAIMS.tsv walk

This is the acceptance layer above ``experiments/``: it re-derives the
``validate.py`` claims from scratch and audits the whole claim table.
"""
