"""Verification runner.

    python -m verify            # Tier-1: check A/B/C + the CLAIMS.tsv walk
    python -m verify --full     # also the non-reproducible contrast variants
    python -m verify claims     # only the CLAIMS.tsv audit

Prints what it skipped, so the Tier-1 run never silently omits anything.
"""

from __future__ import annotations

import sys

from . import check_a, check_b, check_c, check_claims, check_instance


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if argv and argv[0] == "claims":
        return 0 if check_claims.run().get("ok") else 1

    full = "--full" in argv
    passthrough = ["--full"] if full else []

    print("=== check instance: reference-instance regression guard ===")
    check_instance.run()
    print("\n=== check A: decomposition == dense ===")
    check_a.run()
    print("\n=== check B: sampler == Boltzmann ===")
    check_b.run(passthrough)
    print("\n=== check C: end-to-end readout ===")
    check_c.run(passthrough)

    if full:
        print("\n=== contrast variants ===")
        check_b.run(["--legacy-sampler"])          # V-TVBAD (reproduced via legacy sampler)
        # check_c --full already emitted the R-SLOT0 contrast above
    else:
        print("\nskipped (use --full): check B --legacy-sampler (V-TVBAD 0.235, "
              "reproduced via spnn.legacy_sampler), check C --slot0 (R-SLOT0 6/12, "
              "documented as not reproducible).")

    print("\n=== CLAIMS.tsv audit ===")
    res = check_claims.run()
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
