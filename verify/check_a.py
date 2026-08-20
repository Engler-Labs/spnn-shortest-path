"""check A -- the O(|E|) decomposition equals the explicit dense form (OE-VALID).

The segment-reduction membrane update and energy must agree exactly with
``b + W x`` for the explicitly built W. Reproduces OE-VALID: max per-neuron
discrepancy ~1.99e-7 in float32 (validate.check_decomposition).
"""

from __future__ import annotations

from spnn.validate import check_decomposition

from experiments._base import rel, write_json


def run(argv=None) -> dict:
    worst_u, worst_e = check_decomposition(verbose=False)
    result = {
        "OE-VALID": {
            "max_neuron_discrepancy": worst_u,
            "max_energy_discrepancy": worst_e,
            "expected": 1.99e-7,
            "ok": worst_u < 1e-4,
        }
    }
    out = write_json("validate/decomposition.json", result)
    ok = "OK" if worst_u < 1e-4 else "FAIL"
    print(f"[check A] decomposition == dense -> {rel(out)}")
    print(f"  OE-VALID  max|u_fast - u_dense| = {worst_u:.3e}  (expect ~1.99e-7)  {ok}")
    return result


if __name__ == "__main__":
    run()
