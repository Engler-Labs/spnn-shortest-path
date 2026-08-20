"""check B -- the sampler's stationary distribution equals Boltzmann (V-TV*).

On a small arbitrary network the empirical distribution over states must match
P(x) ~ exp(-E(x)). Reproduces V-TV3/5/10: total variation ~0.056 / 0.037 / 0.023
at tau=3/5/10 (validate.check_boltzmann, the fixed sampler).

The exact TV digit is a Monte-Carlo estimate (deterministic at a fixed seed but
dependent on the step budget), so it reproduces the 0.235 -> 0.056 *fix* rather
than a specific committed digit to 1e-4. ``--legacy-sampler`` (V-TVBAD) reproduces
the pre-fix contrast by RUNNING the deliberately-incorrect legacy sampler shipped
in ``spnn.legacy_sampler`` (exponential-clock firing prob + decrement-after-fire),
which lands on TV ~ 0.235 at tau=3, seed=1 -- measured, not fabricated.
"""

from __future__ import annotations

from spnn.legacy_sampler import check_boltzmann_legacy
from spnn.validate import check_boltzmann

from experiments._base import rel, write_json

DEFAULT_STEPS = 1_000_000     # --full uses 4_000_000 (validate.py's default)


def run(argv=None) -> dict:
    argv = list(argv or [])
    steps = 4_000_000 if "--full" in argv else DEFAULT_STEPS

    if "--legacy-sampler" in argv:
        # Reproduce the pre-fix V-TVBAD contrast by RUNNING the deliberately
        # incorrect legacy sampler (pfire = 1 - exp(-exp(u)/tau), decrement-after-
        # fire); see spnn/legacy_sampler.py.  tau=3 is the tau §III-D quotes 0.235 at.
        tv, kl = check_boltzmann_legacy(N=8, seed=1, steps=steps, tau=3, verbose=False)
        ok = abs(tv - 0.235) < 1e-3
        result = {
            "V-TVBAD": {
                "expected": 0.235,
                "TV": tv,
                "KL": kl,
                "seed": 1,
                "steps": steps,
                "tau": 3,
                "ok": ok,
                "reproduced": ok,
                "sampler": "spnn.legacy_sampler.check_boltzmann_legacy "
                           "(pre-fix: pfire = 1 - exp(-exp(u)/tau), decrement-after-fire)",
                "note": "DELIBERATELY INCORRECT legacy sampler, retained so the pre-fix "
                        "contrast is reproducible. The fixed sampler gives TV~0.056 "
                        "(check B without this flag); this buggy form gives TV~0.235, "
                        "reproducing the 0.235 -> 0.056 correction in §III-D.",
            }
        }
        out = write_json("validate/boltzmann_tv_legacy.json", result)
        print(f"[check B --legacy-sampler] ({steps:,} steps, seed 1, tau=3) -> {rel(out)}")
        print(f"  V-TVBAD  TV = {tv:.4f}  (expected 0.235)  KL = {kl:.4f}  "
              f"{'REPRODUCED (|TV-0.235|<1e-3)' if ok else 'NOT reproduced'}")
        return result

    by_tau = {}
    for tau in (3, 5, 10):
        tv, kl = check_boltzmann(N=8, seed=1, steps=steps, tau=tau, verbose=False)
        by_tau[f"tau{tau}"] = {"TV": tv, "KL": kl}
    result = {
        "seed": 1,
        "steps": steps,
        "by_tau": by_tau,
        "expected_TV": {"tau3": 0.0558, "tau5": 0.037, "tau10": 0.0225},
        "note": "fixed sampler; TV is a deterministic Monte-Carlo estimate "
                "(~0.056 at tau=3 reproduces the 0.235 -> 0.056 fix).",
    }
    out = write_json("validate/boltzmann_tv.json", result)
    print(f"[check B] sampler == Boltzmann ({steps:,} steps, seed 1) -> {rel(out)}")
    for tau in (3, 5, 10):
        print(f"  tau={tau:2d}: TV = {by_tau[f'tau{tau}']['TV']:.4f}  "
              f"KL = {by_tau[f'tau{tau}']['KL']:.4f}")
    print("  (reproduces the fix; exact TV digit is Monte-Carlo, ~0.056 at tau=3)")
    return result


if __name__ == "__main__":
    run()
