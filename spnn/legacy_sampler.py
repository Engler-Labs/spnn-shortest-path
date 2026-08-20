"""Legacy (DELIBERATELY INCORRECT) Boltzmann sampler -- the pre-fix contrast.

*** THIS IS A REFERENCE IMPLEMENTATION OF A BUG, RETAINED ON PURPOSE. ***
It is NOT the simulator's sampler and must never be used to run the network.
Its sole reason to exist is to make the paper's §III-D pre-fix contrast
(CLAIMS row V-TVBAD, "TV = 0.235") REPRODUCIBLE, so the size of the correction
can be measured rather than asserted.  The correct sampler is
``spnn.sim.sampler_fire`` (logistic ``q = 1/(1 + tau*exp(-u)) = sigmoid(u - log
tau)``, refractory counter decremented FIRST); use that for everything real.

What the two documented defects are
-----------------------------------
Check B's *reference* sampler (an inline copy inside ``check_boltzmann``, since
replaced) once used the biased discretisation that ``sim.py``'s header warns
against.  Relative to the canonical ``sampler_fire`` it has TWO defects:

1. FIRING PROBABILITY -- it takes the neuron's continuous-time rate
   ``p_k = (1/tau) exp(u_k)`` directly as a per-step event probability via the
   exponential-clock survival ``pfire = 1 - exp(-p_k) = 1 - exp(-exp(u)/tau)``,
   i.e. WITHOUT the ``- log tau`` correction.  This form biases the chain toward
   silence and breaks the Boltzmann stationary distribution.  (The correct form
   is ``q = 1/(1 + tau*exp(-u))``.)

2. REFRACTORY ORDERING -- it decrements the refractory counter AFTER firing
   instead of before, so a neuron that fires stays active for ``tau - 1`` steps
   rather than the full ``tau``, further distorting the stationary odds
   ``P(on)/P(off) = tau / ((1 - q)/q)`` that the ``- log tau`` term is there to
   satisfy.

Both defects together, at ``tau=3, seed=1``, reproduce ``TV = 0.235`` against
the exact enumerated Boltzmann distribution (0.2354 at 1e6 steps, 0.2359 at the
4e6-step default), versus ``TV = 0.056`` for the corrected sampler -- the
0.235 -> 0.056 fix quoted in §III-D of the paper.

Provenance: the buggy inline sampler predates the public repo's first commit and
the internal reference implementation's first commit of its validation harness
(both already carry the fixed sampler), so it is not recoverable from git history;
its exact form is reconstructed here from the description preserved in an internal
analysis note ("``pfire = 1 - exp(-exp(u)/tau)``, decrement-after-fire"), and
confirmed to land on 0.235 +/- 1e-3.
"""

from __future__ import annotations

import itertools

import numpy as np

from .sim import U_CLIP


def legacy_sampler_fire(u, refrac, tau, rng):
    """The pre-fix, DELIBERATELY INCORRECT sampler step (see module docstring).

    Signature-compatible with ``spnn.sim.sampler_fire`` (mutates ``refrac`` in
    place, draws exactly one ``rng.random(N)`` per call, returns the new active
    mask) so it is a drop-in swap in ``check_boltzmann_legacy``.  It encodes the
    two documented defects:

      * DEFECT 1 -- exponential-clock firing prob ``1 - exp(-exp(u)/tau)``
        (the rate taken directly as a per-step probability, no ``- log tau``).
      * DEFECT 2 -- the refractory counter is decremented AFTER firing.

    DO NOT use this to run the network.  ``spnn.sim.sampler_fire`` is correct.
    """
    np.clip(u, -U_CLIP, U_CLIP, out=u)
    # DEFECT 1: rate p_k = exp(u)/tau taken directly as a per-step event prob
    # via the exponential-clock survival function (README's "biased" form).
    pfire = 1.0 - np.exp(-np.exp(u) / tau)
    fired = (refrac == 0) & (rng.random(refrac.shape[0]) < pfire)
    refrac[fired] = tau
    # DEFECT 2: decrement AFTER firing (correct sampler decrements first).
    np.subtract(refrac, 1, out=refrac, where=refrac > 0)
    return refrac > 0


def check_boltzmann_legacy(N=8, seed=1, steps=4_000_000, tau=3, verbose=True):
    """``validate.check_boltzmann`` mirror driven by the LEGACY (buggy) sampler.

    Byte-for-byte the same experiment setup as ``spnn.validate.check_boltzmann``
    -- identical ``W``/``b`` generation from ``seed`` (same RNG draw order), the
    same 2**N exact-Boltzmann enumeration, and the same per-state counting -- with
    the ONE difference that the step uses ``legacy_sampler_fire`` instead of the
    corrected ``sampler_fire``.  Returns ``(tv, kl)``.

    This exists ONLY to reproduce the pre-fix V-TVBAD contrast (TV ~ 0.235); it
    says nothing about the shipped simulator, which uses the correct sampler.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.6, size=(N, N))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    b = rng.normal(-1.0, 0.4, size=N)

    # enumerate the exact Boltzmann distribution (identical to check_boltzmann)
    states = np.array(list(itertools.product([0, 1], repeat=N)), dtype=np.float64)
    E = -(states @ b) - 0.5 * np.einsum("si,ij,sj->s", states, W, states)
    P = np.exp(-E - E.min())
    P /= P.sum()

    # sample with the DELIBERATELY INCORRECT sampler
    refrac = np.zeros(N, dtype=np.int32)
    counts = np.zeros(2**N, dtype=np.int64)
    pw = (2 ** np.arange(N))[::-1]
    x = np.zeros(N)
    for _ in range(steps):
        u = b + W @ x
        x = legacy_sampler_fire(u, refrac, tau, rng).astype(np.float64)
        counts[int(x.astype(int) @ pw)] += 1

    Q = counts / counts.sum()
    tv = 0.5 * float(np.abs(P - Q).sum())
    kl = float(np.sum(np.where(Q > 0, Q * np.log(np.maximum(Q, 1e-300) / P), 0.0)))
    if verbose:
        print(f"B(legacy). buggy sampler vs Boltzmann (N={N}, tau={tau}, {steps:,} steps)")
        print(f"   total variation = {tv:.4f}  (pre-fix contrast; expected ~0.235)")
        print(f"   KL(emp || exact) = {kl:.5f}")
    return tv, kl


if __name__ == "__main__":
    check_boltzmann_legacy(steps=600_000)
