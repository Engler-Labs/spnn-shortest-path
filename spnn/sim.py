"""
Simulator for the Ch. 4 stochastic spiking network.

Neuron model (Eq. 4.1-4.3):

    p_k(t) = (1/tau) exp(u_k(t))   if k has not fired within tau steps
           = 0                      otherwise
    u_k(t) = b_k + sum_l w_kl x_l(t)
    x_l(t) = 1 if l spiked in (t - tau, t]

The PSP length and the refractory period are both tau; that identity is
what makes the stationary distribution exactly Boltzmann (Buesing et al.
2011), so it is preserved here rather than tuned.

Discretisation.  Naively setting the per-step firing probability to
p_k dt, or to the exponential-clock survival 1 - exp(-p_k dt), both bias
the chain toward silence and break the Boltzmann stationary distribution
(see validate.check_boltzmann).  In discrete time with a deterministic
tau-step active window, the stationary odds of a neuron are

    P(on)/P(off) = tau / ((1 - q)/q)

so the neural computability condition P(on)/P(off) = exp(u_k) forces

    q_k = exp(u_k) / (tau + exp(u_k)) = sigmoid(u_k - log tau)

This is the form used below.  It is automatically in [0, 1].  Ordering
matters too: the refractory counter is decremented at the TOP of the
step, so a neuron that fires is active for exactly tau steps rather than
tau - 1.

The membrane update never forms W.  Per step:
    A[s]  = # active neurons in segment s
    Sw[s] = sum of w over active neurons in segment s
which give every alpha and gamma contribution in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .compile import Network

U_CLIP = 20.0  # guards exp() overflow; exp(20)/tau is already ~1 per step


def sampler_fire(u, refrac, tau, rng):
    """Canonical discrete-time Boltzmann update -- the single source of truth.

    Given a membrane potential ``u`` and refractory state ``refrac``, decrement
    the refractory counter (a neuron fires for exactly ``tau`` steps), then fire
    with q = sigmoid(u - log tau).  Mutates ``refrac`` in place and returns the
    new active mask.  Both ``simulate`` and ``validate.check_boltzmann`` MUST go
    through here -- keeping the two sampler loops in sync is what prevents the
    discretisation bug that once silently broke check B.
    """
    np.subtract(refrac, 1, out=refrac, where=refrac > 0)
    np.clip(u, -U_CLIP, U_CLIP, out=u)
    pfire = 1.0 / (1.0 + tau * np.exp(-u))
    fired = (refrac == 0) & (rng.random(refrac.shape[0]) < pfire)
    refrac[fired] = tau
    return refrac > 0


def anneal_gain(p, t: int, steps: int) -> float:
    """Effective inhibition gain at step t under the configured schedule.

    Ramps permissive -> strict so the path can nucleate before off-path leakage
    is suppressed.  "cyclic" is a sawtooth reheat (annealing with restarts).
    """
    g0, g1 = p.inhib_gain_start, p.inhib_gain
    mode = p.inhib_anneal
    if mode == "none":
        return g1
    if mode == "cyclic":
        period = p.inhib_cycle if p.inhib_cycle > 0 else steps
        frac = (t % max(period, 1)) / max(period, 1)
        return g0 + (g1 - g0) * frac
    frac = t / max(steps - 1, 1)
    if mode == "linear":
        return g0 + (g1 - g0) * frac
    if mode == "exponential":
        return g1 - (g1 - g0) * float(np.exp(-5.0 * frac))
    raise ValueError(f"unknown inhib_anneal {mode!r}")


@dataclass
class Trace:
    energy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    energy_nc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    energy_spc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_active: np.ndarray = field(default_factory=lambda: np.zeros(0))
    induced_edges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    irm_times: list = field(default_factory=list)
    best_path: list | None = None
    best_cost: float = np.inf
    first_path: list | None = None
    first_cost: float = np.inf
    first_step: int = -1
    contain_count: int = 0  # # steps the induced graph HELD a tracked edge set
    ident_count: int = 0  # # steps the induced graph WAS EXACTLY the tracked edge set
    realised_sampled: int = 0  # # steps checked for a realised simple s-t path
    path_shaped_count: int = 0  # ... of which the induced graph WAS a simple s-t path
    ident_weight_count: int = 0  # ... of which that path had weight == optimal (ties OK)
    # --- realised path shape.  *** READ THE FRAME NOTE. *** -------------------
    # ``realised_hops`` counts hops IN THE FRAME ``simulate`` WAS DRIVEN IN.  Under
    # ``compile_network(augment_st=True)`` a run driven between ``net.aug_source`` and
    # ``net.aug_target`` traverses the two zero-weight S->source and target->T edges as
    # well, so AN ORIGINAL L-HOP PATH IS COUNTED AS L + 2 HERE.  That offset has now
    # produced a wrong reading twice (once on a dense test graph, and three stage-9 deep
    # instances that looked like near-misses at the optimum's depth when they were the
    # rival at its own depth), and in neither case could a prose-vs-provenance check
    # have caught it: the number was transcribed faithfully and MEANT something else.
    # So the frame is now carried WITH the data instead of being reconstructed by each
    # caller.  *** REPORT ``realised_hops_orig``; it is the original graph's hop count
    # and is the project default. ***  ``realised_hops`` is kept unchanged so that every
    # committed replay still reproduces bit-for-bit, and is the RAW field, not the
    # reporting one.  ``realised_weight_sum`` is frame-free (the added edges weigh 0).
    realised_hops: dict = field(default_factory=dict)  # RAW: hops in ``hop_frame``
    realised_hops_orig: dict = field(default_factory=dict)  # hops in the ORIGINAL graph
    hop_offset: int = 0  # augmented terminal edges included in ``realised_hops``
    hop_frame: str = "original"  # frame of ``realised_hops``: "original" | "augmented"
    realised_weight_sum: float = 0.0  # sum of realised path weights (for comp. ratio)
    q_sum: float = 0.0  # sum over steps of the fraction of SILENT slots (segments)
    # ... and the same at CLUSTER granularity (a cluster = one vertex = two segments).
    # These two are what separate "inhibition thinned everything uniformly" from
    # "inhibition silenced whole clusters": at MATCHED total activity, a mechanism that
    # correlates silence within a cluster must occupy FEWER DISTINCT CLUSTERS, i.e. show a
    # higher q_cluster and more active neurons per occupied cluster.  Pure instrumentation
    # -- one extra bincount under the existing track_q flag, no random draws, so a run's
    # trajectory is unchanged.
    q_cluster_sum: float = 0.0  # sum over steps of the fraction of SILENT clusters
    # --- per-segment / per-cluster resolution of the SAME two bincounts ------
    # Filled only when ``track_detail=True`` (default off; the two arrays stay None
    # and nothing else changes).  ``q_sum`` above averages over ALL n_segments, and
    # under ``compile_network(augment_st=True)`` two of those segments -- component 1
    # of the single-neuron terminals S and T -- CONTAIN NO NEURON and are therefore
    # silent by construction on every step.  That is a fixed +1/(n_vertices) offset on
    # q which is NOT comparable to an unaugmented ladder's q.  Keeping the per-segment
    # silent counts lets any restricted q (original vertices only, live segments only)
    # be formed after the run instead of trusting the aggregate.  The per-cluster
    # active sums answer the matching question for ACTIVITY -- e.g. how much activity
    # sits in the source and target clusters specifically.
    # Pure instrumentation: both are accumulated from the A / Ac bincounts that
    # ``track_q`` already computes, so no extra random draws and no trajectory change.
    seg_silent_steps: np.ndarray | None = None  # (n_segments,) int64
    cluster_active_sum: np.ndarray | None = None  # (n_vertices,) int64
    # --- dwell (seeded-start diagnostic) ------------------------------------
    # How long the chain HOLDS the tracked configuration before first leaving it,
    # counted from t=0.  Only meaningful when the run was seeded at that state
    # (init_state=ideal_state(...)); from empty it is 0 by construction.  This is
    # the direct dwell measurement that the seeded-start diagnostic could only proxy
    # with id_ideal.
    dwell_exact: int = 0  # leading consecutive steps with induced graph == contain_edges
    escape_exact: int = -1  # step at which that run ended (-1 = never left)
    dwell_weight: int = 0  # leading consecutive SAMPLED steps that were optimal-weight paths
    escape_weight: int = -1  # sampled-step index at which that run ended (-1 = never)
    # --- per-step indicator series (record_indicators=True) ------------------
    # Kept so an effective sample size can be estimated on the INDICATOR itself
    # rather than only on a correlated proxy.  uint8 to stay cheap.
    ident_series: np.ndarray | None = None  # len == steps
    contain_series: np.ndarray | None = None  # len == steps
    pshp_series: np.ndarray | None = None  # len == realised_sampled
    identw_series: np.ndarray | None = None  # len == realised_sampled
    # --- matched-activity mass classification (mass_L=...) -------------------
    # The Ch. 4 discrimination condition compares AGGREGATE MASS on chained path
    # states against equal-activity scatter states, so every one of these counters
    # is conditioned on the active count, never on containment of a specific path.
    active_hist: np.ndarray | None = None  # over ALL steps; index = # active neurons
    band_steps: int = 0  # steps with |n_active - 2L| <= band halfwidth (classified)
    matched_steps: int = 0  # ... of which n_active == 2L exactly
    consistent_steps: int = 0  # ... of which ALSO beta==L and gamma==0 (classifiable)
    mass_path: int = 0  # consistent AND alpha == L-1  (a chain of L edges)
    mass_scatter: int = 0  # consistent AND alpha == 0    (L disjoint edges)
    mass_mixed: int = 0  # consistent AND 0 < alpha < L-1 (partial chains)
    mass_closed: int = 0  # consistent AND alpha >= L      (a closed / branched chain)
    alpha_hist: dict = field(default_factory=dict)  # alpha_pairs -> count, consistent only
    mass_e_path: float = 0.0  # sum of E over path-class states
    mass_e_scatter: float = 0.0  # sum of E over scatter-class states
    mass_aw_path: float = 0.0  # sum of the alpha weight-sum over path-class states
    mass_aw_scatter: float = 0.0  # ... over scatter-class states
    # per-step class code, len == steps.  0 = outside the band (never classified),
    # 1 = in band but off the matched shell, 2 = path, 3 = scatter, 4 = mixed,
    # 5 = closed.  Kept so n_eff can be estimated on the class indicator ITSELF
    # rather than on a proxy -- these rates are what a budget gets sized off.
    mass_series: np.ndarray | None = None
    # per-step active count, len == steps.  ``n_active`` above is subsampled by
    # ``record_every`` (it rides along with the O(n) energy call), which is useless as
    # an autocorrelation proxy for a per-step rate.  This one is per-step and is the
    # universal n_eff proxy for the band/class indicators, which are degenerate
    # whenever they never fire -- exactly the case a budget has to be sized in.
    active_series: np.ndarray | None = None


def membrane(net: Network, x: np.ndarray) -> np.ndarray:
    """u_k = b_k + sum_l w_kl x_l, computed in O(n_neurons)."""
    p = net.params
    xf = x.astype(np.float32)

    A = np.bincount(net.seg, weights=xf, minlength=net.n_segments)
    Sw = np.bincount(net.seg, weights=xf * net.w, minlength=net.n_segments)

    A_own = A[net.seg]
    A_opp = A[net.opp_seg]
    Sw_opp = Sw[net.opp_seg]
    # A single-neuron auxiliary terminal points ``partner`` at itself; the
    # validity mask zeroes it so the gamma (ii) term does not become a
    # self-inhibition.  All-ones (hence an exact no-op) unaugmented.
    x_par = xf[net.partner] * net.partner_valid

    # alpha: opposite component, different index.  f is separable, so
    #   sum_l x_l f(w_k, w_l) = (1 - w_k/B) A_opp - Sw_opp/B
    # then subtract the same-index term, which is gamma not alpha.
    # alpha_scale (c) multiplies the whole alpha contribution.
    alpha = (1.0 - net.w / p.B) * A_opp - Sw_opp / p.B
    alpha -= x_par * (1.0 - 2.0 * net.w / p.B)
    u = p.alpha_scale * alpha

    # gamma (i): every other neuron in the same WTA component
    u += p.I * (A_own - xf)
    # gamma (ii): same index, opposite component
    u += p.I * x_par

    # beta: fixed-weight D gather over graph edges
    bi = net.beta_idx
    for f in range(bi.shape[1]):
        idx = bi[:, f]
        valid = idx >= 0
        u[valid] += p.D * xf[idx[valid]]

    return u + net.b


def energy(net: Network, x: np.ndarray) -> tuple[float, float]:
    """Return (E_NC, E_SPC) from Eq. 4.11-4.12, in O(n_neurons)."""
    p = net.params
    xf = x.astype(np.float64)

    A = np.bincount(net.seg, weights=xf, minlength=net.n_segments)
    Sw = np.bincount(net.seg, weights=xf * net.w, minlength=net.n_segments)
    A_own = A[net.seg]
    A_opp = A[net.opp_seg]
    Sw_opp = Sw[net.opp_seg]
    # A single-neuron auxiliary terminal points ``partner`` at itself; the
    # validity mask zeroes it so the gamma (ii) term does not become a
    # self-inhibition.  All-ones (hence an exact no-op) unaugmented.
    x_par = xf[net.partner] * net.partner_valid

    # alpha count |alpha| over active pairs, and the weighted sum
    alpha_pairs = 0.5 * float(np.sum(xf * (A_opp - x_par)))
    alpha_wsum = 0.5 * float(np.sum(xf * ((net.w * (A_opp - x_par)) + (Sw_opp - net.w * x_par))))
    E_spc = p.alpha_scale * alpha_wsum / p.B

    gamma_pairs = 0.5 * float(np.sum(xf * (A_own - xf))) + 0.5 * float(np.sum(xf * x_par))
    beta_pairs = 0.0
    bi = net.beta_idx
    for f in range(bi.shape[1]):
        idx = bi[:, f]
        valid = idx >= 0
        beta_pairs += 0.5 * float(np.sum(xf[valid] * xf[idx[valid]]))

    lin = float(np.sum(net.b * xf))
    # sign convention: E_N = -(sum b x) - 1/2 sum sum x x w; alpha count scaled by c
    E_nc = -(lin + beta_pairs * p.D + gamma_pairs * p.I) - p.alpha_scale * alpha_pairs
    return E_nc, E_spc


def induced_edges(net: Network, x: np.ndarray) -> np.ndarray:
    """Boolean mask over graph edges present in the induced graph (Def. 2).

    Component-agnostic: the WTA slot a neuron occupies is arbitrary on an
    undirected graph, so an edge counts if EITHER slot on each side is active.
    """
    u = x[net.edge_u[:, 0]] | x[net.edge_u[:, 1]]
    v = x[net.edge_v[:, 0]] | x[net.edge_v[:, 1]]
    return u & v


def simulate(
    net: Network,
    steps: int = 1000,
    seed: int = 0,
    irm_threshold: float | None = 1.5,
    record_every: int = 1,
    track_paths: bool = True,
    source: int = 0,
    target: int = 1,
    contain_edges: np.ndarray | None = None,
    track_realised: int = 0,
    opt_weight: float | None = None,
    track_q: bool = False,
    track_detail: bool = False,
    init_state: np.ndarray | None = None,
    record_indicators: bool = False,
    mass_L: int | None = None,
    mass_band: int = 0,
    on_step=None,
    drive: np.ndarray | None = None,
) -> Trace:
    """Run the network dynamics.

    irm_threshold : the IRM fires when the induced graph has more than
        irm_threshold * n_vertices edges, resetting the network to the
        empty state.  None disables the IRM.
    track_detail : keep the ``track_q`` bincounts PER SEGMENT and PER CLUSTER
        (``Trace.seg_silent_steps`` / ``Trace.cluster_active_sum``) instead of only
        their step-averaged scalars.  Requires ``track_q``.  Default False, in which
        case both arrays stay None and nothing else changes -- it draws no random
        numbers, so a run's trajectory is bit-identical either way.  Added for the
        augmented-network gain->q recalibration, which needs (i) a q restricted to
        the ORIGINAL graph's segments, because ``augment_st=True`` adds two segments
        that hold no neuron and are silent by construction, and (ii) activity
        resolved to the source and target clusters specifically.
    record_indicators : also keep the per-step containment / identification
        indicator series.  Pure instrumentation -- it draws no random numbers
        and changes no dynamics -- but it is what lets an effective sample
        size be estimated on the indicator itself instead of on a proxy.
    mass_L : enable the matched-activity mass classification at path length L.
        The active-count histogram is then accumulated over EVERY step (O(1)),
        and ``mass_class`` -- which is O(n_neurons) and is the expensive part --
        is evaluated ONLY on steps whose active count lies within ``mass_band``
        of 2L.  Testing the cheap predicate first is the whole point: the
        measurement conditions on matched activity anyway.
    mass_band : halfwidth (in active neurons) of that pre-filter band.  0 means
        the matched shell exactly; a positive value widens the band that gets
        classified without widening what can be *counted* as path or scatter
        (both classes require n_active == 2L by construction).
    on_step : optional observer ``on_step(t, x, mask, tr)`` invoked at the END of
        every step, after the state, the induced-edge mask and all counters for that
        step are final.  PURE INSTRUMENTATION -- it draws no random numbers and the
        simulator ignores its return value, so a run's trajectory is bit-identical
        with it and without it.  It exists so a diagnostic can look at what the
        induced subgraph is doing BETWEEN the rare interesting configurations without
        either duplicating this loop (the two-sampler-loop divergence this module's
        header warns about) or adding another single-purpose flag to ``Trace``.
        ``x`` and ``mask`` are freshly allocated each step, so retaining them is safe.
    drive : optional per-neuron external drive, shape ``(n_neurons,)``, added to ``u`` on
        exactly the schedule the S/T drive uses (every step under ``st_mode="always_on"``,
        else every ``st_period``).  ***Default None, which is the historical behaviour
        EXACTLY*** -- the fixed ``+= p.D`` on the ``net.st_drive`` mask.  Passing
        ``p.D * net.st_drive`` reproduces that bit-for-bit and stage 11 asserts it does.

        It exists because the drive's magnitude and its GEOMETRY were not separable
        before: ``p.D`` is simultaneously the beta coupling weight and the S/T drive
        magnitude, and ``net.st_drive`` is a bare boolean mask, so "drive the interior
        too" and "drive the terminals less" were both unexpressible.

        *** READ THIS BEFORE USING IT: A STATIC DRIVE IS A BIAS, AND A BIAS ON CLUSTERS A
        LONGER PATH VISITS MORE OF IS A LAMBDA SHIFT. ***  An ideal L_aug-edge path state
        has ``2*L_aug`` active neurons, of which exactly 2 sit in the auxiliary terminals
        S and T.  So a uniform drive ``delta`` on every NON-terminal cluster lowers that
        state's energy by ``delta*(2*L_aug - 2)``, i.e.

            E_NC  =  -(lambda + 2*delta) * L_aug  +  (2*delta - 4)

        -- algebraically ``lambda -> lambda + 2*delta``, verified in an internal
        harness.  Since lambda PRICES
        LENGTH (a negative lambda makes the deep optimum stop being the energy minimum --
        stage 10), an "interior drive" is not a geometric intervention at all: it
        re-prices length, and at depth it can invert the answer the network is supposed to
        give.  Modulo the b_k redefinition that absorbs it, the whole LAMBDA-NEUTRAL family
        of static drives is ONE-DIMENSIONAL -- the terminal drive strength -- because the
        terminals are the only two clusters every path state occupies equally regardless
        of L.  Drive geometry that is genuinely free must therefore be non-static.

    HOP FRAME: the hop counts this fills in are in the frame of the ``source`` and
    ``target`` passed here.  Driving an augmented network between ``net.aug_source``
    and ``net.aug_target`` adds exactly one hop at each end.  ``Trace`` carries the
    offset and the converted ``realised_hops_orig``; see the note on ``Trace``.
    """
    rng = np.random.default_rng(seed)
    p = net.params
    N = net.n_neurons

    refrac = np.zeros(N, dtype=np.int16)  # steps remaining active
    x = np.zeros(N, dtype=bool)
    if init_state is not None:  # seed the sampler at a given state (mass/mixing test)
        x = np.asarray(init_state, dtype=bool).copy()
        refrac = np.where(x, p.tau, 0).astype(np.int16)

    # --- activity-dependent inhibition state (Ch. 5) --------------------
    inhib = net.inhib if p.inhib_kind == "subnetwork" else None
    if inhib is not None:
        refrac_i = np.zeros(inhib.n_inhib, dtype=np.int16)
        x_i = np.zeros(inhib.n_inhib, dtype=bool)
    meanfield = p.inhib_kind == "meanfield"
    Y = 0.0  # leaky integrator of total activity (mean-field)
    Yhist: list[float] = []  # history for the "directed" (lagged) scheme

    nrec = steps // record_every
    tr = Trace(
        energy=np.zeros(nrec),
        energy_nc=np.zeros(nrec),
        energy_spc=np.zeros(nrec),
        n_active=np.zeros(nrec, dtype=np.int32),
        induced_edges=np.zeros(nrec, dtype=np.int32),
    )

    if track_detail:
        if not track_q:
            raise ValueError("track_detail requires track_q (it reuses its bincounts)")
        tr.seg_silent_steps = np.zeros(net.n_segments, dtype=np.int64)
        tr.cluster_active_sum = np.zeros(net.n_vertices, dtype=np.int64)

    if record_indicators:
        n_sampled = (steps + track_realised - 1) // track_realised if track_realised else 0
        tr.ident_series = np.zeros(steps, dtype=np.uint8)
        tr.contain_series = np.zeros(steps, dtype=np.uint8)
        tr.pshp_series = np.zeros(n_sampled, dtype=np.uint8)
        tr.identw_series = np.zeros(n_sampled, dtype=np.uint8)

    if mass_L is not None:
        tr.active_hist = np.zeros(N + 1, dtype=np.int64)
        tr.mass_series = np.zeros(steps, dtype=np.uint8)
        tr.active_series = np.zeros(steps, dtype=np.int32)
        band_lo = max(0, 2 * mass_L - mass_band)
        band_hi = 2 * mass_L + mass_band
        MASS_CODE = {"other": 1, "path": 2, "scatter": 3, "mixed": 4, "closed": 5}

    # Each augmented terminal the run is driven from/to contributes exactly one
    # zero-weight edge to every realised s-t path (S and T have degree 1, so every
    # such path must traverse it).  Computed here so no caller has to know it.
    hop_offset = (int(net.augmented and source == net.aug_source)
                  + int(net.augmented and target == net.aug_target))
    tr.hop_offset = hop_offset
    tr.hop_frame = "augmented" if hop_offset else "original"

    if drive is not None:
        drive = np.asarray(drive, dtype=np.float32)
        if drive.shape != (N,):
            raise ValueError(f"drive has shape {drive.shape}, want ({N},)")

    irm_cap = None if irm_threshold is None else irm_threshold * net.n_vertices
    r = 0
    dwell_open = contain_edges is not None  # still inside the leading identified run
    dwell_w_open = contain_edges is not None and opt_weight is not None

    for t in range(steps):
        u = membrane(net, x)
        # S / T drive: periodically (Ch. 4) or held on every step (Ch. 5).
        # ``drive`` (default None) replaces the fixed D-on-the-S/T-mask injection with an
        # explicit per-neuron vector on the SAME schedule; see the lambda warning above.
        if p.st_mode == "always_on" or t % p.st_period == 0:
            if drive is None:
                u[net.st_drive] += p.D
            else:
                u += drive

        # inhibition subtracts a global term; local drive (S/T, beta) is intact.
        # gain_t follows the (optional) anneal schedule, else the constant gain.
        gain_t = anneal_gain(p, t, steps) if p.inhib_anneal != "none" else p.inhib_gain
        if meanfield:
            y_use = Y
            if p.inhib_scheme == "directed" and p.inhib_delay > 0:
                y_use = Yhist[-p.inhib_delay] if len(Yhist) >= p.inhib_delay else 0.0
            # suppress by the FRACTION active (Y/N), not the count -- a wbar~1/N_inhib
            # weight-scaling convention so the gain is intensive and N-invariant.
            u -= gain_t * y_use / N
        elif inhib is not None:
            # W_mi has -inhib_gain baked in; rescale to the scheduled gain_t.
            scale = 1.0 if p.inhib_gain == 0 else gain_t / p.inhib_gain
            u = u + scale * (inhib.W_mi @ x_i.astype(np.float32))

        x = sampler_fire(u, refrac, p.tau, rng)

        # advance the inhibitory pool / integrator from the new main activity
        if meanfield:
            A_tot = float(x.sum())
            Y += (A_tot - Y) / p.inhib_tau
            if p.inhib_scheme == "directed" and p.inhib_delay > 0:
                Yhist.append(Y)
        elif inhib is not None:
            u_i = (inhib.b_inhib + inhib.W_im @ x.astype(np.float32)
                   + inhib.W_ii @ x_i.astype(np.float32))
            x_i = sampler_fire(u_i, refrac_i, inhib.tau_inhib, rng)

        # matched-activity mass classification.  The active count is O(1); the
        # path/scatter/neither classification is O(n_neurons).  So the cheap
        # predicate is tested first and the expensive one runs only inside the
        # band -- which costs nothing, because the measurement conditions on
        # matched activity anyway.
        if mass_L is not None:
            n_act = int(x.sum())
            tr.active_hist[n_act] += 1
            tr.active_series[t] = n_act
            if band_lo <= n_act <= band_hi:
                tr.band_steps += 1
                if n_act == 2 * mass_L:
                    tr.matched_steps += 1
                klass, a_pairs, e_tot, a_wsum = mass_class(net, x, mass_L)
                tr.mass_series[t] = MASS_CODE[klass]
                if klass != "other":
                    tr.consistent_steps += 1
                    ai = int(round(a_pairs))
                    tr.alpha_hist[ai] = tr.alpha_hist.get(ai, 0) + 1
                    if klass == "path":
                        tr.mass_path += 1
                        tr.mass_e_path += e_tot
                        tr.mass_aw_path += a_wsum
                    elif klass == "scatter":
                        tr.mass_scatter += 1
                        tr.mass_e_scatter += e_tot
                        tr.mass_aw_scatter += a_wsum
                    elif klass == "closed":
                        tr.mass_closed += 1
                    else:
                        tr.mass_mixed += 1

        mask = induced_edges(net, x)
        nedges = int(mask.sum())

        # containment of a specific edge set (e.g. the optimal path) -- gives the
        # chance-baseline comparison (observed vs density^L * steps).  identification
        # is the strict form: induced graph IS EXACTLY that edge set (contains it AND
        # has no other edges) -- immune to both hot-lottery and cold-silence artifacts.
        is_contain = is_ident = False
        if contain_edges is not None and nedges and bool(mask[contain_edges].all()):
            is_contain = True
            tr.contain_count += 1
            if nedges == len(contain_edges):
                is_ident = True
                tr.ident_count += 1
        if dwell_open:  # leading run of identified steps -> dwell before first escape
            if is_ident:
                tr.dwell_exact += 1
            else:
                dwell_open, tr.escape_exact = False, t
        if record_indicators:
            tr.contain_series[t] = is_contain
            tr.ident_series[t] = is_ident

        # realised path shape (sampled): is the induced graph a simple s-t path, and
        # of what length/weight?  tests whether lambda>0 yields LONGER paths (Cor. 1).
        if track_realised and (t % track_realised == 0):
            si = tr.realised_sampled
            tr.realised_sampled += 1
            is_pshp = is_identw = False
            hops, wt = induced_path(net, mask, source, target)
            if hops is not None:
                is_pshp = True
                tr.path_shaped_count += 1
                tr.realised_hops[hops] = tr.realised_hops.get(hops, 0) + 1
                tr.realised_weight_sum += wt
                # identification (weight form): a simple s-t path of optimal weight,
                # ties allowed -- the algorithm's answer, not the specific edge set.
                if opt_weight is not None and abs(wt - opt_weight) < 1e-6:
                    is_identw = True
                    tr.ident_weight_count += 1
            if dwell_w_open:
                if is_identw:
                    tr.dwell_weight += 1
                else:
                    dwell_w_open, tr.escape_weight = False, si
            if record_indicators:
                tr.pshp_series[si] = is_pshp
                tr.identw_series[si] = is_identw

        # q = fraction of slots (segments) electing NO winner this step -- the silent-
        # slot occupancy the configuration-entropy model (ident ~ q^(S-2L)) turns on.
        if track_q:
            xd = x.astype(np.float64)
            A = np.bincount(net.seg, weights=xd, minlength=net.n_segments)
            tr.q_sum += float((A == 0).mean())
            Ac = np.bincount(net.cluster, weights=xd, minlength=net.n_vertices)
            tr.q_cluster_sum += float((Ac == 0).mean())
            if track_detail:
                # same two bincounts, kept per segment / per cluster instead of
                # collapsed to a scalar.  See Trace.seg_silent_steps.
                tr.seg_silent_steps += (A == 0)
                tr.cluster_active_sum += Ac.astype(np.int64)

        if irm_cap is not None and nedges > irm_cap:
            refrac[:] = 0
            x[:] = False
            tr.irm_times.append(t)

        if t % record_every == 0 and r < nrec:
            e_nc, e_spc = energy(net, x)
            tr.energy_nc[r] = e_nc
            tr.energy_spc[r] = e_spc
            tr.energy[r] = e_nc + e_spc
            tr.n_active[r] = int(x.sum())
            tr.induced_edges[r] = nedges
            r += 1

        if track_paths and nedges:
            path, cost = _path_in_induced(net, mask, source, target)
            if path is not None:
                if tr.first_path is None:
                    tr.first_path, tr.first_cost, tr.first_step = path, cost, t
                if cost < tr.best_cost:
                    tr.best_path, tr.best_cost = path, cost

        if on_step is not None:
            on_step(t, x, mask, tr)

    # Convert the raw hop histogram into the ORIGINAL graph's frame.  Asserted, not
    # assumed: an original path has at least one hop, so anything at or below the
    # offset means the offset is wrong and the run should fail rather than report.
    for h, f in tr.realised_hops.items():
        ho = int(h) - hop_offset
        assert ho >= 1, (f"realised hop count {h} with offset {hop_offset} implies "
                         f"{ho} original hops -- the hop frame is wrong")
        tr.realised_hops_orig[ho] = tr.realised_hops_orig.get(ho, 0) + int(f)

    return tr


def induced_path(net, mask, source, target):
    """If the induced subgraph IS exactly a simple s-t path, return (hops, weight).

    Returns (None, None) otherwise (branches, cycles, extra components, or not
    spanning s->t).  Distinct from _path_in_induced, which finds the cheapest path
    *within* a possibly-larger induced subgraph.
    """
    ends = net.edge_ends[mask]
    w = net.edge_w[mask]
    m = ends.shape[0]
    if m == 0:
        return None, None
    deg: dict[int, int] = {}
    adj: dict[int, list] = {}
    for (a, b_), ww in zip(ends, w):
        a, b_ = int(a), int(b_)
        deg[a] = deg.get(a, 0) + 1
        deg[b_] = deg.get(b_, 0) + 1
        adj.setdefault(a, []).append((b_, float(ww)))
        adj.setdefault(b_, []).append((a, float(ww)))
    # a simple s-t path: endpoints degree 1, all interior degree 2, and s/t are the
    # only degree-1 vertices.
    if deg.get(source, 0) != 1 or deg.get(target, 0) != 1:
        return None, None
    ones = [v for v, d in deg.items() if d == 1]
    if set(ones) != {source, target} or any(d not in (1, 2) for d in deg.values()):
        return None, None
    prev, cur, hops, wsum = None, source, 0, 0.0
    while cur != target:
        nxt = [(b_, ww) for (b_, ww) in adj[cur] if b_ != prev]
        if len(nxt) != 1:
            return None, None
        b_, ww = nxt[0]
        prev, cur = cur, b_
        hops += 1
        wsum += ww
        if hops > m:
            return None, None
    if hops != m:  # a disjoint cycle would leave edges unused
        return None, None
    return hops, wsum


def _path_in_induced(net, mask, source, target):
    """Dijkstra restricted to the induced subgraph (Ch. 4 sec. 4.3)."""
    import heapq

    ends = net.edge_ends[mask]
    w = net.edge_w[mask]
    if ends.size == 0:
        return None, np.inf
    adj: dict[int, list] = {}
    for (a, b_), ww in zip(ends, w):
        adj.setdefault(int(a), []).append((int(b_), float(ww)))
        adj.setdefault(int(b_), []).append((int(a), float(ww)))
    if source not in adj or target not in adj:
        return None, np.inf

    dist = {source: 0.0}
    prev: dict[int, int] = {}
    pq = [(0.0, source)]
    seen = set()
    while pq:
        d, v = heapq.heappop(pq)
        if v in seen:
            continue
        seen.add(v)
        if v == target:
            path = [v]
            while path[-1] != source:
                path.append(prev[path[-1]])
            return path[::-1], d
        for nb, ww in adj.get(v, ()):
            nd = d + ww
            if nd < dist.get(nb, np.inf):
                dist[nb] = nd
                prev[nb] = v
                heapq.heappush(pq, (nd, nb))
    return None, np.inf


# ----------------------------------------------------------------------
# Operating-point diagnostics (Ch. 5 tuning target).
# ----------------------------------------------------------------------
def ideal_state(net: Network, path: list[int]) -> np.ndarray:
    """The hand-built minimum-energy state for a given s->t path.

    Activates exactly the two neurons per path edge (the edge's endpoints),
    placing a vertex's two path edges in opposite WTA slots so they reinforce
    (alpha) rather than compete (gamma).  Induces exactly the path and nothing
    else; its energy is the target the sampler should approach.  2*L neurons
    active for an L-edge path.
    """
    x = np.zeros(net.n_neurons, dtype=bool)
    L = len(path) - 1
    for i, v in enumerate(path):
        if i > 0:  # edge to the previous vertex -> slot 0
            x[net.neuron_at(v, path[i - 1], 0)] = True
        if i < L:  # edge to the next vertex -> slot 1
            x[net.neuron_at(v, path[i + 1], 1)] = True
    return x


def ideal_energy(net: Network, path: list[int]) -> float:
    """Total energy E of the ideal-path state (the -6.49-style reference)."""
    e_nc, e_spc = energy(net, ideal_state(net, path))
    return float(e_nc + e_spc)


def state_counts(net: Network, x: np.ndarray) -> tuple[int, float, float, float]:
    """(active, alpha_pairs, beta_pairs, gamma_pairs) for a state.

    The pair counts that enter the energy -- used for verification (a path state
    should give 2L active, L beta, L-1 alpha) and as diagnostics.
    """
    xf = x.astype(np.float64)
    A = np.bincount(net.seg, weights=xf, minlength=net.n_segments)
    A_opp = A[net.opp_seg]
    A_own = A[net.seg]
    # A single-neuron auxiliary terminal points ``partner`` at itself; the
    # validity mask zeroes it so the gamma (ii) term does not become a
    # self-inhibition.  All-ones (hence an exact no-op) unaugmented.
    x_par = xf[net.partner] * net.partner_valid
    alpha_pairs = 0.5 * float(np.sum(xf * (A_opp - x_par)))
    gamma_pairs = (0.5 * float(np.sum(xf * (A_own - xf)))
                   + 0.5 * float(np.sum(xf * x_par)))
    beta_pairs = 0.0
    bi = net.beta_idx
    for f in range(bi.shape[1]):
        idx = bi[:, f]
        valid = idx >= 0
        beta_pairs += 0.5 * float(np.sum(xf[valid] * xf[idx[valid]]))
    return int(x.sum()), alpha_pairs, beta_pairs, gamma_pairs


def mass_class(net: Network, x: np.ndarray, L: int) -> tuple[str, float, float, float]:
    """Classify a state into the two classes the Ch. 4 discrimination condition compares.

    The condition ``c*(L-1)*alpha_bar > ln(#scatter/#path)`` sets the AGGREGATE
    Boltzmann mass on *chained* configurations against the mass on *equal-activity
    scatter* configurations -- a set of L pairwise-disjoint consistent edges.  Both
    classes carry the same active count (2L), the same beta-pair count (L) and no
    gamma conflicts; **only the alpha count separates them** (L-1 versus 0), which is
    why alpha is the lever at all.  So the classification is a
    pure function of the three pair counts -- no containment of, or reference to, the
    optimal path.  A chain elsewhere in the graph is a path-class state.

    Returns ``(klass, alpha_pairs, E, alpha_wsum)`` where ``klass`` is one of
    ``"path" | "scatter" | "mixed" | "closed" | "other"`` and ``E`` is the total
    energy E_NC + E_SPC of the state.  ``"other"`` means the state is not on the
    matched shell at all (wrong active count, an unpaired active neuron, or a gamma
    conflict) and its alpha count is not comparable.

    O(n_neurons), one pass -- this is the expensive predicate the active-count
    pre-filter exists to avoid paying on every step.
    """
    p = net.params
    xf = x.astype(np.float64)
    n_act = int(xf.sum())

    A = np.bincount(net.seg, weights=xf, minlength=net.n_segments)
    Sw = np.bincount(net.seg, weights=xf * net.w, minlength=net.n_segments)
    A_own = A[net.seg]
    A_opp = A[net.opp_seg]
    Sw_opp = Sw[net.opp_seg]
    # A single-neuron auxiliary terminal points ``partner`` at itself; the
    # validity mask zeroes it so the gamma (ii) term does not become a
    # self-inhibition.  All-ones (hence an exact no-op) unaugmented.
    x_par = xf[net.partner] * net.partner_valid

    alpha_pairs = 0.5 * float(np.sum(xf * (A_opp - x_par)))
    alpha_wsum = 0.5 * float(np.sum(xf * ((net.w * (A_opp - x_par))
                                          + (Sw_opp - net.w * x_par))))
    gamma_pairs = (0.5 * float(np.sum(xf * (A_own - xf)))
                   + 0.5 * float(np.sum(xf * x_par)))
    beta_pairs = 0.0
    bi = net.beta_idx
    for f in range(bi.shape[1]):
        idx = bi[:, f]
        valid = idx >= 0
        beta_pairs += 0.5 * float(np.sum(xf[valid] * xf[idx[valid]]))

    lin = float(np.sum(net.b * xf))
    e_nc = -(lin + beta_pairs * p.D + gamma_pairs * p.I) - p.alpha_scale * alpha_pairs
    e_spc = p.alpha_scale * alpha_wsum / p.B
    E = e_nc + e_spc

    a = int(round(alpha_pairs))
    consistent = (n_act == 2 * L
                  and abs(beta_pairs - L) < 1e-9
                  and gamma_pairs < 1e-9)
    if not consistent:
        klass = "other"
    elif a == L - 1:
        klass = "path"
    elif a == 0:
        klass = "scatter"
    elif a >= L:
        klass = "closed"
    else:
        klass = "mixed"
    return klass, alpha_pairs, E, alpha_wsum


def path_edge_indices(net: Network, path: list[int]) -> np.ndarray:
    """Indices into net.edge_ends of each edge on a vertex path (for containment)."""
    lookup = {frozenset((int(a), int(b))): i
              for i, (a, b) in enumerate(net.edge_ends)}
    return np.array([lookup[frozenset((path[i], path[i + 1]))]
                     for i in range(len(path) - 1)], dtype=int)


def trace_summary(tr: Trace, net: Network, ideal_e: float | None = None) -> dict:
    """Standing-rule operating-point columns for a run.

    Every success-metric table must also carry these, so a metric that improves
    only because the induced graph got denser is not mistaken for progress.
    """
    out = {
        "mean_induced_density": float(tr.induced_edges.mean()) / max(net.n_edges, 1),
        "mean_energy": float(tr.energy.mean()),
        "mean_active": float(tr.n_active.mean()),
        "irm_firings": len(tr.irm_times),
    }
    if ideal_e is not None:
        out["ideal_energy"] = ideal_e
        out["energy_gap"] = out["mean_energy"] - ideal_e
    return out
