"""
Compile a weighted graph into the stochastic spiking network of
Engler (2018), Ch. 4, in a form whose per-timestep cost is O(|E|)
rather than O(N d^2).

Neuron layout
-------------
Each vertex v_i of G becomes a WTA *cluster* of 2 * deg(i) neurons:
two identical WTA components, each holding one neuron per neighbour
of v_i.  Neuron n[i, c, s] represents "vertex v_i, component c,
adjacency to neighbour adj[i][s]".

Neurons are flattened so that a cluster's two components are two
contiguous *segments*.  Segment id = 2*i + c, so there are 2*|V|
segments and all intra-cluster interactions reduce to segment sums.

The ONE exception is the pair of auxiliary terminals S and T added by
``compile_network(augment_st=True)``.  A WTA cluster encodes a
*traversal* -- its two components are the two edges a path uses to pass
THROUGH that vertex -- and a path does not pass through its own source,
it starts there.  So S and T are SINGLE NEURONS, not clusters: they get
one component, hence one neuron each, and their segment 2*i+1 stays
empty.  Consequently they carry no alpha, no gamma (i) and no gamma (ii)
coupling of any kind, because every one of those classes is a relation
between two neurons of the SAME cluster and there is only one.  See
``compile_network``.

Edge classes (Ch. 4, Eq. 4.6-4.8)
---------------------------------
alpha : excitatory, opposite components of one cluster, different index.
        weight f(e1,e2) = 1 - (w_j + w_k)/B          -- SEPARABLE
beta  : excitatory, weight D, between clusters i and j for each
        graph edge (i,j).  Exactly |E| * 2 ordered pairs per component.
gamma : inhibitory, weight I.  Two sub-cases (Ch. 4 sec. 4.2.1 M(x)):
        (i)  all pairs within one WTA component  -> I * (A_s - x_k)
        (ii) same index, opposite components     -> I * x_partner

Because f is separable, alpha never needs to be stored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


class _Unset:
    """Sentinel for a ``b_st`` the caller did not name (option 3).

    Three-valued distinct from both ``None`` and a float:

    * ``UNSET`` -- the caller did not choose a terminal bias.  This is the
      default, and ``compile_network(augment_st=True)`` REFUSES to build while
      ``b_st`` is still UNSET, so an augmented network can never silently adopt a
      terminal-bias convention.
    * ``None``  -- an explicit request to inherit ``bias`` (b_S = b_T = b_k).
    * ``float`` -- an explicit terminal bias (``0.0`` is the historical anchor).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _Unset()


@dataclass
class Params:
    """Network parameters.  See Ch. 4 sec. 4.2.1."""

    D: float = 4.0  # beta (inter-cluster) excitatory strength
    I: float = -40.0  # gamma inhibitory strength; |I| >> |D|
    B: float | None = None  # alpha normaliser; default 2*max edge weight
    bias: float | None = None  # b_k; default -(D/4 + 1)  (Eq. 4.15)
    alpha_scale: float = 1.0  # factor c on the alpha weight (Ch.4 Eq. 4.8);
    # c=1 is the published construction.  Scaling alpha widens the path-vs-scatter
    # energy margin (only alpha distinguishes a chained path from a disjoint set of
    # equal-cost consistent edges).  Interacts with b_k via the length-bias
    # lambda = 2*b_k + D + c; set b_k = (lambda - D - c)/2 to fix lambda.
    tau: int = 10  # PSP length == refractory period, in steps
    dt: float = 1.0  # ms per step
    st_period: int = 10  # S and T neurons fire every this many steps
    beta_fanout: int = 2  # 2 = canonical (both slots); 1 = slot-matched, non-canonical

    # --- S / T drive mode (Ch. 4 vs Ch. 5 discrepancy) -------------------
    # "periodic"  : S and T fire every st_period steps (Ch. 4 step 4).
    # "always_on" : S and T held active every step (Ch. 5 sec. 5.2).
    st_mode: str = "periodic"

    # --- the AUXILIARY-TERMINAL bias b_S = b_T (stage 5 part 2 -> library) -----
    # Applies ONLY under ``compile_network(augment_st=True)``, where the two
    # terminals S and T are single neurons of their own clusters and so have a
    # bias that is separable from every original cluster's b_k.
    #
    # Three-valued (option 3) -- see the ``UNSET`` sentinel above:
    # ``UNSET`` (THE DEFAULT) -- NOT named.  ``compile_network(augment_st=True)``
    #          REFUSES to build on this: an augmented network must state which
    #          terminal-bias convention it wants rather than inherit a silent
    #          default.  Inert on an unaugmented build (b_st is never read there).
    # ``0.0``  -- the terminals are ANCHORED.  Combined with the always-on S/T
    #          drive (D = 4) this leaves u = +4 at each terminal rather than
    #          u = 0: the drive does NOT switch off when the bias changes.
    #          Measured worth at N=60, q=0.900: x5.26 identification and x5.76
    #          nucleation over the inherited value.
    # ``None`` -- INHERIT ``bias``, i.e. b_S = b_T = b_k.  This reproduces the
    #          earlier condition where the terminal bias equalled b_k; kept
    #          reachable and NAMED so choosing it is a parameter, not a checkout.
    # Resolved in ``resolved()`` (None -> bias); ``compile_network`` refuses an
    # unnamed augmented build, then writes b_st and ASSERTS it, and STAMPS the
    # resolved value + whether it was named onto the returned ``Network``.
    b_st: float | None = UNSET

    # --- activity-dependent inhibition (Ch. 5) ---------------------------
    # Global suppression that rises with total activity while drive stays
    # local, so beta-supported (on-path) neurons survive as inhibition
    # climbs.  Two implementations, selectable, plus two coupling schemes.
    #   inhib_kind   : "off" | "meanfield" | "subnetwork"
    #   inhib_scheme : "reciprocal" (symmetric -> smooth cooling)
    #                  "directed"   (lagged/asymmetric -> oscillatory reset)
    #                  "mixed"      (subnetwork only) -- reciprocal and directed
    #                               are the two ENDPOINTS of one axis, so this
    #                               interpolates the inhibitory->main limb between
    #                               them by ``inhib_asym`` in [0, 1].  It exists so
    #                               coupling asymmetry can be SWEPT rather than
    #                               toggled.
    #   inhib_locality : "global"  (the dissertation's construction -- ONE unstructured
    #                               pool wired at random across the WHOLE main network)
    #                    "cluster" (*** A DELIBERATE DEPARTURE ***)
    #                              -- the same spiking pool, PARTITIONED BY VERTEX CLUSTER:
    #                              each cluster gets its own inhibitory cells, driven only
    #                              by that cluster's own occupancy and projecting only back
    #                              into it.  The connectivity becomes block diagonal; every
    #                              other element of the construction is untouched.
    #                              This is NOT in Ch. 5 and must never be reported as the
    #                              dissertation's mechanism -- it is a probe of whether the
    #                              shallow basin is a property of HOW inhibition is
    #                              DISTRIBUTED, which every mechanism tested so far (global
    #                              mean-field, exact global pool, imposed cyclic gain,
    #                              directed oscillation) leaves untouched because each of
    #                              them shifts every neuron by the SAME term.
    inhib_kind: str = "off"
    inhib_scheme: str = "reciprocal"
    inhib_locality: str = "global"
    inhib_gain: float = 0.0  # coupling strength g (the final gain when annealing)
    inhib_tau: int = 40  # inhibitory time constant (should exceed tau)
    inhib_delay: int = 0  # feedback lag for "directed" meanfield
    # gain annealing: ramp the effective inhibition permissive -> strict over the
    # run so the path nucleates first (low g) and off-path leakage is suppressed
    # later (high g).  Applies to both meanfield and subnetwork kinds.
    #   "none"        : constant gain = inhib_gain
    #   "linear"      : inhib_gain_start -> inhib_gain, linearly
    #   "exponential" : inhib_gain_start -> inhib_gain, fast approach
    #   "cyclic"      : sawtooth reheat between the two, period inhib_cycle
    inhib_anneal: str = "none"
    inhib_gain_start: float = 0.0  # gain at t=0 for annealed schedules
    inhib_cycle: int = 0  # "cyclic" period in steps (0 -> whole run)
    # subnetwork-only:
    inhib_fraction: float = 0.2  # inhibitory share of the WHOLE network
    inhib_p_ei: float = 0.1  # main -> inhibitory connection probability
    inhib_p_ie: float = 0.1  # inhibitory -> {main, inhibitory} probability
    inhib_excitability: float = -2.0  # bias b_k of inhibitory cells (< main)
    inhib_seed: int = 0  # RNG seed for the random inhibitory wiring
    # inhib_scheme == "mixed" only: fraction of the inhibitory->main synapses drawn
    # INDEPENDENTLY of the main->inhibitory pattern rather than reused from its
    # transpose.  0 -> exactly the reciprocal construction; 1 -> a fully
    # non-reciprocal feedback limb.  Density-preserving when p_ie == p_ei, so this
    # varies the ASYMMETRY of the loop without also varying its total strength.
    inhib_asym: float = 1.0
    # inhib_locality == "cluster" only: connection probability WITHIN a cluster's block.
    # Defaults to 1.0 (dense within block) rather than to ``inhib_p_ei``, and the reason is
    # structural rather than a preference: a cluster holds only 2*deg(i) main neurons and
    # gets ~0.25*2*deg(i) inhibitory cells, so at p=0.1 the median block would contain ONE
    # or ZERO synapses and most clusters would have no working local loop at all.  Locality
    # is the manipulation under test; within-block sparsity is not, so it is set out of the
    # way.  A consequence worth stating: at p=1.0 the local wiring is DETERMINISTIC, while
    # the global pool's is a random draw per graph.
    inhib_local_p: float = 1.0

    def resolved(self, wmax: float) -> "Params":
        p = Params(**vars(self))
        if p.B is None:
            p.B = 2.0 * wmax
        if p.bias is None:
            p.bias = -(p.D / 4.0 + 1.0)
        if p.b_st is None:            # "inherit": the pre-stage-5-part-2 behaviour
            p.b_st = p.bias
        return p


@dataclass
class InhibitorySubnet:
    """Explicit Ch. 5 inhibitory subnetwork (small-network validation target).

    A pool of inhibitory cells (lower excitability, larger tau) that integrate
    main-network activity and feed suppression back.  Stored as sparse coupling
    matrices; its mean-field limit is the O(1) ``inhib_kind="meanfield"`` path.
    """

    n_inhib: int
    b_inhib: np.ndarray  # (n_inhib,) bias (excitability), < main bias
    tau_inhib: int  # inhibitory PSP/refractory length (> main tau)
    W_mi: sp.csr_matrix  # (n_main, n_inhib) inhibitory -> main, weights < 0
    W_im: sp.csr_matrix  # (n_inhib, n_main) main -> inhibitory, weights > 0
    W_ii: sp.csr_matrix  # (n_inhib, n_inhib) inhibitory -> inhibitory, <= 0
    locality: str = "global"  # "global" | "cluster"  (see Params.inhib_locality)
    cells_per_cluster: np.ndarray | None = None  # (n_vertices,) when locality=="cluster"


@dataclass
class Network:
    """Segment-structured stochastic spiking network."""

    n_vertices: int
    n_neurons: int  # excludes the S and T driver neurons (unaugmented build)
    n_segments: int

    seg: np.ndarray  # (n_neurons,) int32  segment id = 2*cluster + comp
    cluster: np.ndarray  # (n_neurons,) int32  vertex this cluster represents
    comp: np.ndarray  # (n_neurons,) int8    WTA component, 0 or 1
    nbr: np.ndarray  # (n_neurons,) int32  neighbour vertex represented
    w: np.ndarray  # (n_neurons,) float32 w(v_cluster, v_nbr)
    partner: np.ndarray  # (n_neurons,) int32  same index, opposite component
    beta_idx: np.ndarray  # (n_neurons, F) int32  beta partners, -1 = none
    opp_seg: np.ndarray  # (n_neurons,) int32  segment of opposite component
    # 1.0 normally; 0.0 for a neuron that HAS no opposite-component twin (the
    # single-neuron auxiliary terminals S and T, which have one component).  Such
    # a neuron's ``partner`` entry points at ITSELF, so every consumer must read
    # the partner occupancy as ``x[partner] * partner_valid`` -- otherwise the
    # gamma (ii) term I*x_partner becomes a SELF-inhibition of strength I, which
    # is exactly the artefact the single-neuron representation exists to remove.
    # All-ones on an unaugmented network, so the multiply is a numerical no-op
    # there (float 1.0 is exact).
    partner_valid: np.ndarray  # (n_neurons,) float32
    # (n_vertices,) int8 -- WTA components this vertex's cluster actually has:
    # 2 for a real graph vertex, 1 for a single-neuron auxiliary terminal.
    n_comps: np.ndarray

    b: np.ndarray  # (n_neurons,) float32 bias, incl. S/T drive
    st_drive: np.ndarray  # (n_neurons,) bool  receives S or T input

    # readout: for each graph edge, the two neurons that must both be
    # active for the edge to appear in the induced graph (Def. 2)
    edge_u: np.ndarray  # (n_edges, 2) int32  both slots, cluster u
    edge_v: np.ndarray  # (n_edges, 2) int32  both slots, cluster v
    edge_w: np.ndarray  # (n_edges,) float32
    edge_ends: np.ndarray  # (n_edges, 2) int32 vertex endpoints

    params: Params

    inhib: InhibitorySubnet | None = None  # set when inhib_kind == "subnetwork"

    # --- S / T augmentation (Ch. 4 step 1); see compile_network(augment_st=) ----
    augmented: bool = False
    n_vertices_orig: int = -1  # |V(G)| before augmentation (== n_vertices if not)
    source: int = -1  # the ORIGINAL source vertex, always a vertex of G
    target: int = -1  # the ORIGINAL target vertex, always a vertex of G
    aug_source: int = -1  # the new vertex S, or -1 when not augmented
    aug_target: int = -1  # the new vertex T, or -1 when not augmented

    # --- b_st provenance stamp (option 4) --------------------------------------
    # What terminal bias this network was actually built with, and whether the
    # caller chose it.  Only meaningful under augmentation; on an unaugmented
    # build b_st is never applied, so these stay (b_st_named as given, resolved
    # None).  A result that references a Network can thus show the terminal-bias
    # convention in force rather than leave it to a git archaeology.
    b_st_named: bool = False  # did the caller name b_st explicitly?
    b_st_resolved: float | None = None  # the b_st actually applied (augmented only)

    @property
    def n_edges(self) -> int:
        return int(self.edge_u.shape[0])

    def augmented_path(self, path: list[int]) -> list[int]:
        """Extend an s->t path of G to the S->T path of the augmented graph.

        On an unaugmented network this is the identity, so callers can write
        ``ideal_state(net, net.augmented_path(path))`` unconditionally.
        """
        if not self.augmented:
            return list(path)
        if int(path[0]) != self.source or int(path[-1]) != self.target:
            raise ValueError(
                f"path {path[0]}->{path[-1]} does not run source->target "
                f"({self.source}->{self.target})")
        return [self.aug_source] + list(path) + [self.aug_target]

    def is_single_neuron(self, vertex: int) -> bool:
        """True if this vertex is a single-neuron auxiliary terminal (S or T)."""
        return int(self.n_comps[int(vertex)]) == 1

    def neuron_at(self, cluster: int, nbr: int, comp: int) -> int:
        """Index of the neuron (cluster, neighbour, component). O(deg) scan.

        A single-neuron auxiliary terminal has no WTA slots to choose between,
        so any ``comp`` request there resolves to its one neuron.
        """
        if self.is_single_neuron(cluster):
            comp = 0
        m = (self.cluster == cluster) & (self.nbr == nbr) & (self.comp == comp)
        idx = np.nonzero(m)[0]
        if idx.size != 1:
            raise KeyError(f"no unique neuron for ({cluster}, {nbr}, {comp})")
        return int(idx[0])


def compile_network(
    n: int,
    edges: np.ndarray,
    weights: np.ndarray,
    source: int,
    target: int,
    params: Params | None = None,
    augment_st: bool = False,
) -> Network:
    """Build the Ch. 4 network from a weighted simple graph.

    Parameters
    ----------
    n        : number of vertices
    edges    : (m, 2) int array of undirected edges, u < v
    weights  : (m,) positive edge weights
    source   : source vertex (receives the S driver neuron)
    target   : target vertex (receives the T driver neuron)
    augment_st : build the dissertation's AUGMENTED graph (Ch. 4 step 1) --
        two NEW vertices S and T joined to ``source`` and ``target`` by
        ZERO-WEIGHT edges, after which the original source/target are ordinary
        interior clusters with both WTA slots occupied on a path state.  This
        is what makes E_SPC a function of TOTAL PATH WEIGHT alone; without it
        the terminal edges are charged once and the interior edges twice, and
        the energy ordering over the path sector can invert.  ***Default False
        -- the historical, unaugmented build.***

        *** S AND T ARE GENUINE SINGLE NEURONS. ***  An earlier
        build compiled them as degree-1 CLUSTERS of two neurons, on the argument
        that a degree-1 cluster can never form an alpha pair.  That argument is
        correct as far as it goes, but two neurons in a degree-1 cluster share
        ONE adjacency index across OPPOSITE slots -- the gamma (ii) pattern --
        so they carried an inhibitory coupling of strength I between them, and
        |I| >> D.  Measured on a compiled network: W[32,33] = -40.000000 = I
        exactly, and turning the second S neuron on cost +40.000000 in energy.
        A true single neuron has no such coupling, and the penalty acted
        directly on S's ability to stay active and anchor the path -- the very
        thing the augmentation exists to provide.  It was inert for the counts
        and energies measured so far (``ideal_state`` activates exactly one of
        the two, and nothing had sampled under the flag) and LIVE for every
        sampling run.

        So an auxiliary terminal is allocated ONE neuron (``n_comps == 1``),
        joined by BETA of strength D to the S-index neuron in EACH component of
        the source cluster -- one per slot, per the author's description.  The
        invariant, asserted in the compiler: NO coupling of any class exists
        between two neurons of the same auxiliary terminal, because there is
        only one such neuron.

        The counts are untouched by this: the aux terminals contributed no alpha
        pair under either reading, ``ideal_state`` activated exactly one of the
        two old neurons and activates the one new one, and the beta pair at each
        terminal is the same synapse.  So |alpha| = L+1, beta = L+2,
        active-in-original-clusters = 2(L+1) and total active = 2(L+2) all hold
        unchanged, as does lambda = 2*b_k + D + c and the length-neutral b_k;
        only the affine constant moves (-lambda*L + c -> -lambda*L + c -
        2*lambda, i.e. the bias acts on the AUGMENTED length L+2).  At
        lambda = 0 -- the operating point -- the two networks give the SAME E_NC.

        The S/T periodic driver attaches to the S NEURON and the T NEURON, per
        Ch. 4 step (4) ("the single neurons representing the original nodes S
        and T are activated"); their drive reaches the original source and
        target through the beta synapses above.  That is 2 marked neurons,
        against the unaugmented build's 2*deg(s) + 2*deg(t).  Nothing that
        samples has been run under this flag yet.
    """
    edges = np.asarray(edges, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float32)
    n_vertices_orig = int(n)
    source_orig, target_orig = int(source), int(target)
    aug_source = aug_target = -1

    # Options 3+4: capture whether b_st was NAMED before ``resolved()``
    # folds an explicit None into b_k (so the provenance stamp can report it),
    # and REFUSE to build an augmented network on an unnamed b_st -- the historic
    # silent default was 0.0 (anchored) while other series used None (inherit
    # b_k), so an augmented compile must choose rather than inherit a default.
    _params_in = params or Params()
    b_st_named = _params_in.b_st is not UNSET
    if augment_st and not b_st_named:
        raise ValueError(
            "compile_network(augment_st=True) requires b_st to be named "
            "explicitly. Pass Params(b_st=0.0) for anchored terminals "
            "(the historical default) or Params(b_st=None) to inherit b_k.")

    if augment_st:
        if source_orig == target_orig:
            raise ValueError("augment_st needs distinct source and target")
        wmax_before = float(weights.max())
        aug_source, aug_target = n, n + 1
        edges = np.vstack([
            edges,
            np.array([[source_orig, aug_source], [target_orig, aug_target]],
                     dtype=np.int32)]).astype(np.int32)
        weights = np.concatenate(
            [weights, np.zeros(2, dtype=np.float32)]).astype(np.float32)
        n = n + 2
        # B is resolved from max edge weight, so the two zero-weight edges must
        # leave it untouched -- otherwise the augmented and unaugmented networks
        # are on different energy scales and nothing below is comparable.
        # ASSERTED, not assumed (brief, Part A).
        if float(weights.max()) != wmax_before:
            raise AssertionError(
                f"augmentation moved wmax {wmax_before} -> {float(weights.max())}")

    m = edges.shape[0]
    params = _params_in.resolved(float(weights.max()))

    # --- adjacency, sorted, with edge weight alongside ------------------
    deg = np.zeros(n, dtype=np.int32)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    if deg.min() == 0:
        raise ValueError("graph has isolated vertices")

    # CSR-style adjacency
    adj_off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=adj_off[1:])
    fill = adj_off[:-1].copy()
    adj_nbr = np.empty(adj_off[-1], dtype=np.int32)
    adj_w = np.empty(adj_off[-1], dtype=np.float32)
    adj_eid = np.empty(adj_off[-1], dtype=np.int32)
    for a, b_, e in ((edges[:, 0], edges[:, 1], np.arange(m)),
                     (edges[:, 1], edges[:, 0], np.arange(m))):
        for u, v, ei in zip(a, b_, e):
            k = fill[u]
            adj_nbr[k] = v
            adj_w[k] = weights[ei]
            adj_eid[k] = ei
            fill[u] = k + 1

    # sort each adjacency block by neighbour id so lookups are bisectable
    for i in range(n):
        lo, hi = adj_off[i], adj_off[i + 1]
        order = np.argsort(adj_nbr[lo:hi], kind="stable")
        adj_nbr[lo:hi] = adj_nbr[lo:hi][order]
        adj_w[lo:hi] = adj_w[lo:hi][order]
        adj_eid[lo:hi] = adj_eid[lo:hi][order]

    # --- neuron layout --------------------------------------------------
    # cluster i occupies [clus_off[i], clus_off[i+1]) and holds
    # n_comps[i] * deg[i] neurons: component 0 first (deg_i of them), then
    # component 1.  n_comps[i] is 2 for every real graph vertex and 1 for a
    # single-neuron auxiliary terminal, whose component-1 segment (2*i+1) is
    # then simply EMPTY -- bincount over a gap-free segment id range handles
    # that with no special case, and A_opp/Sw_opp read 0 there, which is
    # exactly right: a single neuron has no opposite-slot partner to couple to.
    n_comps = np.full(n, 2, dtype=np.int8)
    if augment_st:
        n_comps[aug_source] = 1
        n_comps[aug_target] = 1
        if int(deg[aug_source]) != 1 or int(deg[aug_target]) != 1:
            raise AssertionError(
                f"auxiliary terminal degree must be 1, got "
                f"{int(deg[aug_source])} / {int(deg[aug_target])}")

    per_cluster = n_comps.astype(np.int64) * deg.astype(np.int64)
    clus_off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(per_cluster, out=clus_off[1:])
    n_neurons = int(clus_off[-1])

    cluster = np.empty(n_neurons, dtype=np.int32)
    comp = np.empty(n_neurons, dtype=np.int8)
    nbr = np.empty(n_neurons, dtype=np.int32)
    w = np.empty(n_neurons, dtype=np.float32)
    partner = np.empty(n_neurons, dtype=np.int32)
    # A single-neuron terminal has no twin; ``partner`` points at itself and the
    # validity flag zeroes it out at every consumer (see Network.partner_valid).
    partner_valid = np.ones(n_neurons, dtype=np.float32)

    for i in range(n):
        d = int(deg[i])
        lo = int(clus_off[i])
        nc = int(n_comps[i])
        alo, ahi = int(adj_off[i]), int(adj_off[i + 1])
        for c in range(nc):
            s0 = lo + c * d
            sl = slice(s0, s0 + d)
            cluster[sl] = i
            comp[sl] = c
            nbr[sl] = adj_nbr[alo:ahi]
            w[sl] = adj_w[alo:ahi]
            if nc == 2:
                other = lo + (1 - c) * d
                partner[sl] = np.arange(other, other + d, dtype=np.int32)
            else:
                partner[sl] = np.arange(s0, s0 + d, dtype=np.int32)  # self
                partner_valid[sl] = 0.0

    seg = (2 * cluster + comp).astype(np.int32)
    opp_seg = (2 * cluster + (1 - comp)).astype(np.int32)

    # --- beta partners --------------------------------------------------
    # neuron (i, c, ->j) pairs with neurons (j, c', ->i).
    # beta_fanout=2 : c' in {0,1};  beta_fanout=1 : directed comp1 -> comp0.
    F = params.beta_fanout
    beta_idx = np.full((n_neurons, F), -1, dtype=np.int32)

    def locate(i: int, j: int, c: int) -> int:
        """index of neuron (cluster i, component c, neighbour j).

        A single-neuron auxiliary terminal has only component 0, so any slot
        request there resolves to its one neuron.
        """
        if int(n_comps[i]) == 1:
            c = 0
        alo, ahi = int(adj_off[i]), int(adj_off[i + 1])
        pos = int(np.searchsorted(adj_nbr[alo:ahi], j))
        return int(clus_off[i]) + c * int(deg[i]) + pos

    for k in range(n_neurons):
        i, j, c = int(cluster[k]), int(nbr[k]), int(comp[k])
        # Gather over the components cluster j ACTUALLY HAS.  When j is a
        # single-neuron terminal that is one entry, not two: the second slot
        # stays -1 rather than duplicating the same synapse, which would double
        # both the beta drive in ``membrane`` and the beta pair count in
        # ``energy``.  The relation stays symmetric -- the terminal's own neuron
        # gathers BOTH slots of the S-index neuron opposite it, and each of
        # those gathers the terminal once -- so 2 ordered entries per side and
        # the 0.5 in the pair count is still the ordered->unordered conversion.
        j_comps = int(n_comps[j])
        if F == 2:
            beta_idx[k, 0] = locate(j, i, 0)
            beta_idx[k, 1] = locate(j, i, 1) if j_comps == 2 else -1
        else:  # directed: comp 1 (outgoing) -> comp 0 (incoming)
            want = 0 if c == 1 else 1
            beta_idx[k, 0] = locate(j, i, want if j_comps == 2 else 0)

    # --- bias, and the S / T driver neurons ------------------------------
    b = np.full(n_neurons, params.bias, dtype=np.float32)

    # --- the auxiliary-terminal bias b_S = b_T = params.b_st -------------
    # Only meaningful under augment_st: without it there is no S neuron to
    # carry a bias of its own, and the source/target clusters are ordinary
    # clusters whose b_k must not be special-cased.  Promoted out of stage 5's
    # harness (its module constant ``B_ST`` / ``apply_b_st``) so that no
    # harness has to remember to apply it.  ``b_st is None`` was resolved to
    # ``bias`` above, which reproduces the pre-stage-5-part-2 condition
    # EXACTLY: the write below is then a no-op by value.
    if augment_st:
        aux_idx = np.array([int(clus_off[aug_source]), int(clus_off[aug_target])],
                           dtype=np.int64)
        for v, k in zip((aug_source, aug_target), aux_idx):
            if int(clus_off[v + 1]) - int(clus_off[v]) != 1:
                raise AssertionError(
                    f"auxiliary terminal {v} spans "
                    f"{int(clus_off[v + 1]) - int(clus_off[v])} neurons, expected 1")
        b[aux_idx] = params.b_st
        # ASSERT, do not trust: b is exactly params.bias on every neuron of every
        # ORIGINAL cluster and exactly params.b_st at the two terminal indices.
        orig_mask = np.ones(n_neurons, dtype=bool)
        orig_mask[aux_idx] = False
        if not np.all(b[orig_mask] == np.float32(params.bias)):
            raise AssertionError(
                "bias is not exactly params.bias on every original-cluster neuron")
        if not np.all(b[aux_idx] == np.float32(params.b_st)):
            raise AssertionError(
                f"b at the auxiliary terminals is {b[aux_idx]}, want {params.b_st}")

    st_drive = np.zeros(n_neurons, dtype=bool)
    # Ch. 4 step (4): "the single neurons representing the original nodes S and
    # T are activated, representing that they fire regularly every tau ms".
    # Under augment_st those single neurons EXIST, so the driver attaches to
    # exactly them -- 2 neurons -- and reaches the original source and target
    # over the beta synapses built above.  Without the flag there is no S neuron
    # to attach to, so the historical behaviour stands: mark every neuron of the
    # source and target clusters (2*deg(s) + 2*deg(t)), which implements S's
    # EFFECT without allocating S.
    st_vertices = ((aug_source, aug_target) if augment_st
                   else (source_orig, target_orig))
    for v in st_vertices:
        st_drive[int(clus_off[v]): int(clus_off[v + 1])] = True

    # --- induced-graph readout (Def. 2) ----------------------------------
    # Edge (u,v) present iff SOME neuron in cluster u represents v and is
    # active, AND some neuron in cluster v represents u and is active.
    # Def. 2 writes n^u_v with no component superscript: the readout does
    # not care which WTA slot the neuron sits in, because the slot
    # assignment is arbitrary on an undirected graph.
    edge_u = np.array([[locate(int(u), int(v), 0), locate(int(u), int(v), 1)]
                       for u, v in edges], np.int32)
    edge_v = np.array([[locate(int(v), int(u), 0), locate(int(v), int(u), 1)]
                       for u, v in edges], np.int32)

    if augment_st:
        # THE INVARIANT: no coupling of any class may exist between
        # two neurons of the same auxiliary terminal, because there is only one
        # such neuron.  Asserted structurally rather than trusted -- alpha,
        # gamma (i) and gamma (ii) are all relations WITHIN a cluster, so a
        # one-neuron cluster forecloses every one of them, and beta is a relation
        # BETWEEN clusters, so it cannot land here either.
        for v in (aug_source, aug_target):
            idx = np.nonzero(cluster == v)[0]
            if idx.size != 1:
                raise AssertionError(
                    f"auxiliary terminal {v} got {idx.size} neurons, expected 1")
            k = int(idx[0])
            if partner_valid[k] != 0.0 or int(partner[k]) != k:
                raise AssertionError(
                    f"auxiliary terminal neuron {k} has a live partner "
                    f"({int(partner[k])}, valid {float(partner_valid[k])})")
            if np.any(beta_idx[k] == k) or np.any(beta_idx == k, axis=1).sum() != 2:
                raise AssertionError(
                    f"auxiliary terminal neuron {k} beta wiring is not the two "
                    f"slots of its one graph neighbour")

    inhib = None
    if params.inhib_kind == "subnetwork":
        inhib = build_inhibitory(n_neurons, params, cluster=cluster)

    return Network(
        n_vertices=n,
        n_neurons=n_neurons,
        n_segments=2 * n,
        seg=seg,
        cluster=cluster,
        comp=comp,
        nbr=nbr,
        w=w,
        partner=partner,
        beta_idx=beta_idx,
        opp_seg=opp_seg,
        partner_valid=partner_valid,
        n_comps=n_comps,
        b=b,
        st_drive=st_drive,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=weights.copy(),
        edge_ends=edges.copy(),
        params=params,
        inhib=inhib,
        b_st_named=b_st_named,
        b_st_resolved=(float(params.b_st) if augment_st else None),
        augmented=bool(augment_st),
        n_vertices_orig=n_vertices_orig,
        source=source_orig,
        target=target_orig,
        aug_source=aug_source,
        aug_target=aug_target,
    )


def build_inhibitory(n_main: int, params: Params,
                     cluster: np.ndarray | None = None) -> InhibitorySubnet:
    """Construct the explicit Ch. 5 inhibitory subnetwork (validation scale).

    ``inhib_fraction`` is the inhibitory share of the *whole* network, so
    n_inhib = f/(1-f) * n_main.  Excitatory main->inhibitory drive has weight
    +1; inhibitory feedback has weight -inhib_gain.  The dense random wiring
    is O(n_main * n_inhib), so this is a small-network target -- at scale use
    the O(1) mean-field limit (inhib_kind="meanfield").

    ``cluster`` (the per-neuron vertex id) is required only for
    ``inhib_locality == "cluster"``, which builds the block-diagonal variant.
    """
    if params.inhib_locality == "cluster":
        if cluster is None:
            raise ValueError("inhib_locality='cluster' needs the cluster assignment")
        return build_cluster_local_inhibitory(n_main, params, cluster)
    if params.inhib_locality != "global":
        raise ValueError(f"unknown inhib_locality {params.inhib_locality!r}")
    if n_main > 20000:
        raise MemoryError(
            f"exact inhibitory subnetwork refuses n_main={n_main}; "
            "use inhib_kind='meanfield' at this scale")
    f = params.inhib_fraction
    if not 0.0 < f < 1.0:
        raise ValueError("inhib_fraction must be in (0, 1)")
    n_inhib = max(1, int(round(f / (1.0 - f) * n_main)))
    g = params.inhib_gain
    rng = np.random.default_rng(params.inhib_seed)

    # main -> inhibitory: excitatory drive (+1), so inhibitory cells integrate
    # how much of the main network is currently active.
    P_ei = rng.random((n_inhib, n_main)) < params.inhib_p_ei
    W_im = sp.csr_matrix(P_ei.astype(np.float32))  # (n_inhib, n_main)

    if params.inhib_scheme == "reciprocal":
        # symmetric: inhibitory -> main reuses the same pattern (transpose).
        W_mi = sp.csr_matrix((-g) * P_ei.T.astype(np.float32))
        P_ii = rng.random((n_inhib, n_inhib)) < params.inhib_p_ie
        P_ii = np.triu(P_ii, 1)
        P_ii = P_ii | P_ii.T
    elif params.inhib_scheme == "directed":
        # non-reciprocal: independent inhibitory -> main pattern -> oscillation.
        P_ie = rng.random((n_main, n_inhib)) < params.inhib_p_ie
        W_mi = sp.csr_matrix((-g) * P_ie.astype(np.float32))
        P_ii = rng.random((n_inhib, n_inhib)) < params.inhib_p_ie
        np.fill_diagonal(P_ii, False)
    elif params.inhib_scheme == "mixed":
        # ONE AXIS between the two schemes above, so asymmetry can be swept.
        # Each inhibitory->main synapse is drawn independently with probability
        # ``inhib_asym`` and otherwise reused from the main->inhibitory transpose:
        #   asym = 0  ->  W_mi == the reciprocal construction, exactly
        #   asym = 1  ->  a fully independent (non-reciprocal) feedback limb
        # Both source patterns are Bernoulli, so at p_ie == p_ei the expected
        # synapse count is INVARIANT in asym -- the knob moves how non-reciprocal
        # the loop is, not how strongly it inhibits.  The inhibitory->inhibitory
        # pool is kept in the SYMMETRIC (reciprocal) form throughout so that
        # ``inhib_asym`` is a single-axis knob on the main<->inhibitory loop; this
        # is why mixed@1.0 is not byte-identical to ``directed``, which also
        # de-symmetrises W_ii.  Run ``directed`` itself to get that.
        a = float(params.inhib_asym)
        if not 0.0 <= a <= 1.0:
            raise ValueError("inhib_asym must be in [0, 1]")
        # P_ii is drawn in the SAME RNG position as the reciprocal branch, and the
        # asymmetry draws come after, so that mixed@0.0 reproduces reciprocal
        # BYTE-IDENTICALLY (all three matrices) rather than merely in distribution.
        # That makes the asym=0 rung an exact internal replication of the static
        # baseline instead of an independent redraw of it.
        P_ii = rng.random((n_inhib, n_inhib)) < params.inhib_p_ie
        P_ii = np.triu(P_ii, 1)
        P_ii = P_ii | P_ii.T
        P_ind = rng.random((n_main, n_inhib)) < params.inhib_p_ie
        take_ind = rng.random((n_main, n_inhib)) < a
        P_ie = np.where(take_ind, P_ind, P_ei.T)
        W_mi = sp.csr_matrix((-g) * P_ie.astype(np.float32))
    else:
        raise ValueError(f"unknown inhib_scheme {params.inhib_scheme!r}")

    np.fill_diagonal(P_ii, False)
    W_ii = sp.csr_matrix((-g) * P_ii.astype(np.float32))
    b_inhib = np.full(n_inhib, params.inhib_excitability, dtype=np.float32)
    return InhibitorySubnet(
        n_inhib=n_inhib,
        b_inhib=b_inhib,
        tau_inhib=int(params.inhib_tau),
        W_mi=W_mi,
        W_im=W_im,
        W_ii=W_ii,
        locality="global",
    )


def build_cluster_local_inhibitory(n_main: int, params: Params,
                                   cluster: np.ndarray) -> InhibitorySubnet:
    """CLUSTER-LOCAL inhibition -- the same pool, PARTITIONED BY VERTEX CLUSTER.

    *** THIS IS NOT THE DISSERTATION'S MECHANISM.  Ch. 5 specifies ONE unstructured
    inhibitory pool wired at random across the whole main network; this partitions that
    pool so a cluster is suppressed in proportion to its OWN occupancy. ***  It exists
    because every inhibitory lever tested so far -- global mean-field, the exact global
    pool, imposed cyclic gain, directed non-reciprocal oscillation -- shifts EVERY neuron
    by the same term, which raises q and leaves the WTA slots INDEPENDENT.  A slot-
    independent mechanism cannot change the combinatorial cost of silencing every off-path
    slot at once, which is what the shallow basin is made of.  Cluster-local feedback is
    the one available mechanism that CORRELATES SILENCE WITHIN A CLUSTER: a cluster that
    fires drives its own inhibitory cells, which (with tau_inhib >> tau) shut that whole
    cluster down for a stretch, so occupancy becomes cluster-wise bursty rather than
    uniformly thinned.

    Construction -- everything except the SUPPORT of the connectivity is held fixed
    against ``build_inhibitory``:

      * cell budget.  The global pool has n_inhib = f/(1-f)*n_main cells.  Here each
        cluster gets ``max(1, round(f/(1-f) * |cluster|))``, so the per-neuron inhibitory
        budget is the same up to the rounding forced by small (degree-1 and degree-2)
        clusters, which would otherwise get zero cells and no local loop at all.  The
        realised total is returned and reported rather than assumed.
      * weights.  main->inhibitory +1, inhibitory->main and inhibitory->inhibitory
        -inhib_gain, identical to the global construction.  ``sim.simulate`` rescales
        W_mi by gain_t/inhib_gain exactly as before, so annealing still works.
      * scheme.  Within a block the wiring is symmetric (reciprocal), matching the
        ``inhib_scheme="reciprocal"`` control this is compared against.  With
        ``inhib_local_p == 1.0`` reciprocal and directed coincide inside a block, so the
        asymmetry axis is not defined here and is not swept.

    *** THE GAIN SCALES OF THE TWO MECHANISMS ARE NOT COMPARABLE AND MUST NEVER BE
    MATCHED ON. ***  A main neuron in the global pool receives from ~p_ie*n_inhib cells
    (~15 at N=60); here it receives from its own cluster's ~2-3.  More importantly the
    DRIVE differs: a global inhibitory cell integrates ~p_ei*n_main main neurons, a local
    one integrates only its cluster's 2*deg(i), so at equal occupancy the local pool sees
    an order of magnitude less excitation and sits far below threshold.  The physical
    quantity both mechanisms produce is the measured silent-slot fraction q, and that is
    what comparisons are matched on (see the internal harness's gain->q calibration).
    """
    f = params.inhib_fraction
    if not 0.0 < f < 1.0:
        raise ValueError("inhib_fraction must be in (0, 1)")
    p_local = float(params.inhib_local_p)
    if not 0.0 < p_local <= 1.0:
        raise ValueError("inhib_local_p must be in (0, 1]")
    g = params.inhib_gain
    rng = np.random.default_rng(params.inhib_seed)
    ratio = f / (1.0 - f)

    cluster = np.asarray(cluster)
    n_clusters = int(cluster.max()) + 1
    sizes = np.bincount(cluster, minlength=n_clusters)
    cells = np.maximum(1, np.rint(ratio * sizes).astype(np.int64))
    starts = np.concatenate(([0], np.cumsum(cells)))
    n_inhib = int(starts[-1])

    r_im, c_im = [], []          # inhibitory <- main
    r_ii, c_ii = [], []          # inhibitory <- inhibitory (within block, symmetric)
    for ci in range(n_clusters):
        mains = np.nonzero(cluster == ci)[0]
        inhs = np.arange(starts[ci], starts[ci + 1])
        # main -> inhibitory, restricted to the block
        mask = (np.ones((len(inhs), len(mains)), dtype=bool) if p_local >= 1.0
                else rng.random((len(inhs), len(mains))) < p_local)
        ii, jj = np.nonzero(mask)
        r_im.append(inhs[ii])
        c_im.append(mains[jj])
        # inhibitory -> inhibitory within the block, symmetric, no self-synapse
        if len(inhs) > 1:
            m2 = (np.ones((len(inhs), len(inhs)), dtype=bool) if p_local >= 1.0
                  else rng.random((len(inhs), len(inhs))) < p_local)
            m2 = np.triu(m2, 1)
            m2 = m2 | m2.T
            ai, aj = np.nonzero(m2)
            r_ii.append(inhs[ai])
            c_ii.append(inhs[aj])

    r_im = np.concatenate(r_im) if r_im else np.zeros(0, dtype=np.int64)
    c_im = np.concatenate(c_im) if c_im else np.zeros(0, dtype=np.int64)
    W_im = sp.csr_matrix((np.ones(len(r_im), dtype=np.float32), (r_im, c_im)),
                         shape=(n_inhib, n_main))
    # reciprocal within the block: the feedback limb is the transpose of the drive limb.
    W_mi = sp.csr_matrix((np.full(len(r_im), -g, dtype=np.float32), (c_im, r_im)),
                         shape=(n_main, n_inhib))
    r_ii = np.concatenate(r_ii) if r_ii else np.zeros(0, dtype=np.int64)
    c_ii = np.concatenate(c_ii) if c_ii else np.zeros(0, dtype=np.int64)
    W_ii = sp.csr_matrix((np.full(len(r_ii), -g, dtype=np.float32), (r_ii, c_ii)),
                         shape=(n_inhib, n_inhib))
    b_inhib = np.full(n_inhib, params.inhib_excitability, dtype=np.float32)
    return InhibitorySubnet(
        n_inhib=n_inhib,
        b_inhib=b_inhib,
        tau_inhib=int(params.inhib_tau),
        W_mi=W_mi,
        W_im=W_im,
        W_ii=W_ii,
        locality="cluster",
        cells_per_cluster=cells,
    )


# ----------------------------------------------------------------------
# Dense reference construction -- for validation only, O(N^2) memory.
# ----------------------------------------------------------------------
def dense_weights(net: Network) -> np.ndarray:
    """Materialise the full weight matrix W (Eq. 4.2). Validation only."""
    N = net.n_neurons
    if N > 6000:
        raise MemoryError(f"dense reference refuses N={N}")
    W = np.zeros((N, N), dtype=np.float64)
    p = net.params
    B = p.B

    for k in range(N):
        ck, compk, wk = net.cluster[k], net.comp[k], net.w[k]
        for l in range(N):
            if k == l:
                continue
            if net.cluster[l] != ck:
                continue
            if net.comp[l] != compk:  # opposite component
                if net.nbr[l] == net.nbr[k]:  # gamma (ii): same index
                    W[k, l] = p.I
                else:  # alpha
                    W[k, l] = p.alpha_scale * (1.0 - (wk + net.w[l]) / B)
            else:  # gamma (i): same component, all pairs
                W[k, l] = p.I

    # beta
    for k in range(N):
        for f in range(net.beta_idx.shape[1]):
            l = net.beta_idx[k, f]
            if l >= 0:
                W[k, l] = p.D
    return W
