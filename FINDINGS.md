# Findings — stochastic spiking networks for shortest paths

This is the findings companion to the released `spnn` artifact: a reference implementation of the
graph → stochastic spiking-network construction of Engler (2018), *Stochastic Neural Algorithms*,
Ch. 4, rebuilt so that a single Boltzmann sampling step costs **O(|E|)** instead of O(N·d²).

The narrative below is curated for the public record. Every printed number is grounded in the
committed ground truth of this repository: `CLAIMS.tsv` lists one row per number in the paper, and
`results/` holds the reproduced output each number is read from. Where a figure is cited here, the
row id in `CLAIMS.tsv` (e.g. `CNT-CROSS`) names the file it comes from. The reference graph used
throughout is pinned in `instances/README.md` as `random_graph(60, 0.05, seed=19)`.

The arc is: (1) the construction and why its energy minima are shortest paths; (2) the O(|E|)
reduction; (3) two discrete-time dynamics corrections the Boltzmann validation forced; (4) the
exactness checks; (5) the discrimination threshold — how strongly the energy landscape has to be
sharpened before the shortest path is the *unique* minimum, quantified across a 1,080-instance
population; and (6) the honest limits of the bare single network.

---

## 1. The construction

A weighted, undirected graph `G = (V, E, w)` is compiled into a network of winner-take-all (WTA)
neuron clusters — one cluster per vertex, two WTA "slots" per cluster (the two edges a simple path
uses to enter and leave a vertex). A neuron in a slot stands for a candidate incident edge. The
cluster structure enforces the local constraint that a path uses at most one edge per slot.

The couplings are set so that the network's Boltzmann energy, read over the space of firing
configurations, has its minima exactly at the configurations encoding shortest `s`–`t` paths. Two
energy pieces carry this:

- a **connectivity / neighbour-continuity** term (E_NC) that rewards firing patterns which chain
  into a connected `s`–`t` walk, and
- a **shortest-path-cost** term (E_SPC) that charges the total edge weight along the encoded walk.

The network is then run as a plain **discrete-time Boltzmann sampler**: each step, every neuron
fires with a probability set by its instantaneous drive, subject to a refractory window of `tau`
steps. Over many steps the sampler visits low-energy configurations most often. The path is read
out of the **induced subgraph** — the set of graph edges whose representing neurons are currently
active — by tracing the connected `s`–`t` component.

**Slot assignment is arbitrary, and the readout must respect that.** The two WTA components in a
cluster are *not* "incoming" and "outgoing" — the graph is undirected and nothing orients it, so a
legitimate path may occupy either slot at a vertex. Two consequences shaped the code:

1. Beta coupling connects all slot pairs (`beta_fanout = 2`). Slot-matched wiring would penalise
   valid paths whose slot assignment is inconsistent across vertices, forcing the network to
   discover a global orientation the construction never asks for.
2. The induced-graph readout is **component-agnostic**: it reads an edge as present whenever *either*
   slot's neuron is active. Reading a single slot silently drops edges and lets the network find
   paths the decoder cannot see. Fixing this took the exact-hit rate against Dijkstra from **6 of
   12** to **12 of 12** on the same graphs, seeds and parameters (`R-SLOT0`, `R-AGN`).

---

## 2. The O(|E|) reduction

The naive transcription of the intra-cluster coupling stores a weight matrix with
`Σ_i (2·deg_i)²` nonzeros and performs a sparse matrix–vector product every timestep. That is the
dominant cost, and it scales as O(N·d²).

The key observation is that the alpha weights are **separable**. For a pair of candidate edges
`(e_ij, e_ik)` sharing vertex `v_i`,

    f(e_ij, e_ik) = 1 − ( w(v_i, v_j) + w(v_i, v_k) ) / B ,

which splits into a term depending only on the pre-synaptic neuron and one depending only on the
post-synaptic neuron. The whole dense intra-cluster block therefore collapses to **two scalar
reductions per WTA component**:

    A[s]  = number of active neurons in segment s
    Sw[s] = sum of edge weights over active neurons in segment s .

Every alpha and gamma contribution follows in closed form from `A` and `Sw`; the beta term is a
fixed gather over graph edges. Cost and memory drop from **O(N·d²) to O(|E|)** per step. The energy
readouts (E_NC / E_SPC) come from the same two reductions, so tracking energy during a run is free.

The decomposition is not an approximation. Computed against the explicit weight matrix on the
reference network, the maximum per-neuron discrepancy is **1.99×10⁻⁷** in float32 (`OE-VALID`),
unchanged after scaling the couplings by the continuity scale `c`. The scaling win over a
sparse-matrix baseline is recorded in `results/bench/scaling.tsv` (`T-SCALE`); it is offered as a
scaling demonstration, not a tuned benchmark.

---

## 3. Two discrete-time dynamics corrections

An independent Boltzmann validation — comparing the sampler's stationary distribution against exact
enumeration on small graphs — surfaced two corrections to the discrete-time dynamics. Both are
load-bearing: without either, the sampler does not converge to the intended Boltzmann distribution.

1. **Firing probability.** The dissertation's dynamics give a firing *rate*. Using `p·dt` or the
   exponential-clock survival `1 − exp(−p·dt)` as a per-step probability biases every neuron toward
   silence and breaks the stationary distribution. With a deterministic `tau`-step active window the
   stationary odds are `tau · (q/(1−q))`, so the neural-computability condition forces the per-step
   probability

       q = exp(u) / (tau + exp(u)) = sigmoid(u − log tau) .

2. **Refractory ordering.** The refractory counter must be decremented at the *top* of the step.
   Decrementing after firing leaves neurons active for `tau − 1` steps and shifts the stationary
   distribution.

Together these take the total variation from the exact distribution from **0.235** (rate-as-
probability with decrement-after; `V-TVBAD`) down to **0.0225** at `tau = 10` (`V-TV10`).

---

## 4. The network is exact, and it is verified end to end

Three independent checks establish correctness, and each is re-derived from scratch by the shipped
verifier (`verify/`) rather than asserted:

- **Decomposition == dense.** Max per-neuron discrepancy **1.99×10⁻⁷** (`OE-VALID`).
- **Sampler == Boltzmann.** Total variation from exact enumeration is **0.0558** at `tau = 3`,
  **0.037** at `tau = 5`, and **0.0225** at `tau = 10` (`V-TV3`/`V-TV5`/`V-TV10`), with the pre-
  correction sampler at 0.235 kept as a regression contrast.
- **End-to-end readout.** The induced-subgraph decoder recovers Dijkstra's shortest path on **12 of
  12** held-out instances (`R-AGN`), against **6 of 12** for the slot-blind decoder (`R-SLOT0`).

Because the energy identities are exact, they can be stated in closed form and checked against the
compiled network. On a path state the active/β/α neuron counts are exactly `2(L+1) / L+2 / L+1`,
`E_NC = −λ·L + c` with the length-neutral bias `λ = 2·b_k + D + c`, and on the augmented
construction (below) `E_SPC = 2c·Ẽ(P)` exactly — the network's cost term is exactly twice the
continuity-scaled path weight, which is what makes *shortest* coincide with *minimum energy*.

**Two corrections to the dissertation's published algebra** came out of this exact accounting, and
both are stated honestly in the paper:

- Several miscounts in the length-neutral bias (`b_k`) derivation are genuine arithmetic errors and
  are corrected.
- An apparent boundary-term discrepancy in the cost energy turned out **not** to be an error but a
  dropped construction step. The dissertation's Step 1 augments the graph with two new vertices
  `S`, `T` joined to the source and target by **zero-weight** edges; an earlier version of the
  compiler omitted them. Restoring the augmentation (`compile_network(..., augment_st=True)`) makes
  the alpha pair-sum equal `2Σφ` exactly and the boundary term vanish, so the code is faithful and
  the doubling in the cost energy is a global ×2 that preserves the argmin.

The distinction matters for honesty about the construction. Without the augmentation the terminal
edges are under-charged — a one-hop path costs exactly zero and a two-hop path is charged at half
rate — an implicit length discount. The paper documents this by construction: on the unaugmented
"variant," an explicit weight-inverted pair prefers the **heavier** path by a margin of `0.58` in
node energy at `c = 1` (`VAR-INV`), widening to `2.32` at `c = 4` (`VAR-INV4`, the constructed
counterexample cited in the abstract — constructed, not sampled). On the augmented construction the
path-sector minimum agrees with Dijkstra and the ordering corollary holds.

---

## 5. The discrimination threshold — how sharp the landscape must be

Correct energy *ordering* is necessary but not sufficient. The operational question is quantitative:
how strongly must the continuity scale `c` sharpen the landscape before the shortest path is the
*unique* Boltzmann minimum, rather than one microstate lost in a sea of near-degenerate competitors?
The competitor that matters is not another path but a **scatter** — a set of disjoint, equal-cost
edges that carries no `s`–`t` connectivity yet occupies the same energy shell. Only the alpha term
distinguishes a chained path from a scatter (the length-bias term `D` cancels between them — a
proof, not a tuning artifact), so `c` is the real discrimination lever.

### 5.1 The reference instance

On the pinned reference graph (`|E| = 142`, an 8-hop optimum, `Ẽ(P) = 0.276267`):

- there are **1,754** simple `s`–`t` paths at the optimum's depth (`CNT-PATH`), and
- **3.352349×10¹⁹** matched-size scatter configurations (`CNT-SCAT`, from an exact frontier dynamic
  program over the matching polynomial).

The scatter sea outnumbers the paths by a **log-ratio of 31.25 nats** (microstate convention). At
`c = 1` the shortest path leads by an energy margin of only **8.447**, so the scatter wins on sheer
multiplicity. Solving for the scale at which the path's energy advantage exactly offsets the scatter's
entropy gives the crossing

    c* = 3.699428677298893      (CNT-CROSS)

reproduced byte-identically across three independent regenerations of the count. Below `c*` the
scatter dominates; above it the shortest path is the unique minimum.

**Convention matters, and the paper names it every time.** Counting *microstates* (each active edge
carries a per-edge multiplicity, `4^k` per matching against `2^(L+1)` per path — a ratio of 2,048,
worth `7.62` nats or `+0.89` in `c`) gives `c* = 3.699`; counting *edge sets* gives `2.808`
(`M-MICRO`, `M-VERIFY`). The microstate convention is the headline throughout, and the two
conventions are reported side by side wherever they disagree.

### 5.2 The population sweep — 1,080 instances

The threshold was measured across a population spanning four graph families (sparse random, dense
random, city lattice, terrain k-NN), sizes from N = 20 to N = 100, and many seeds. Of **1,080**
instances generated, **1,027 completed** and **53 (4.9%) were censored** — and the censoring is
disclosed, not hidden, because it is depth-correlated and biases the summary *low* (`POP-N`).

The headline result, under the microstate convention:

> **13.4% of instances clear at `c = 4`** (`C4-POOL`).

That is, for all the exactness of the construction, sharpening the landscape to `c = 4` makes the
shortest path the unique minimum on only about one instance in seven. The verdict is convention-
sensitive: under the edge-set convention **40.0%** clear, and the two conventions **disagree on 273
of the 1,027** instances (`C4-CONV`) — which is exactly why every `c = 4` sentence in the paper
states its convention.

The population structure is informative:

- **By family:** the city lattice clears **0 of 360**; terrain clears **35.1%** (`C4-FAM`). Lattices
  fail because their corner-to-corner optima are deep (median required `c ≈ 13.6`), not because of
  ties — even instances with thousands of optimal paths do not clear.
- **By size:** the requirement **rises with size**. Sparse-family median required `c` climbs
  `4.22 → 5.01 → 5.88 → 6.15 → 6.46 → 7.77` from N = 20 to N = 100 (`POP-SIZE`); at N = 60 only
  **1 of 40** sparse instances and **0 of 39** dense clear at `c = 4` (`C4-SIZE`).
- **Spread:** required `c` spans **1.65 to 19.59** across the population, pooled IQR `4.47–10.32`
  (`POP-SPAN`).

The population figure (`F-CREQ-B`) records an informative *negative*: the size-slope of required `c`
has **opposite signs across families** (`−0.61 / −0.76 / +0.81 / +0.02`), so there is no single
"requirement grows with N" law — the size dependence is mediated by depth and weight structure.

### 5.3 A predictive law for the requirement

The requirement is not noise. It is captured by a two-parameter law for the log-ratio numerator,
fitted once over all 1,027 completed instances with **no per-family term**:

    ln(#scatter / #path) = c0 + a·L + b·ln|E| ,      required c = numerator / margin,
    margin = (1 − 2ρ)·L + 1

with **a = 2.289** [2.250, 2.328], **b = 11.566** [11.224, 11.909], `c0 = −42.451`, and
**R² = 0.966** (`LAW-FIT`). Against each instance's own measured margin the law predicts required `c`
to a **median relative error of 8.6%** (R² = 0.965, `LAW-PRED`).

It genuinely generalises out of sample. Leave-one-family-out refits predict held-out family medians
to ratios **1.16 / 0.87 / 0.84 / 1.08** (`LAW-HOLDOUT`). The depth term transfers across weight
conventions where the absolute level does not: at `L ≈ 14` the lattice sits at required `c = 13.57`
against terrain's `4.30` (a factor 3.2), and the law fitted on the other three families predicts
**4.01** for the lattice's depth — i.e. it transfers to terrain but not to the lattice's near-tie
weight convention (`LAW-DEPTH`). The whole family spread lives in the tie-density `ρ`, whose medians
are `0.119 / 0.076 / 0.430 / 0.105` for sparse / dense / grid / terrain (`LAW-RHO`); the lattice's
`ρ = 0.43` drives its `(1 − 2ρ)` margin close to zero and its requirement toward the asymptote.

**A held-out convention change confirms the law prospectively.** Switching edge weights from `U(0,1)`
to `U[1,2]` was *predicted* — before measuring — to raise `ρ` from 0.119 to ≈ 0.36 and compress the
margin `(1 − 2ρ)` from 0.76 to ≈ 0.29 (`WT-PRED`). Measured: `ρ ≈ 0.357`, `(1 − 2ρ) ≈ 0.286`, with
per-cell required-`c` ratios of `1.95 / 1.95 / 1.89`, **0 of 119** instances clearing at `c = 4`,
and a median relative error of **7.5%** from the unseen-convention fit (`WT-MEAS`, `WT-ERR`). The
weight arm also carries an honest self-correction: it was first analysed off a partially written
output and re-run to completion (119 of 120 instances), and the corrected figures are the published
ones.

### 5.4 The optimality gap — how much length-bias the answer tolerates

A companion measurement asks the dual question: given the operating point, how large a `λ`
(length-bias) perturbation can the network absorb before a *different*, shorter rival overtakes the
true optimum? On the reference instance the optimum is `W* = 0.548769` at `L = 8`; its nearest
length-changing rival is `W_Q = 0.550628` at `L = 4` (`TOL-REF`). The weight gap is tiny — a
denominator of `4.648×10⁻⁴`, giving a tolerance of `1.9×10⁻³` at `c = 4` (`TOL-DENOM`). Across a
72-instance population (depths 3–15) the median denominator is `5.2×10⁻²`, with family medians
`1.04 / 0.067 / 0.042 / 1.3×10⁻³` for lattice / sparse / dense / terrain (`TOL-POP`, `TOL-FAM`).
Every reported tolerance is an **upper bound**: a `K = 4000` loopless k-shortest-paths enumeration
reached `K` on all 72 instances, and a direct DFS cross-check agrees on **147 of 162** strata
(`TOL-K`). The near-tie families (terrain, dense) tolerate almost no length-bias — the answer is a
photo finish — which is the same near-degeneracy the ratio law reads through `ρ`.

### 5.5 Counting was the hard part, and the overflow is in the public data

Every number in §5 rests on exact integer counts of paths and matched-size scatters. The scatter
count is a matching-polynomial coefficient that grows explosively with graph size, and a fixed-width
accumulator will silently **carry and mask** the top coefficient once it exceeds the field — returning
a too-small count and thereby *understating* the required `c`. This is the most dangerous failure
mode in the paper because it fails quietly toward a more favourable answer.

The released counter is a **packed, auto-sized** accumulator that records the top-coefficient bit
length of every count. The bit ladder for the lattice family runs

    12  20  30  41  53  65  78  90  103        (k = 4 … 12, CNT-BITS)

so a legacy fixed **96-bit** field is exact through `k = 11` and **fails at `k = 12`**. Auditing the
whole population against that legacy width, **49 of 1,080 instances would have been silently wrong**
(`CNT-49`) — 40 in the deepest lattice cell and 9 in the deepest terrain cell — and in every case the
masked count ran *toward understating* required `c`. The fix is verified, and the promise the paper
makes in its Code availability statement is kept literally: `max_coeff_bits` is **persisted per
instance** in `results/creq/population.csv`, and every published instance sits at least **51 bits
below the ceiling** (`CNT-HEAD`), so the overflow condition is visible in the released data rather
than only in an assertion.

The exact counter has a hardware boundary, pinned rather than reproduced: at a 6 GB memory cap the
sparse family stops completing past `N ≈ 140` and the dense family past `N ≈ 100` (`CNT-BIND`), the
frontier dynamic program's peak memory growing with the elimination frontier.

---

## 6. Honest limits of the bare single network

The exactness results (§4) and the threshold results (§5) are landscape statements: they say where
the shortest path *is* the minimum energy configuration. They do not say the sampler *reaches* it.
Paper 1 is explicit that the bare single network has real limits, and quantifies them.

**The continuity scale, not the length-bias, is what moves the network into the path sector.** A
behavioural sweep along `λ = 0` — a design of 8 graphs with `L ≥ 8`, N = 60, 3 seeds, 10,000 steps
each (`SW-DESIGN`) — shows containment selection switching on with `c`: the `c = 1`, `λ = 0` control
sits at chance, so the bias correction alone does nothing, and genuine selection requires `c > 1`,
exactly as the discrimination analysis predicts. The quantitative enrichment figures from this sweep
(`SW-ENRICH`, `SW-EVENTS`) are held back pending a provenance audit and are not published here.

**The sampler runs hot — an Θ(N) activity obstruction.** A mass-ratio pilot ran 72 units for
**143,280,000 steps** and found the network occupying the matched-energy shell for only **2,116
steps** (a fraction of `1.5×10⁻⁵`), with the path-and-scatter mass in that shell **identically zero**
(`MR-NULL`). Mean activity ran from **25.0 to 43.5** active neurons against an ideal path occupancy of
`2N = 24` (`MR-ACT`) — the network is far denser than any single path, so from an empty start it
rarely nucleates the sparse optimum, even though the optimum is by far the energy minimum. This is
the informative failure: the landscape is correct and the mixing is the obstacle.

**Identification is hardest exactly where the requirement is highest — deep, dense, near-tie
instances.** The three results reinforce one another: required `c` rises with depth and size (§5.2),
the near-tie families tolerate almost no length-bias (§5.4), and the sampler's hot, high-activity
regime is worst on the largest graphs (this section). The lattice family, which is both deepest and
most tie-dense, clears at no size — its optima are deep, and depth is the dominant axis in the fitted
law (§5.3). Taken together these are the paper's honest case that a *single* bare network is a
small-scale, shallow-optimum device: it computes the right landscape at O(|E|) cost and reads the
answer out exactly when it reaches it, but on deep or dense instances the requirement outruns what a
single sampler nucleates. That is stated as the measured direction the results indicate — the remedy
they point toward is bounding *depth per level* by composition — not as a solved claim, and the
compositional sequel is out of scope for this paper.

---

## Reproducing these numbers

Every number above is committed. `CLAIMS.tsv` maps each printed value to the script that regenerates
it and the tolerance it must meet; `results/` holds the reference outputs. The verifier re-derives
the deterministic (Tier 1) claims from scratch:

    pip install -e .
    python -m verify            # Tier 1: exact / deterministic claims, runs in CI
    python -m verify --full     # adds the seeded, expensive population claims

A claim reproduces only if its freshly computed value matches the committed output within tolerance;
an unlabelled claim is a failure. That number-by-number diff — not a passing test suite — is the
acceptance test for this artifact.
