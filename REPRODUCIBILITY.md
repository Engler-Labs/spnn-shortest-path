# Reproducibility

The acceptance test for this artifact is **not** a passing test suite — it is a
**number-by-number diff** of freshly computed values against the committed
`results/`, driven by `CLAIMS.tsv`. This document states the tier policy, what
each label and runner state means, the cost of a full run, and the disclosures.

## Tiers

Every claim is tagged Tier 1 or Tier 2 in `CLAIMS.tsv`:

| Tier | Meaning | When it runs |
|---|---|---|
| **1** | Deterministic / exact — closed forms, exact enumeration, or a fixed-seed computation. Reproduces bit-for-bit. | Default; cheap enough for CI. |
| **2** | Seeded or expensive — a population sweep or a long sampler run. | Only under `--full`. |

`python -m verify` and `python -m experiments all` run Tier 1 by default and print
exactly what they skipped; `--full` adds Tier 2. Tier-2 experiments ship their
**committed** population outputs and re-derive the summaries from them by default;
regenerating the raw population from scratch is gated behind a flag
(`--regenerate` / `--population` / `--full`, per experiment) because it is
expensive (the required-c population is ~90 core-minutes; the mass-ratio pilot is
143.28M sampler steps).

## Labels (in `CLAIMS.tsv`)

Every row carries one; an **unlabelled row is a failure** (`check_claims` enforces this).

- **reproduces** — regenerated here and matched to the committed value within the row's tolerance.
- **pinned** — hardware- or environment-dependent (a memory ceiling, an untuned timing table). The value is recorded, not diffed.
- **not-published** — imported from other work and flagged as such in the paper; not reproduced here.
- **TODO** — cannot yet be labelled `reproduces` (see *Known gaps* below).

## Runner states (in `experiments/registry.py`)

The runner reports a build state per experiment, orthogonal to the claim labels:

- **built** — module runs and its claims are verified against `CLAIMS.tsv`.
- **needs-diff** — runs and is internally self-consistent, but its cells live in
  the manuscript (not in this repo), so they can't be value-verified here; they
  need a manual diff against the printed table.
- **blocked** — runs but *documents a reproduction gap* rather than reproducing a
  claim (writes a status file explaining what and why). Excluded from `all`.
- **planned** — not yet built.

`python -m experiments list` shows the state of every experiment.

## Known gaps (the honest TODOs)

- **`tab:pathcounts`** — resolved (`reproduces`). It counts active neurons, β and
  α pairs and the energy decomposition on *one* path configuration (not s–t
  paths); all cells match their derivations.
- **`tab:encverify`** (`needs-diff`, `TODO`): a b_k sweep from −3.00 to −1.00 at
  L=5; the decomposition==dense and closed-form checks pass, but the ten cells
  still need a manual diff against the printed table (not in this repo). Its
  augmented column is computed at `b_st=None` to match the manuscript (see below).
- **`tab:degeneracy`**: the target row (16 active / 8 induced / E_N = −6.489) is
  reproduced deterministically (`DEG-TARGET`, `reproduces`); the 36 *sampled*
  cells (`T-DEGENERACY`, `TODO`) have no identified harness and an unrecorded
  seed, and are pending a decision to re-run on the augmented construction with a
  named seed or ship `pinned`.
- **The (c, λ) enrichment grid** (`SW-ENRICH`, `SW-EVENTS`, `pinned`): the source
  harness is **unaugmented**, so the auxiliary-terminal bias is never read — the
  "unaudited parameter" concern is moot (provenance in
  `results/grid/bst_provenance.json`). The rows are `pinned`, not because of that
  parameter, but because the published enrichment digits come from a superseded
  graph population; the current harness gives the same qualitative turn-on with
  different digits. *Manuscript item: re-quote to the current population or note
  the change.*
- **The pre-fix sampler contrast** (`V-TVBAD`) — resolved (`reproduces`). The
  deliberately-incorrect legacy sampler (`spnn/legacy_sampler.py`: exponential-
  clock firing probability + decrement-after-fire) is shipped and reproduces
  TV ≈ 0.235 at τ=3, so the whole §III-D correction is reproducible.
- **The slot-restricted readout** (`R-SLOT0`, `not-published`): the exact 6/12
  configuration is not identified in the released library (`beta_fanout=1` gives
  12/12), so the measurement is not reproduced; the point is made by derivation
  instead (a slot-restricted decoder misses a constant fraction of induced edges,
  so its success falls geometrically in path length). `R-AGN` (12/12) reproduces.

## Terminal bias (`b_st`) and the competitor convention

Under the published convention, a size-matched **scatter competitor admits the two
zero-weight auxiliary edges** (S,s) and (t,T). With that convention S and T are
active in *both* the path and the competitor, the auxiliary bias `b_S` cancels
identically, and **the discrimination margin is `b_st`-independent**: the
path-vs-scatter margin identity holds exactly at both `b_st=0.0` (anchored, the
paper's §III-A choice) and `b_st=None` (inherit b_k). `results/margin/verify.tsv`
emits all sixteen rows with an explicit `arm` column and the discrepancy at both
`b_st` values, so this is visible in the data. A same-size *all-real* scatter (no
terminals) is a different comparison whose discrepancy differs by `2·b_S − 2·b_k`
(= 4.0 at the reference `b_k=−2`); that is the terminal-bias discount the
augmentation removes, not a failure, and it is not part of the margin claim.

**Terminal bias — settled: `b_S = b_T = b_k`.** The *raw augmented energies* in
`tab:pathcounts` / `tab:encverify` do depend on the terminal bias (they differ by
`2·b_k` between conventions): the printed augmented E_NC (e.g. −6.0 at the sweep's
center) is the `b_S=b_k` (`b_st=None`) value; the anchored `b_S=0` value is −10.0.
`b_S=b_k` is the **convention in force** — it is what the energy tables were
measured at and what the paper's equation (13) computes with. It is a **stated
design parameter**: `b_S` does not enter `λ`, so Definition 2, the ordering
condition, the tolerance and the whole margin analysis are invariant; it only
shifts the affine constant, which is exactly the quantity Proposition 2 is now
stated *conditionally* against (silence undercuts the path sector under `b_S=b_k`;
under `b_S=0` only when `Ẽ(P) > D/2c`). This repo computes the augmented columns at
`b_st=None` accordingly and emits the `b_st=0.0` value as a labelled companion.

## Disclosures

- **Censoring.** Of 1,080 population instances, 1,027 completed and **53 (4.9%)
  were censored** — **52 of the 53 hit the 120 s per-count wall** (the `#scatter`
  count on sparse/dense, the `#path` count on terrain), and **1 hit the memory
  cap**; not "at the memory cap". They fall in `sparse_100` (16), `dense_80` (17),
  `mtn_knn_140` (18), `sparse_80` (1), `dense_60` (1) — **zero lattice/grid**.
  The bias direction **cannot be assumed favourable**: within every affected cell,
  required `c` *falls* with optimum depth among the completed instances (Spearman
  −0.48…−0.90 across the irregular cells), so the censored (deepest) instances would,
  on that trend, require *less* — making the reported quantiles if anything too high
  and the clear-fractions too low. Nothing is imputed; censored rows are flagged in
  `results/creq/population.csv`, not dropped (`POP-N`). (The separate
  competitor-count *lower bound* keeps its own favourable-direction bias.)
- **Counting overflow is in the data.** The matching-polynomial counter is
  auto-sized and records `max_coeff_bits` per instance in the population CSV, so
  the overflow condition is visible in the released data rather than only in an
  assertion. A legacy fixed 96-bit field would have silently miscounted 49 of the
  1,080 instances, always toward understating required `c` (`CNT-49`, `CNT-HEAD`).
  The `CNT-BITS` bit ladder (`12 20 30 41 53 65 78 90 103` for k=4…12) is the
  **largest coefficient over all k** (`max_coeff_bits`), i.e. the physically
  correct overflow quantity — it diverges from the matched-k coefficient only at
  k=4,5 (where the matching polynomial peaks below the matched scatter size), and
  it is what a "top coefficient grows ~12 bits per rung" statement must refer to.
- **A self-corrected arm.** The `U[1,2]` weight-convention results were first
  analysed off a partially written output; the corrected 119-of-120-instance data
  are the published ones (`WT-MEAS`).

## The deficit profile subsample (experiment 21)

Experiment 18 established the deficit ladder's shape **on the reference instance
only**, and experiments 19–20 both found that instance to be the *minority* case on
its own cell — so the shape needed testing off it. Experiment 21 does that on a
declared 12-instance stratified subsample. What is disclosed here is the budget, the
censor, and the validation chain.

- **Budgets declared in advance, not after the fact.** Per-instance wall **240 s**
  and memory **4.0 GB** (`RLIMIT_AS` in a child process), concurrency ≤ 4. Both are
  recorded in `results/deficit_profile/s30_meta.json` alongside the declared
  subsample, and `s30_meta.json` is written **last** so a cut leaves no analysable
  chunk (`n_instances` is asserted against the row count before anything reads it).
- **One censored instance, and which resource.** `grid` **k = 12** (N = 144,
  L = 22, m = 24) breached the **memory** cap — `RLIMIT_AS 4.0 GB` — at 66.7 s. It
  was **not retried, the budget was not raised mid-run, and nothing was substituted
  for it**; its row is kept with `status = MEMCAP`, its censor note, its wall and its
  peak RSS. It is also the **deepest cell in the plan**, so the single coverage gap
  sits at the top of the depth range — exactly where a shape change would most
  plausibly hide, which is why 11/11 is reported qualified rather than bare. Note the
  asymmetry with experiment 8, which finished that same instance in 0.59 s at 38 MiB:
  this DP is `3^frontier` where the required-`c` count is `2^frontier` (`DP-COST`).
  Like-for-like against experiment 8 on the completed instances, the profile costs
  **1.3× – 637×** the required-`c` count. The originating report quoted the top ratio as
  **641×**; recomputed here from the `scatter_wall_s` committed in
  `results/creq/population.csv` it is **636.9×** — 641 came from dividing by that wall
  rounded to 0.33 s. Both figures are on the record rather than one being quietly
  preferred.
- **The validation chain, in the order it was run.** (1) The DP was validated
  against **exhaustive brute-force enumeration** before any instance was measured —
  **156/156** over six tiny graphs × all four auxiliary sub-cases × every *K*, plus
  **27/27** on the identity `N_(m−1) == scatter_counts.micro_aux`. (2) On the sweep
  itself that identity holds **exactly, as integers, on 11/11** completed instances,
  and this repository re-checks it against the `scatter_micro_aux` column already
  committed in `results/creq/population.csv` — i.e. against *independent* code, with
  no recount required. (3) The **full profile** is re-derived by **exhaustive
  enumeration of every edge subset, at every d**, on **two real subsample instances**
  (`grid_k4_N16_seed1`, 1,562,275 subsets; `sparse_N20_seed1`, 7,059,052) — ground
  truth on real instances, not toys, covering the d = −1 cycle sector and the d = 2
  rung as well. (4) `python -m experiments 21` additionally **recounts** the
  committed profiles with this repository's own `count_degree2_by_deficit`
  (`--recount-all` for all 11; the default is a declared cheap subset).
  **Named rather than skipped:** the nine larger instances are *not* exhaustively
  verified — enumeration is `C(|E|, m)` and out of reach (the reference alone is
  `C(142,10) ≈ 1.3e15`). They rest on the independent identity plus the DP's
  pre-validation.
- **Wall and RSS do not reproduce.** `DP-COST` is labelled **pinned** for the same
  reason `T-SCALE` is: the state counts, frontier and coefficient widths are
  structural and exact, but the timing and memory columns are hardware-local to the
  originating box.
- **A printed digit, corrected — and where it was settled.** An earlier manuscript
  draft printed the d = 2 rung of the reference ladder as **11.25**. The computed
  rung is **11.244618410987663**, i.e. **11.24** at two places under any rounding
  convention. The originating run could not distinguish a transcription slip from a
  0.7 % difference in `N_2` and **said so rather than resolving it**; it was settled
  **from the deposit side**, with this repository's own `results/counts/deficit.json`
  — `N_2 = 1,320,177,088,811,008` and `#path = 898,048` reproduce all nine rungs at
  two places *with* the corrected digit. The rung is **non-binding** (the argmax is
  d = 1 at 18.11), so nothing the paper concludes depended on it. The manuscript now
  prints 11.24, and prints the crossing margin as **8.4475** — `CNT-CROSS` deposits it
  exactly, `8.447466352389776`; the claim's own value string quotes it at four
  significant figures as 8.447, the same number at lower precision — so that a reader
  reconstructing the ladder from the paper's printed inputs lands on 11.24 rather
  than on 11.2452 → 11.25. **`results/counts/deficit.json` was correct throughout and
  is untouched.** `results/deficit_profile/s30_anchor.json` is the originating lane's
  record, committed **unmodified**, so its `verdict` field still reads
  `DOES_NOT_REPRODUCE` against the superseded 11.25; against the corrected printed
  ladder experiment 21 reports all 23 anchor checks agreeing.
- **Section labels.** `CLAIMS.tsv`'s `where` column uses the section numbering the
  claim register was built on. In the submitted render the discrimination
  condition/ladder is **§V-C**, the below-path sector is **§V-D** and the
  measurements are **§V-E**; the register's `V-D` (ladder, `DEF-*`/`CYC-*`/`DP-*`)
  and `V-E` (below-path, `DZ-*`) predate that renumbering. `V-B` is unaffected. The
  register is left as it stands rather than renumbered, since the manuscript itself
  is `\ref`-based and was never wrong.

## The acceptance test

`check_claims` (run by `python -m verify`) walks `CLAIMS.tsv` and requires that
every row is labelled and every `reproduces` row has its committed output on disk.
Combined with each script's own value comparison, the guarantee is specific: a
claim reproduces only if a from-scratch recomputation matches the committed number
within tolerance — which catches a helper silently resolving to a different
implementation, where a green test suite would not.
