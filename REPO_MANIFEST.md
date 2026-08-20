# `spnn-shortest-path` — repository build manifest

**Written 2026-08-17** for the submodule at `public_facing/spnn-shortest-path`.
Companion to `CLAIMS.tsv` (59 rows) and `spnn-public-repo-plan-v2.md`.

This document says exactly which files the repo needs, where each comes from, and which printed
numbers in the paper each one is responsible for. It is meant to be executable: every row in §3 is
a unit of work with a defined done-condition.

---

## 0. What this repo has to do

Paper 1's Code availability section, as committed in the npj submission build, promises:

> the compiler, the $O(|E|)$ simulator, the exact path enumerator, the frontier dynamic program for
> the matching polynomial, and **the scripts reproducing every table and figure in this paper** …
> The packed matching-polynomial counter records the top-coefficient bit length of every count, so
> the overflow condition … is visible in the released data rather than only in an assertion.

That is the acceptance spec. Nothing more is owed; nothing less is acceptable.

**Scope ruling stands** (2026-08-06): only code needed to run or reproduce **Paper 1**. Not
Paper 2, not the ~50 other harnesses, not research infrastructure.

---

## 1. Layout — and a reversal, with the reason

The mount point decides this. `public_facing/spnn-shortest-path` contains a hyphen, so it is not a
legal Python package path and cannot be imported in place from the monorepo. It therefore has to be
**installed** (`pip install -e public_facing/spnn-shortest-path`), and once installation is the
mechanism, the import path inside the repo is free.

So: **standard package layout, `spnn/` at the repo root.** An earlier note argued for flat layout
to preserve `python -m research.spnn.validate` — that argument only applied if the submodule mounted
*at* `research/spnn`, and it does not. Consumers do `from spnn import compile_network` after an
editable install.

```
spnn-shortest-path/
├── README.md
├── LICENSE
├── CITATION.cff
├── REPRODUCIBILITY.md
├── FINDINGS.md
├── CLAIMS.tsv
├── pyproject.toml
├── .gitignore
├── .github/workflows/verify.yml
├── spnn/                 the library
├── experiments/          one script per CLAIMS.tsv row group
├── figures/
├── verify/
├── instances/
└── results/              committed outputs
```

---

## 2. The reference instance, pinned

Every document in this project refers to "the reference instance." It is, per channel entry #176:

```python
random_graph(60, 0.05, seed=19)
```

with `B = 2·w_max`, $|E| = 142$, an $8$-hop optimum, $\widetilde{E}(P) = 0.276267$, and required
$c = 3.699428677298893$. **Pin this in `instances/README.md`** — it has been carried in prose
across three manuscripts and a register and has never been written down as a callable expression in
one place.

---

## 3. File manifest

**Origin key:** `PORT` = copy from the monorepo unchanged · `PORT+FIX` = copy and modify ·
`NEW` = write from scratch · `EXTRACT` = pull out of a larger monorepo harness.

### 3.1 Library — `spnn/`

| file | origin | source | claims | notes |
|---|---|---|---|---|
| `__init__.py` | NEW | — | — | export `compile_network`, `Params`, `simulate`, generators |
| `compile.py` | **PORT+FIX** | `research/spnn/compile.py` | all | ⚠️ **must carry the N.16 fix — see §5** |
| `sim.py` | PORT | `research/spnn/sim.py` | most | O(\|E\|) update, `E_NC`/`E_SPC`, `induced_edges`, `induced_path`, `mass_class`, `sampler_fire` |
| `graphs.py` | PORT | `research/spnn/graphs.py` **current tree** | all | ⚠️ **NOT from PR #3866** — see §6 |
| `validate.py` | PORT | `research/spnn/validate.py` | `OE-VALID`, `V-TV*`, `R-*` | checks A/B/C |
| `bench.py` | PORT | `research/spnn/bench.py` | `T-SCALE` | |
| `counting.py` | EXTRACT | `spnn_threshold_and_margin_counts.py` @ `be94acf7` | `CNT-*`, `POP-*`, `T-CREQDIST` | `count_matchings` (packed, auto-sized `DIG`, `CoefficientOverflow`), `scatter_counts` (frontier DP), path DFS. **Must persist `max_coeff_bits` per row** — the paper promises it |
| `yen.py` | EXTRACT | vendored Yen from the stage-26 pilot | `TOL-*` | loopless k-shortest-paths, K configurable |

**Library delta across the whole research chain was only `compile.py` and `sim.py`** (A38), so the
port surface here is genuinely small.

### 3.2 Experiments — `experiments/`

One script per CLAIMS row group. Fourteen files.

| file | origin | claims rows | tier | notes |
|---|---|---|---|---|
| `table_pathcounts.py` | **NEW** | `T-PATHCOUNTS` | 1 | ⚠️ gap — §4.1 |
| `table_encverify.py` | **NEW** | `T-ENCVERIFY` | 1 | ⚠️ gap — §4.1 |
| `variant_ordering.py` | NEW/EXTRACT | `VAR-REF`, `VAR-EQWT`, `VAR-INV`, `VAR-INV4` | 1 | the constructed 4-hop pairs; cheap to write fresh |
| `margin_verify.py` | EXTRACT | `M-VERIFY`, `M-MICRO`, `M-VARIANT` | 1 | from `spnn_threshold_and_margin_parta.py` |
| `reference_counts.py` | EXTRACT | `CNT-PATH`, `CNT-SCAT`, `CNT-CROSS` | 1 | the anchor; re-derives byte-identical (#176) |
| `creq_sweep.py` | PORT | `POP-N`, `T-CREQDIST`, `POP-SPAN`, `POP-SIZE`, `POP-SIZED`, `C4-*`, `CNT-HEAD` | 2 | the stage-28 sweep. ⚠️ **filename not recorded in the register** — likely `spnn_stage28_required_c_population.py`; confirm against commits `77d9e291`…`8922d6b1` |
| `ratio_law.py` | EXTRACT | `LAW-FIT`, `LAW-PRED`, `LAW-HOLDOUT`, `LAW-DEPTH`, `LAW-RHO` | 1 | fit + leave-one-family-out; runs off committed CSVs, so Tier 1 |
| `weight_convention.py` | PORT | `WT-PRED`, `WT-MEAS`, `WT-ERR` | 2 | the U[1,2] arm. **119 of 120 completed** — use the corrected figures from #176, not #175 |
| `lambda_tolerance.py` | PORT | `TOL-REF`, `TOL-DENOM`, `TOL-POP`, `TOL-FAM`, `TOL-K` | 1/2 | 72 instances, K=4000, DFS cross-check |
| `counting_limits.py` | EXTRACT | `CNT-BIND`, `CNT-BITS`, `CNT-49` | 1/2 | stage 26 pilot + stage 27 fix verification |
| `cooling_sweep.py` | **NEW** | `T-DEGENERACY`, `DEG-TARGET` | 2 | ⚠️ gap — §4.2 |
| `clambda_grid.py` | PORT | `SW-ENRICH`, `SW-EVENTS`, `SW-DESIGN`, `SW-VARIANT` | 2 | ⚠️ **harness name unrecorded and never `b_st`-audited** — §4.3 |
| `mass_ratio_pilot.py` | PORT | `MR-NULL`, `MR-ACT` | 2 | the only consumer of `sim.mass_class` (A11) |
| `bench_scaling.py` | PORT | `T-SCALE` | 2 | thin driver over `spnn/bench.py`; label `pinned` |

### 3.3 Figures — `figures/`

| file | origin | claims | notes |
|---|---|---|---|
| `fig_creq.py` | NEW | `F-CREQ-A`, `F-CREQ-B` | emits the TikZ coordinates for both panels from `results/creq/`. **Emit the numbers the manuscript's TikZ consumes, not a rendered image** — that keeps figure and data provably the same object |

Figures 1 and 2 are hand-drawn TikZ schematics with no data behind them. Nothing to ship.

### 3.4 Verification — `verify/`

| file | origin | notes |
|---|---|---|
| `__main__.py` | NEW | the runner. `python -m verify` = Tier 1 only; `--full` = everything. **Must print what it skipped** |
| `check_a.py` | EXTRACT | decomposition == dense, `1.99e-7` |
| `check_b.py` | EXTRACT | sampler == Boltzmann, TV at τ=3/5/10, plus `--legacy-sampler` for the 0.235 contrast |
| `check_c.py` | EXTRACT | end-to-end 12/12, plus `--slot0` for the 6/12 contrast |
| `check_claims.py` | NEW | walks `CLAIMS.tsv`, runs each row's script, diffs against the committed output at the row's tolerance. **An unlabelled row fails.** |

`check_claims.py` is the piece that makes the Code availability sentence literally true rather than
aspirational. It should be the thing the CI badge reports.

### 3.5 Instances — `instances/`

| file | origin | notes |
|---|---|---|
| `README.md` | NEW | pins the reference instance (§2) and the conventions: `L≥3` by terminal selection, `MIN_EDGE_WEIGHT=1e-3`, aux edges exactly `0.0`, weights `U(0,1)` (and `U[1,2]` for the rider arm) |
| `population.csv` | PORT | all **1,080** rows — the 1,027 completed **and the 53 censored**, flagged not dropped |
| `manifest.json` | NEW | family × size × seed grid; 4 families, 27 cells, seeds 1–40 |

**Ship seeds and the generator, not serialized graphs.** That is what the Data availability
statement promises ("reconstructible exactly rather than only in distribution"), and it is why
§6's PR #3866 hazard matters.

### 3.6 Root documents

| file | origin | notes |
|---|---|---|
| `README.md` | NEW | what it is, one-command verify, link to the paper and its DOI |
| `LICENSE` | NEW | **MIT** recommended — Nature wants OSI-approved; fills one manuscript `XXX` |
| `CITATION.cff` | NEW | ⚠️ point at the **tag or Zenodo DOI, never `main`** |
| `REPRODUCIBILITY.md` | NEW | tier policy, `--full` cost, label semantics, the censoring disclosure |
| `FINDINGS.md` | NEW | curated from register PART C + PART D, **sanitized of channel entries #3–#184 and PR numbers #3811–#3907** |
| `CLAIMS.tsv` | DRAFTED | already written — 59 rows, in the project as `claude/spnn-paper1-CLAIMS.tsv` |
| `pyproject.toml` | NEW | package `spnn`; deps numpy + scipy; no networkx unless already used |
| `.github/workflows/verify.yml` | NEW | Tier 1 on every PR. **The green badge is worth real credibility with a referee** — it shows the 33 Tier-1 claims re-derived from scratch, which a code-availability sentence alone does not |

---

## 4. The three provenance gaps

47 of 59 claims rows can be labelled `reproduces` today. Six cannot, in three clusters.

### 4.1 `tab:pathcounts` and `tab:encverify` — no harness was ever identified

Register **A1** records both as verified but names no script. **Do not do archaeology.** Both are
deterministic enumerations on an instance the captions specify completely: the path
$0$–$1$–$2$–$3$–$4$–$5$ with weights $1,2,3,4,5$, a pendant at each end so $\deg(s)=\deg(t)=2$,
$B = 2\varphi_{\max} = 12$, energies at $c=1$, $D=4$, $b_k=-2$.

Write both fresh and diff against the committed table cells. **If the fresh script disagrees with
the manuscript, that is a finding, and it is one you want before a referee has it.** Half a day.

### 4.2 `tab:degeneracy` — the weakest artifact in the paper

No register entry, no harness, **seed unrecorded**, and measured on the **variant** (the target's
$16$ active neurons is $2L$, not $2(L{+}2)$). It is the last [VAR] number in Paper 1 and it supports
§IV-D's structural claim about the terminal drive.

**Recommendation: re-run on the augmented construction with a named seed.** Six rows at $20{,}000$
steps and $N=60$ is seconds on the O(\|E\|) simulator. **Risk, stated:** if [AUG] behaves
qualitatively differently, §IV-D needs rewriting close to the deadline. Judged unlikely — Prop. 2
says the empty configuration is strictly cheaper than every path at $\lambda \le 0$ on *both*
compilations — but it is a real risk and the decision is Gary's.

Fallback: ship `pinned` with the seed declared unrecoverable and one clause added to the caption.

### 4.3 The $(c,\lambda)$ grid — the one unaudited harness

`SW-ENRICH` / `SW-EVENTS` come from register **A5**. The harness name is not recorded anywhere and
it is **the only Paper 1 harness never audited for the `b_st` default**. The 2026-08-06 plan said
all four sampled results carried `b_st` exposure; that is over-stated for three — `tab:degeneracy`,
the mass-ratio pilot and `tab:scale` all ran pre-augmentation or on the variant, where `b_st` cannot
bite. **This one is [AUG] and genuinely exposed.**

These numbers are load-bearing: the paper's only behavioural evidence that the continuity scale,
not the length-bias condition, moves the network into the path sector.

**Do the source-scan** — same shape as A50's `s26_bst_provenance.json`. An afternoon. Until it
lands these rows stay `TODO`, and the standing policy — *"we do not publish unknown"* — forbids
shipping them labelled.

---

## 5. N.16 blocks the tag

Ruled 2026-08-05: **option 3 + option 4** — library-level refusal to compile augmented unless
`b_st` is named, plus stamping the resolved value and whether it was named onto the returned
`Network`.

**Implemented: nothing.** A49 verified from the diff that `compile.py`, `sim.py` and `graphs.py`
are untouched; stage 26 edited nothing, stage 27 touched only the counter, stage 28 never
constructed `Params`. **`compile.py:88` still reads `b_st: float | None = 0.0`.**

Scope is known and small: of 57 files constructing `Params`, 24 compile augmented and **9 do so
without naming `b_st`** — option 3 breaks exactly those 9. Implementation is a tri-valued
`UNSET`/`None`/float field.

The register's old line that N.16 *"blocks nothing about the paper"* was true **only while the code
stayed private.** It no longer is: repo tag → Zenodo deposit → two of the manuscript's four
remaining `XXX` placeholders.

The monorepo-wide `Params` import refactor across all 57 sites remains a **separate** job and is not
on this path.

---

## 6. Do not include, and one hazard

**Exclude:** the job runner and queue consumer (`claude_code_jobs_consumer.py`), Discord alerting,
the tables library, service code, credentials, channel logs, `py_lib/`, `.ai_changelog/`,
`zz_scratch/`, root `CLAUDE.md` / `FINDINGS.md` / `EXPERIMENTS.md`, `pr-checks.yml`, and all ~50
Paper 2 harnesses (`spnn_depth_law_augmented`, `spnn_recalibrate_gain_q_augmented` and its five
importers, `spnn_level_drop_drive_vs_repricing`, `spnn_nucleation_tax_*`,
`spnn_matched_depth_decoupling_*`, `spnn_exclusivity_lever_readouts`, `n60_gating_report`, …).

**Fresh single commit — do not filter monorepo history into this repo.** Two committed artefacts
make that concrete rather than hypothetical: the stage-25 changelog claims the harness reproduces
"on both committed populations" when only 25 of 480 units ran, and `s25_provenance.json` reports
12 matches / 28 mismatches against a stage report claiming 40/40. Both stay private under a fresh
start. Both go public under a filter.

⚠️ **Hazard — PR #3866 (unmerged).** It rewrites the `graphs.py` generator bodies including the
`np.maximum(w, 1e-3)` floor. If it merges before extraction it may move the graph produced at a
given seed and invalidate 26 result directories. **Port `graphs.py` from the current tree and pin
the source commit in `instances/README.md`.**

---

## 7. Acceptance test

Not a passing test suite. **A number-by-number diff against the committed monorepo values**, driven
by `CLAIMS.tsv`.

The failure mode being guarded against is specific: a harness in the new tree silently resolving a
*different* implementation of a helper that was excluded from the port. A green test suite does not
catch that. A value mismatch does.

Done-condition: `python -m verify --full` walks all 59 rows, every row resolves to `reproduces`,
`pinned` or `not-published`, and no row is unlabelled.

---

## 8. Sequence

| | work | gates |
|---|---|---|
| 1 | Post Paper 3 to arXiv; record the id | manuscript line 49; also unblocks the Crucible Reference stub pending from #184 |
| 2 | **N.16 — implement options 3+4, fix the 9 call sites** | the tag, and everything after it |
| 3 | Port `spnn/` (8 files); `pyproject.toml`; editable install from the monorepo | everything below |
| 4 | `b_st` source-scan of the $(c,\lambda)$ harness (§4.3) | 2 TODO rows |
| 5 | Write `table_pathcounts.py`, `table_encverify.py` fresh (§4.1) | 2 TODO rows |
| 6 | Decide and act on `tab:degeneracy` (§4.2) | 2 TODO rows, possibly a manuscript table |
| 7 | Port the remaining 11 experiment scripts; commit `results/` | `check_claims.py` |
| 8 | `verify/`, CI workflow, root documents | the badge |
| 9 | Tag; two Zenodo deposits; fill the remaining `XXX` | Data + Code availability |

Steps 4–6 are independent of each other and of step 2, and can run in parallel with it.

---

## 9. Open decisions

| # | decision | recommendation |
|---|---|---|
| 1 | `tab:degeneracy` — re-run on [AUG] or ship `pinned` | **re-run**; risk named in §4.2 |
| 2 | license | **MIT** |
| 3 | one Zenodo deposit or two | **two** — the manuscript has two DOI slots |
| 4 | how much register goes public | curated `FINDINGS.md`, sanitized |
| 5 | acknowledgements | Gary's |
