"""The experiment catalog -- the single source of truth for the runner.

Every Paper 1 experiment gets a short id (a number, optionally with a trailing
letter for a split/variant).  The runner (``python -m experiments <id>``) looks
the id up here and dispatches to ``experiments.<module>.run``.  ``state`` says
whether the module is built yet; unbuilt rows still show up in ``list`` so the
catalog doubles as the build plan.

Ordering follows the paper (the CLAIMS.tsv ``where`` column): section IV then V
then VI.  ``claims`` are the CLAIMS.tsv ids each experiment is responsible for.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    id: str
    module: str                 # experiments/<module>.py
    title: str
    section: str                # paper section (CLAIMS 'where')
    claims: tuple               # CLAIMS.tsv ids produced
    outputs: tuple              # results/ paths written
    tier: int                   # 1 = CI-cheap/exact, 2 = seeded/expensive
    state: str                  # "built" | "planned"
    origin: str = ""            # provenance / how to build it
    note: str = ""


EXPERIMENTS = [
    Experiment("1", "table_pathcounts", "tab:pathcounts -- path enumeration",
               "IV-A", ("T-PATHCOUNTS",), ("results/tables/pathcounts.tsv",), 1,
               "built", "NEW -- counts ONE config (not s-t paths); cleared by A4",
               "all 9 cells match derivations; aug E_NC=-6 via b_st=None (b_st=0 companion=-10)"),
    Experiment("2", "table_encverify", "tab:encverify -- encoding verification",
               "IV-B", ("T-ENCVERIFY",), ("results/tables/encverify.tsv",), 1,
               "built", "NEW -- b_k sweep -3..-1 at b_st=b_k (settled convention)",
               "reproduces the manuscript table (aug E_NC=-6.0 at b_k=-2); decomp==dense verified"),
    Experiment("3", "lambda_tolerance", "lambda-tolerance / optimality gap",
               "IV-C", ("TOL-REF", "TOL-DENOM", "TOL-POP", "TOL-FAM", "TOL-K"),
               ("results/tolerance/reference.json", "results/tolerance/population.csv",
                "results/tolerance/dfs_crosscheck.json"), 1,
               "built", "PORT; uses spnn.yen; --population/--crosscheck re-run (tier-2)",
               "all 5 TOL-* verified; reference reproduced fresh, population off committed data"),
    Experiment("4", "cooling_sweep", "tab:degeneracy -- b_k cooling sweep",
               "IV-D", ("T-DEGENERACY", "DEG-TARGET"),
               ("results/cooling/bk_sweep.csv", "results/cooling/status.json"), 2,
               "built", "re-run AUGMENTED, named seed=19 (Gary decision #1)",
               "DEG-TARGET [VAR] 16/8/-6.49 + [AUG] 20/10/-8.447; 6-row aug sweep; manuscript adopts [AUG]"),
    Experiment("5", "variant_ordering", "constructed variant orderings",
               "IV-E", ("VAR-REF", "VAR-EQWT", "VAR-INV", "VAR-INV4"),
               ("results/variant/reference.json", "results/variant/constructed.json"), 1,
               "built", "NEW -- constructed 4-hop pairs",
               "all 4 VAR-* verified vs CLAIMS (1e-6)"),
    Experiment("6", "margin_verify", "margin verification",
               "V-A", ("M-VERIFY", "M-MICRO", "M-VARIANT"),
               ("results/margin/verify.tsv", "results/margin/conventions.json",
                "results/margin/variant.json"), 1,
               "built", "EXTRACT from an internal margin harness",
               "M-VERIFY exact-zero under Params(b_st=None); required_c cross-checks CNT-CROSS"),
    Experiment("7", "reference_counts", "reference-instance counts + crossing c",
               "V-B", ("CNT-PATH", "CNT-SCAT", "CNT-CROSS"),
               ("results/counts/reference.json",), 1,
               "built", "EXTRACT -- the Part B/C anchor; pure combinatorics"),
    Experiment("8", "creq_sweep", "required-c population sweep",
               "V-B", ("POP-N", "T-CREQDIST", "POP-SPAN", "POP-SIZE", "POP-SIZED",
                       "C4-POOL", "C4-FAM", "C4-SIZE", "C4-CONV", "CNT-HEAD"),
               ("results/creq/population.csv", "results/creq/by_family.tsv",
                "results/creq/by_size.tsv", "results/creq/c4_verdict.json"), 2,
               "built", "EXTRACT; population.csv = 3 committed chunks, byte-identical",
               "all 10 verified vs the committed summary + CLAIMS; --regenerate re-runs"),
    Experiment("9", "ratio_law", "numerator ratio-law fit",
               "V-B", ("LAW-FIT", "LAW-PRED", "LAW-HOLDOUT", "LAW-DEPTH", "LAW-RHO",
                       "LAW-INTERACT"),
               ("results/law/numerator_fit.json", "results/law/per_instance.csv",
                "results/law/holdout.json", "results/law/depth_transfer.json",
                "results/law/rho.tsv", "results/law/interaction_fit.json"), 1,
               "built", "EXTRACT; fresh OLS fit off committed population.csv",
               "all 5 LAW-* reproduced to 1e-9; LAW-INTERACT is the W4 refit "
               "(beta1 test, --interaction)"),
    Experiment("10", "weight_convention", "U[1,2] weight-convention arm",
               "V-B", ("WT-PRED", "WT-MEAS", "WT-ERR"),
               ("results/weights/prediction.json", "results/weights/measured.csv"), 1,
               "built", "PORT; corrected 119-instance data shipped",
               "10/10 verified; measured.csv is the corrected 119-instance data (not the 103-partial)"),
    Experiment("11", "counting_limits", "counting limits + overflow audit",
               "V-C", ("CNT-BIND", "CNT-BITS", "CNT-49"),
               ("results/counts/limits.json", "results/counts/bit_ladder.tsv",
                "results/counts/overflow_audit.json"), 1,
               "built", "EXTRACT; uses spnn.counting overflow guard",
               "CNT-BITS ladder + CNT-49 (49/1080) verified; CNT-BIND is pinned (6GB cap)"),
    Experiment("12", "clambda_grid", "(c, lambda) enrichment grid",
               "V-D", ("SW-ENRICH", "SW-EVENTS", "SW-DESIGN", "SW-VARIANT"),
               ("results/grid/config.json", "results/grid/enrichment.csv",
                "results/grid/bst_provenance.json"), 2,
               "built", "--rerun-augmented at b_S=b_k reproduces the §V-D behavioural numbers",
               "SW-ENRICH/SW-EVENTS reproduce (orig-hop 0/12/80; events 0/31/426); paper states no improvement vs variant"),
    Experiment("13", "mass_ratio_pilot", "mass-ratio null pilot",
               "V-D", ("MR-NULL", "MR-ACT"),
               ("results/mass/null.json", "results/mass/mass_ratio_pilot_by_unit.csv"), 2,
               "built", "PORT; only consumer of spnn.sim.mass_class; committed CSV shipped",
               "8/8 verified off the 72-unit by-unit CSV; --full re-runs to a separate file"),
    Experiment("14", "bench_scaling", "tab:scale -- O(|E|) scaling",
               "VI-C", ("T-SCALE",), ("results/bench/scaling.tsv",), 2,
               "built", "PORT; thin driver over spnn.bench",
               "PINNED: structural columns exact, timing columns hardware-local"),
    Experiment("15", "b_floors", "B floors per instance",
               "VII-B", ("B-FLOOR-FAM", "B-FLOOR-UNREACH"),
               ("results/creq/b_floors.json",), 1,
               "built", "NEW -- arithmetic off population.csv (floor = num/(L+1))",
               "per-instance B->inf floor + % unreachable at c=4; per-family medians "
               "cross-check the W2 B-sweep (sparse 4.70 / dense 4.74 / grid 2.68 / mtn 3.46)"),
    Experiment("16", "ordering_tolerance", "ordering tolerance (20) vs compiled net",
               "IV-C", ("ORD-TOL",), ("results/tolerance/ordering.json",), 1,
               "built", "NEW -- removes the last [DERIVED] tag (Prop 2)",
               "L in {3,5,8}; compiled-network E_N argmin vs Dijkstra across lambda; "
               "measured inversion threshold / tau >= 1 confirms the bound"),
    Experiment("17", "variant_rate", "variant-vs-Dijkstra rate + augmented control",
               "IV-E", ("VAR-RATE", "ORD-CONTROL"),
               ("results/variant/disagreement.json",), 2,
               "built", "NEW -- population enumeration at lambda=0, c=4",
               "variant E_N-min vs Dijkstra 16.9% disagree (excess median 22.3%); the "
               "augmented control is 0/427 (Cor 1(iii)) -- a population-scale ordering check"),
    Experiment("18", "deficit_counts", "N_d / N'0 via degree-<=2 deficit DP",
               "V-D", ("DEF-COUNTS", "DEF-33", "DEF-36"),
               ("results/counts/deficit.json",), 2,
               "built", "NEW -- degree-<=2 frontier DP; reference instance (~46s)",
               "PROVISIONAL (#227): both-aux subset undercounts; see cycle_prevalence + "
               "the corrected full-space result pending the section V-D framing"),
    Experiment("19", "cycle_prevalence", "d=-1 disjoint-cycle sector prevalence",
               "V-D", ("CYC-PREV",), ("results/counts/cycle_prevalence.json",), 2,
               "built", "NEW -- two-sided certificates (absence/presence/undetermined)",
               "how common is a disjoint-cycle set cheaper than the path (W_cyc<w_max+W*); "
               "rigorous per-instance certificates, undetermined never imputed"),
]

BY_ID = {e.id: e for e in EXPERIMENTS}
