"""Experiment runner: ``python -m experiments <id>`` (and list / all / info).

    python -m experiments 7            # run experiment 7
    python -m experiments list         # show the catalog
    python -m experiments info 7       # show one experiment's plan
    python -m experiments all          # run every BUILT tier-1 experiment
    python -m experiments all --full   # include tier-2 (seeded/expensive)

Any extra args after an id are passed straight through to the experiment, e.g.
``python -m experiments 3 --reference``.
"""

from __future__ import annotations

import importlib
import sys

from .registry import BY_ID, EXPERIMENTS

USAGE = __doc__


def _load(exp):
    return importlib.import_module(f"experiments.{exp.module}")


def cmd_list() -> int:
    w_id = max(2, max(len(e.id) for e in EXPERIMENTS))
    print(f"{'id':<{w_id}}  {'sec':<5} {'tier':<4} {'state':<7} {'claims'}")
    print("-" * 78)
    for e in EXPERIMENTS:
        print(f"{e.id:<{w_id}}  {e.section:<5} {e.tier:<4} {e.state:<7} "
              f"{e.title}")
    print("\nRun one with:  python -m experiments <id>")
    return 0


def cmd_info(id_) -> int:
    e = BY_ID.get(id_)
    if e is None:
        print(f"unknown experiment id: {id_!r} (try `python -m experiments list`)",
              file=sys.stderr)
        return 2
    print(f"experiment {e.id}: {e.title}")
    print(f"  module   experiments/{e.module}.py   (state: {e.state})")
    print(f"  section  {e.section}   tier {e.tier}")
    print(f"  claims   {', '.join(e.claims)}")
    print(f"  outputs  {', '.join(e.outputs)}")
    if e.origin:
        print(f"  origin   {e.origin}")
    if e.note:
        print(f"  note     {e.note}")
    return 0


def run_one(e, argv) -> int:
    if e.state == "planned":
        print(f"experiment {e.id} ({e.title}) is PLANNED, not built yet.",
              file=sys.stderr)
        print(f"  build plan: {e.origin}", file=sys.stderr)
        return 3
    if e.state == "blocked":
        print(f"note: experiment {e.id} is BLOCKED -- it runs but documents a gap "
              f"rather than reproducing its claims. {e.note}", file=sys.stderr)
    mod = _load(e)
    if not hasattr(mod, "run"):
        print(f"experiment {e.id}: module experiments/{e.module}.py has no run()",
              file=sys.stderr)
        return 4
    mod.run(argv)
    return 0


# states whose run() produces a real result file (so `all` should run them)
RUNNABLE_IN_ALL = {"built", "needs-diff"}


def cmd_all(argv) -> int:
    full = "--full" in argv
    rest = [a for a in argv if a != "--full"]
    ran, skipped = [], []
    for e in EXPERIMENTS:
        if e.state not in RUNNABLE_IN_ALL:
            skipped.append((e, e.state))          # blocked / planned
            continue
        if e.tier > 1 and not full:
            skipped.append((e, "tier-2 (use --full)"))
            continue
        print(f"=== experiment {e.id}: {e.title} ===")
        run_one(e, rest)
        ran.append(e)
    print(f"\nran {len(ran)} experiment(s); skipped {len(skipped)}.")
    for e, why in skipped:
        print(f"  skipped {e.id:<3} {e.title}  [{why}]")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "list":
        return cmd_list()
    if cmd == "info":
        if not rest:
            print("usage: python -m experiments info <id>", file=sys.stderr)
            return 2
        return cmd_info(rest[0])
    if cmd == "all":
        return cmd_all(rest)
    e = BY_ID.get(cmd)
    if e is None:
        print(f"unknown experiment id: {cmd!r} (try `python -m experiments list`)",
              file=sys.stderr)
        return 2
    return run_one(e, rest)


if __name__ == "__main__":
    raise SystemExit(main())
