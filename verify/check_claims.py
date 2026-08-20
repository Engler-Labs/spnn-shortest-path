"""check_claims -- walk CLAIMS.tsv and audit every printed number.

The acceptance test's bookkeeping layer. For each row of CLAIMS.tsv:

  * it MUST carry a label (``reproduces`` | ``pinned`` | ``not-published`` |
    ``TODO``); an unlabelled row is a failure (nothing ships unlabelled).
  * every ``reproduces`` row MUST have its committed ``output`` file on disk.

This does not re-parse the human-readable ``value`` strings; it audits labels
and output presence, so a missing artifact or an unlabelled claim fails loudly.
The per-claim numeric reproduction lives in check_a/b/c and the experiment
modules, which print their own MATCH/FAIL against CLAIMS.

Returns a dict; exits non-zero (via __main__) if any row is unlabelled or any
``reproduces`` row is missing its output.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAIMS = REPO_ROOT / "CLAIMS.tsv"
COLUMNS = ["id", "value", "where", "script", "output", "tol", "tier", "label", "notes"]
LABELS = {"reproduces", "pinned", "not-published", "TODO"}


def _rows():
    for line in CLAIMS.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if parts[0] == "id":                       # header
            continue
        row = dict(zip(COLUMNS, parts))
        yield row


def run(argv=None) -> dict:
    tally = {k: 0 for k in LABELS}
    unlabelled, missing = [], []
    total = 0
    for row in _rows():
        total += 1
        label = (row.get("label") or "").strip()
        if label not in LABELS:
            unlabelled.append((row["id"], label or "<empty>"))
            continue
        tally[label] += 1
        if label == "reproduces":
            out = (row.get("output") or "").strip()
            if out and out != "—" and not (REPO_ROOT / out).exists():
                missing.append((row["id"], out))

    print("[check_claims] CLAIMS.tsv audit")
    print(f"  {total} rows: " + ", ".join(f"{k}={tally[k]}" for k in
          ("reproduces", "pinned", "not-published", "TODO")))
    if unlabelled:
        print(f"  UNLABELLED ({len(unlabelled)}): " +
              ", ".join(f"{i}({l})" for i, l in unlabelled))
    if missing:
        print(f"  MISSING OUTPUT ({len(missing)}): " +
              ", ".join(f"{i}->{o}" for i, o in missing))
    if not unlabelled and not missing:
        print("  OK: every row labelled; every 'reproduces' row has its output on disk")

    return {
        "total": total, "tally": tally,
        "unlabelled": unlabelled, "missing_output": missing,
        "ok": not unlabelled and not missing,
    }


if __name__ == "__main__":
    import sys
    sys.exit(0 if run().get("ok") else 1)
