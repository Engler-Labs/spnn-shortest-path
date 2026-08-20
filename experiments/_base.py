"""Shared helpers for experiments: deterministic result writers under results/.

Outputs are written deterministically (sorted keys, fixed formatting) so a
re-run produces byte-identical files -- the acceptance test is a number-by-number
diff against the committed results, so stability matters.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


def results_path(relpath: str) -> Path:
    """Absolute path under results/, with parent directories created."""
    p = RESULTS / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(relpath: str, obj) -> Path:
    """Write ``obj`` as deterministic JSON to results/<relpath>; return the path."""
    p = results_path(relpath)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return p


def write_tsv(relpath: str, header, rows) -> Path:
    """Write a TSV (header + rows) to results/<relpath>; return the path."""
    p = results_path(relpath)
    lines = ["\t".join(str(h) for h in header)]
    lines += ["\t".join(str(x) for x in row) for row in rows]
    p.write_text("\n".join(lines) + "\n")
    return p


def rel(path: Path) -> str:
    """Repo-root-relative string for logging."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
