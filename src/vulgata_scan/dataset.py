"""Loading and exploring the paper's dataset (gold + silver).

The dataset is a major contribution of the paper:

- **gold** -- the hand-verified parsed apparatus of Genesis 1-4 (106
  verse-verse entries), every field reviewed against the printed edition;
- **silver** -- the pipeline output over the whole book (Genesis, ~249
  scanned pages; 357 verse entries), machine-produced and human-spot-checked.

Both ship as JSON in the paper's flat reading model
(``[{chapter, verse, apparatus: [{variant_type, reference_reading / \
alternative_reading, witnesses, reading, missing, ...}]}]``).

A copy of each file is installed with the package (``data/``), so the
examples work without the surrounding repository; point ``cache_dir`` at
the repository's own ``dataset/`` if you want the authoritative copies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

#: Directory holding the packaged copies of gold + silver.
PACKAGE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

#: Default locations probed by :func:`load_gold` / :func:`load_silver` when
#: no explicit path is given: the packaged copy, then the repository layout.
GOLD_CANDIDATES = [
    PACKAGE_DATA_DIR / "gold_data.json",
    Path("dataset/gold_data/gold_data.json"),
]
SILVER_CANDIDATES = [
    PACKAGE_DATA_DIR / "genesis_p1_249.json",
    Path("dataset/silver_data/parsed_data/genesis_p1_249.json"),
]


def _load_first(candidates: List[Path], what: str) -> list:
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"could not find the {what} dataset; probed: "
        + ", ".join(str(p) for p in candidates))


def load_gold(path: Optional[Path] = None) -> list:
    """Load the gold dataset (hand-verified parsed apparatus, Genesis 1-4).

    Args:
        path: explicit JSON path (defaults to the packaged copy or the
            repository's ``dataset/gold_data/gold_data.json``).
    """
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return _load_first(GOLD_CANDIDATES, "gold")


def load_silver(path: Optional[Path] = None) -> list:
    """Load the silver dataset (pipeline-parsed apparatus of Genesis p1-249).

    Args:
        path: explicit JSON path (defaults to the packaged copy or the
            repository's ``dataset/silver_data/parsed_data/genesis_p1_249.json``).
    """
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return _load_first(SILVER_CANDIDATES, "silver")


def census(entries: list) -> dict:
    """Quick statistics of a parsed apparatus (used by the dataset explorer)."""
    apps = [a for e in entries for a in e["apparatus"]]
    return {
        "verse_entries": len(entries),
        "apparatus_units": len(apps),
        "chapters": sorted({e["chapter"] for e in entries},
                           key=lambda c: (0, int(c)) if str(c).isdigit()
                           else (1, str(c))),
        "variant_types": sorted({a["variant_type"] for a in apps}),
        "witnesses_total": len({w for a in apps
                                for w in (a.get("alternative_reading", {})
                                          or {}).get("witnesses", [])
                                + (a.get("reference_reading", {}) or {})
                                .get("witnesses", [])}),
    }


def pretty_entry(entry: dict) -> str:
    """Render one parsed verse entry as a compact human-readable block."""
    head = f"chapter {entry['chapter']}" + (
        f", verse {entry['verse']}" if entry.get("verse") else "")
    lines = [head]
    for i, unit in enumerate(entry["apparatus"], start=1):
        alt = unit.get("alternative_reading") or {}
        ref = unit.get("reference_reading") or {}
        parts = [f"  [{i}] {unit['variant_type']}"]
        if ref.get("reading"):
            parts.append(f"{ref['reading']} ]")
        if alt.get("reading"):
            parts.append(alt["reading"])
        if alt.get("witnesses"):
            parts.append("(" + " ".join(alt["witnesses"]) + ")")
        if unit.get("missing"):
            parts.append("[_desunt: " + " ".join(unit["missing"]) + "]")
        if unit.get("versions"):
            parts.append("ed./vers.: " + " ".join(unit["versions"]))
        if unit.get("citations"):
            parts.append("cites: " + " ".join(unit["citations"]))
        lines.append(" ".join(parts))
    return "\n".join(lines)


def show_samples(dataset: str = "gold", n: int = 3,
                 path: Optional[Path] = None) -> None:
    """Pretty-print ``n`` sample entries of the gold or silver dataset."""
    entries = load_gold(path) if dataset == "gold" else load_silver(path)
    print(f"--- {dataset} dataset: {census(entries)} ---")
    for entry in entries[:n]:
        print(pretty_entry(entry))
        print()
