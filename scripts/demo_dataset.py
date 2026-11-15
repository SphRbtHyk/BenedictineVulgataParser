#!/usr/bin/env python3
"""Pretty-print sample entries of the gold and silver datasets.

Uses the packaged dataset copies (no backend, no model):

    python scripts/demo_dataset.py [--samples N] [--which gold|silver|both]

The same census is available from the CLI: ``vulgata-scan dataset``.
"""

from __future__ import annotations

import argparse
import sys

from vulgata_scan.dataset import census, load_gold, load_silver, pretty_entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=2,
                    help="entries per dataset to pretty-print")
    ap.add_argument("--which", default="both",
                    choices=["gold", "silver", "both"])
    args = ap.parse_args(argv)

    for name, loader in (("gold", load_gold), ("silver", load_silver)):
        if args.which not in (name, "both"):
            continue
        entries = loader()
        c = census(entries)
        print(f"=== {name} dataset "
              f"({c['verse_entries']} verses, {c['apparatus_units']} units, "
              f"{c['witnesses_total']} witnesses, "
              f"chapters {','.join(c['chapters'])}) ===")
        for entry in entries[:args.samples]:
            print(pretty_entry(entry))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
