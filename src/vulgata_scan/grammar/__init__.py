"""Stage 4: the formal grammar of the apparatus and its pyparsing parser.

Ported unchanged from ``parser/parse_apparatus.py`` of the paper's codebase.
``parse_apparatus`` maps the reconciled LaTeX-flavoured transcription onto
the flat JSON reading model used by the released gold/silver datasets.
"""

from .parser import (enforce_witness_inventory, expand_witness, parse_apparatus,
                     preprocess)

__all__ = ["parse_apparatus", "preprocess", "expand_witness",
           "enforce_witness_inventory"]
