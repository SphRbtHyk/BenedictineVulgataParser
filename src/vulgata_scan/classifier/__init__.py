"""Stage 2: apparatus zoning, segmentation and glyph-level CNN classification.

Ported unchanged from ``pipeline_final_code/`` of the paper's codebase (the
glyph classifier, its two-pass decoding, the geometric segmentation and the
zone selection); only the import layout is new.
"""

from .infer import SiglaClassifier
from .stage import run, run_chunk
from .taxonomy import load_taxonomy

__all__ = ["SiglaClassifier", "load_taxonomy", "run", "run_chunk"]
