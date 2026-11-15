"""Stage 3: reconciliation of the VLM and classifier readings (write-back).

Ported unchanged from ``pipeline_final_code/writeback/writeback.py`` of the
paper's codebase; the library entry point is :func:`reconcile_chunk`.
"""

from .writeback import reconcile_chunk, writeback_region

__all__ = ["reconcile_chunk", "writeback_region"]
