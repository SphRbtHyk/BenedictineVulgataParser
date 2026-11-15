"""Stage 1: VLM transcription (model-agnostic, via vulgata_scan.backends)."""

from .transcribe import (APPARATUS_PROMPT, Stage1Artifact, transcribe_page,
                         transcribe_pdf)
from .rendering import render_pdf

__all__ = ["APPARATUS_PROMPT", "Stage1Artifact", "transcribe_page",
           "transcribe_pdf", "render_pdf"]
