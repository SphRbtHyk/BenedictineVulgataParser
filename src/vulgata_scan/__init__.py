"""vulgata-scan: parse the critical apparatus of the Benedictine Vulgate.

Four-stage pipeline (the paper's method):

1. ``vlm``      -- VLM transcription of scanned apparatus pages via a
   pluggable :class:`~vulgata_scan.backends.ModelBackend` (Mistral API or
   any local OpenAI-compatible server);
2. ``classifier`` -- glyph-level siglum classification (CNN) over the
   segmented apparatus crops;
3. ``reconcile`` -- reconciliation of the two readings into a consensus
   transcription (write-back);
4. ``grammar``  -- formal-grammar parse into the flat JSON reading model.
"""

from .backends import (LocalBackend, MistralApiBackend, ModelBackend,
                       build_backend)
from .grammar.parser import parse_apparatus, preprocess as preprocess_apparatus
from .pipeline import PipelineResult, run_pipeline

__version__ = "1.0.0"

__all__ = [
    "ModelBackend", "MistralApiBackend", "LocalBackend", "build_backend",
    "parse_apparatus", "preprocess_apparatus", "run_pipeline",
    "PipelineResult", "__version__",
]
