"""Stage 1: VLM transcription of the critical apparatus, via any ModelBackend.

For every input page this stage:

1. renders the native 330 PPI grayscale page (``rendering.py``);
2. detects the printed divider rules to crop the *apparatus band* (the
   middle band between the two horizontal rules -- the parsing target),
   falling back to the full page when the rules are not readable;
3. sends the band image to the configured :class:`~vulgata_scan.backends.ModelBackend`
   with the canonical apparatus prompt;
4. emits the stage-1 artifact in the same contract as the paper's original
   GLM runner: ``model_pages`` (a list of pages, each a list of regions with
   ``bbox_norm_1000`` and ``content``) plus ``page_sizes``.

Downstream (classifier and reconciliation stages) consumes only that
artifact, so the backend is fully swappable: Mistral API, mlx-vlm, llama.cpp,
vLLM, LM Studio...
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

from ..backends import ModelBackend
from ..classifier.apparatus_filter import detect_rules
from .rendering import image_size, render_pdf

#: The canonical apparatus-transcription prompt (backend-independent).
#: The LaTeX-flavoured notation it requests is the pipeline's interchange
#: notation: the reconciliation stage spans ``$\\Sigma^{\\mathrm{T}}$``-style
#: macro groups, and the grammar's preprocessor accepts them.
APPARATUS_PROMPT = """\
Transcribe the critical apparatus printed on this scanned Vulgate page band.
Input is the apparatus band of one page of the *Biblia Sacra iuxta latinam \
vulgatam versionem* (Benedictine Vulgate).

The band has the standing "Desunt" (or "Deest") witness-absence line at its \
top followed by the apparatus proper, printed in columns (left column first, \
then right column) or full measure when the page has no second column.

Answer with a single JSON object with exactly three string values:
- "desunt": the Desunt/Deest line exactly as printed (empty if absent);
- "left": the apparatus of the left column (below the Desunt line, or the \
full-measure apparatus when the band has a single column);
- "right": the apparatus of the right column (empty when there is no second \
column).
Transcribe each part EXACTLY as printed, from top to bottom, without \
translating, summarizing, skipping or adding anything.

Notation rules:
- Sigla in LaTeX macros with superscripts in the superscript braces: \
$\\Sigma^{\\mathrm{T}}$, $\\Theta^{\\mathrm{HG}}$, $\\mathrm{M}^{\\mathrm{2}}$, \
hand marks *, 1, 2 attached to their witness; Fraktur witnesses as \
$\\mathfrak{g}$; dagger $\\dagger$; pilcrow $\\P$.
- Apparatus signs kept literally: | separates readings, ~ marks the \
reconstructed text, + marks an addition, ] announces a substitution, = marks \
a positive reading.
- Verse numbers as line-initial "N." exactly as printed.
Output only the JSON object, nothing else.
"""


def apparatus_band(image: Image.Image) -> tuple[int, int]:
    """Detect the ruled apparatus band on a native page; ``(y0, y1)`` pixels.

    Uses the same rule detector as stage 2 (not by coincidence: both stages
    must agree on the zones).  Falls back to the full page minus small margins
    when the two printed rules are not detectable.
    """
    rule_ys = detect_rules(image)
    if len(rule_ys) == 2 and rule_ys[1] - rule_ys[0] > 0.05 * image.height:
        return int(rule_ys[0]), int(rule_ys[1])
    return round(0.04 * image.height), round(0.96 * image.height)


def _column_gutter(image: Image.Image, skip_top: float = 0.0) -> int | None:
    """Return the x of the widest vertical whitespace run of a crop.

    A run strictly inside the central window (25%-70% of the width) counts as
    the column gutter; ``None`` when no run at least 1.5% of the width exists
    (full-measure, single-column band).  ``skip_top`` skips the first
    ``skip_top`` pixels of the crop (skips the full-measure Desunt line).
    """
    import numpy as np

    arr = np.array(image.convert("L"))
    if skip_top:
        arr = arr[int(skip_top):]
    ink = arr < 128
    filled = ink.any(axis=0)
    w = len(filled)
    lo, hi = round(0.25 * w), round(0.70 * w)
    best_x, best_run = None, 0
    run = 0
    for x in range(lo, hi):
        if not filled[x]:
            run += 1
        else:
            if run > best_run:
                best_x, best_run = x, run
            run = 0
    if run > best_run:
        best_x, best_run = hi, run
    if best_x is None or best_run < 0.015 * w:
        return None
    return best_x - best_run // 2


def parse_column_json(text: str) -> dict[str, str] | None:
    """Extract the ``{desunt, left, right}`` columns from a VLM reply.

    Accepts the JSON directly or a fenced/loose variant; returns ``None``
    when the reply is not column-shaped (caller falls back to plain text).
    """
    cleaned = clean_response(text)
    # try strict JSON first
    try:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            obj = json.loads(cleaned[start:end + 1])
            if isinstance(obj, dict) and \
                    any(k in obj for k in ("desunt", "left", "right")):
                return {k: str(obj.get(k) or "") for k in
                        ("desunt", "left", "right")}
    except json.JSONDecodeError:
        pass
    # lenient: string-valued keys in order
    out: dict[str, str] = {}
    parts = re.split(r'"(desunt|left|right)"\s*:\s*"', cleaned)
    if len(parts) < 3:
        return None
    for i in range(1, len(parts), 2):
        value = parts[i + 1].split('"')[0]
        out[parts[i]] = value.replace("\\n", "\n").replace("\\\\", "\\")
    return out or None


def regions_from_response(band: Image.Image, y0: int, y1: int,
                          page_size: tuple[int, int],
                          response: str) -> list[dict]:
    """Build the stage-1 region list for one page from a VLM reply.

    The layout reproduces the region contract stage 2 expects: the full-width
    Desunt witness-line region anchor, then the left column ending before the
    gutter, then the right column opening after it (a single full-measure
    region below the anchor when the band has no gutter).

    Args:
        band: the apparatus-band crop (native pixels) sent to the VLM.
        y0, y1: the band's position on the native page (pixels).
        page_size: ``(width, height)`` of the native page (norm-1000 frame).
        response: the raw VLM text.
    """
    w, h = band.size  # w == page width; h == band height
    page_w, page_h = page_size

    desunt = left = right = ""
    columns = parse_column_json(response)
    if columns is not None:
        desunt, left, right = (columns.get("desunt", ""),
                               columns.get("left", ""),
                               columns.get("right", ""))
    else:
        left = clean_response(response)

    # first-line height of the band (the Desunt line) from the ink mask
    import numpy as np

    band_arr = np.array(band.convert("L"))
    rows_ink = (band_arr < 128).any(axis=1)
    line_h = 0
    seen_ink = False
    for y, filled in enumerate(rows_ink):
        if filled:
            seen_ink = True
        elif seen_ink:
            line_h = y
            break
    if not line_h:
        line_h = h // max(1, band_arr.shape[0] // 64)

    def norm(x0: int, ya: int, x1: int, yb: int) -> list[int]:
        """bbox on the native page -> norm-1000 frame of the whole page."""
        return [round(x0 * 1000 / page_w), round(ya * 1000 / page_h),
                round(x1 * 1000 / page_w), round(yb * 1000 / page_h)]

    regions: list[dict] = []
    if desunt:
        bb_bottom = min(y1, y0 + line_h + 2)
        regions.append({"index": len(regions), "label": "witness_line",
                       "content": desunt,
                       "bbox_norm_1000": norm(0, y0, page_w, bb_bottom)})
    body_top = y0 + (line_h + 2 if desunt else 0)
    gutter = _column_gutter(band, skip_top=line_h + 4)
    if gutter is not None and left and right:
        regions.append({"index": len(regions), "label": "apparatus",
                        "content": left,
                        "bbox_norm_1000": norm(0, body_top, gutter, y1)})
        regions.append({"index": len(regions), "label": "apparatus",
                        "content": right,
                        "bbox_norm_1000": norm(gutter, body_top, page_w, y1)})
    elif left or right:
        content = left or right
        regions.append({"index": len(regions), "label": "paragraph",
                        "content": content,
                        "bbox_norm_1000": norm(0, body_top, page_w, y1)})
    if not regions:  # empty reply: keep a placeholder so the page is visible
        regions.append({"index": 0, "label": "apparatus_band",
                        "content": clean_response(response),
                        "bbox_norm_1000": norm(0, y0, page_w, y1)})
    return regions


def clean_response(text: str) -> str:
    """Strip the most common assistant chit-chat wrappers from a VLM reply."""
    t = text.strip()
    m = re.match(r"^```[\w-]*\s*\n?(.*?)\n?```$", t, re.DOTALL)
    if m:  # a single fenced block: keep exactly its content
        return m.group(1).strip()
    return t.replace("```", "").strip()


@dataclass
class Stage1Artifact:
    """The stage-1 output consumed by stages 2-3.

    Attributes:
        model_pages: one list of region dicts per page.  Each region carries
            ``index``, ``label``, ``content`` (the VLM transcription) and
            ``bbox_norm_1000`` (bboxes normalized to a 1000x1000 frame).
        page_sizes: ``[[width, height], ...]`` in pixels, index-aligned with
            ``model_pages`` and with the native page renders -- the norm-1000
            bboxes resolve against these sizes.
        native_pages: the rendered native 330 PPI page paths (stage 2 crops
            the classifier inputs from these).
    """
    model_pages: List[List[dict]] = field(default_factory=list)
    page_sizes: List[List[int]] = field(default_factory=list)
    native_pages: List[Path] = field(default_factory=list)

    def save(self, work_dir: Path, stem: str) -> None:
        """Write the artifact as ``<stem>_model.json`` + ``<stem>_page_sizes.json``."""
        work_dir = Path(work_dir)
        (work_dir / f"{stem}_model.json").write_text(
            json.dumps(self.model_pages, ensure_ascii=False, indent=2),
            encoding="utf-8")
        (work_dir / f"{stem}_page_sizes.json").write_text(
            json.dumps(self.page_sizes), encoding="utf-8")

    @classmethod
    def load(cls, work_dir: Path, stem: str,
             native_pages: Optional[List[Path]] = None) -> "Stage1Artifact":
        """Load a previously saved artifact (resume / reuse a VLM run)."""
        work_dir = Path(work_dir)
        model_pages = json.loads(
            (work_dir / f"{stem}_model.json").read_text(encoding="utf-8"))
        page_sizes = json.loads(
            (work_dir / f"{stem}_page_sizes.json").read_text(encoding="utf-8"))
        return cls(model_pages=model_pages, page_sizes=page_sizes,
                   native_pages=list(native_pages or []))


def artifact_from_transcript(pdf_path: Path, transcript: str,
                             work_dir: Path) -> "Stage1Artifact":
    """Build a stage-1 artifact from an existing LaTeX-flavoured transcription.

    The "resume" path of the pipeline: reuse an archived VLM run (or a
    hand-corrected transcription) instead of calling a backend.  A
    page-marked transcript (``%% page N``) maps block N onto page N; an
    unmarked one is used for page 1 only.  Each page's Desunt line becomes
    the full-width witness-line anchor region, the rest the body region.
    """
    import re as _re

    native = render_pdf(Path(pdf_path), Path(work_dir) / "native")
    blocks: dict[int, str] = {}
    if "%% page" in transcript:
        parts = _re.split(r"(?m)^%%\s*page\s+(\d+)", transcript)
        for i in range(1, len(parts), 2):
            blocks[int(parts[i])] = parts[i + 1].strip()
    else:
        blocks[1] = transcript.strip()

    artifact = Stage1Artifact(native_pages=native)
    import PIL.Image  # local: keep the heavy import out of the module header

    for pi, p in enumerate(native, start=1):
        with PIL.Image.open(p) as im:
            size = list(im.size)
            y0, y1 = apparatus_band(im)
            band = im.crop((0, y0, im.width, y1))
        block = blocks.get(pi, "")
        desunt, rest = "", block
        for j, line in enumerate(block.splitlines()):
            if _re.sub(r"[^A-Za-z]", "", line).lower() \
                    .startswith(("desunt", "deest")):
                desunt = line
                rest = "\n".join(block.splitlines()[j + 1:]).strip()
                break
        response = json.dumps({"desunt": desunt, "left": rest, "right": ""})
        artifact.model_pages.append(
            regions_from_response(band, y0, y1, tuple(size), response))
        artifact.page_sizes.append(size)
    return artifact


def transcribe_page(image_path: Path, backend: ModelBackend,
                    prompt: Optional[str] = None) -> tuple[list, list]:
    """Transcribe one native page with ``backend``.

    ``prompt=None`` selects the canonical apparatus prompt (never send a
    null text part to the API).

    Returns:
        ``(regions, page_size)``: the page's region list in the stage-1
        contract (norm-1000 bboxes) and the native page size.
    """
    prompt = prompt if prompt is not None else APPARATUS_PROMPT
    with Image.open(image_path) as page:
        page.load()
        size = list(page.size)
        y0, y1 = apparatus_band(page)
        band = page.crop((0, y0, page.width, y1))
    response = backend.transcribe_page(band, prompt)
    regions = regions_from_response(band, y0, y1, tuple(size), response)
    return regions, size


def transcribe_pdf(pdf_path: Path, backend: ModelBackend,
                   work_dir: Path, prompt: Optional[str] = None,
                   pages: Optional[List[Path]] = None) -> Stage1Artifact:
    """Stage 1: render + transcribe every page of a PDF with ``backend``.

    Args:
        pdf_path: the scanned input PDF.
        backend: any configured :class:`~vulgata_scan.backends.ModelBackend`.
        work_dir: directory for the native renders.
        prompt: transcription prompt (``None`` = the canonical one).
        pages: pre-rendered native pages (skips rendering; index-aligned).

    Returns:
        The stage-1 artifact.
    """
    prompt = prompt if prompt is not None else APPARATUS_PROMPT
    pdf_path = Path(pdf_path)
    native = list(pages) if pages else render_pdf(pdf_path, Path(work_dir) / "native")
    artifact = Stage1Artifact(native_pages=native)
    for page_png in native:
        regions, size = transcribe_page(image_path=page_png, backend=backend,
                                         prompt=prompt)
        artifact.model_pages.append(regions)
        artifact.page_sizes.append(size)
        print(f"[stage1] {pdf_path.name} p{len(artifact.model_pages)}: "
              f"{len(regions)} regions, "
              f"{sum(len(r['content']) for r in regions)} chars transcribed")
    return artifact
