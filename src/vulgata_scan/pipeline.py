"""End-to-end pipeline: PDF -> VLM -> classifier -> reconciliation -> grammar.

The four paper stages, wired together:

1. :mod:`vulgata_scan.vlm` -- a :class:`~vulgata_scan.backends.ModelBackend`
   transcribes the apparatus band of every page (model-agnostic);
2. :mod:`vulgata_scan.classifier` -- zoning, segmentation and per-glyph CNN
   decoding of the sigla crops;
3. :mod:`vulgata_scan.reconcile` -- the VLM reading and the classifier reading
   are reconciled (write-back) into one consensus transcription;
4. :mod:`vulgata_scan.grammar` -- the reconciled, LaTeX-flavoured text is
   parsed by the formal grammar into a flat JSON model of readings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .backends import ModelBackend
from .classifier.stage import run_chunk
from .grammar.parser import parse_apparatus
from .reconcile.writeback import reconcile_chunk
from .vlm.transcribe import Stage1Artifact, transcribe_pdf
from .vlm.rendering import render_pdf


def corrected_transcription(model_pages: list, corrected: list) -> str:
    """Flatten the reconciled regions into the grammar's plain-text input.

    Keeps the ``%% page N`` markers of the paper's raw datasets; the grammar's
    preprocessor drops those lines, but they keep the artifact comparable to
    the released corpus files.
    """
    chunks: List[str] = []
    for pi, (page_in, page_corr) in enumerate(zip(model_pages, corrected),
                                              start=1):
        corr_by_index = {r["index"]: r for r in page_corr}
        lines = [f"%% page {pi}"]
        for region in page_in:
            hit = corr_by_index.get(region["index"])
            lines.append(hit.get("content", "") if hit is not None
                         else region.get("content", ""))
        chunks.append("\n\n".join(lines))
    return "\n\n".join(chunks)


@dataclass
class PipelineResult:
    """Everything one pipeline run produces.

    Attributes:
        entries: the grammar's parsed readings (the parsed JSON model,
            identical in shape to the released dataset).
        artifact: the stage-1 artifact (VLM transcription + layout).
        report: the stage-2 region/token report (classifier output).
        grammar_input: the reconciled LaTeX-flavoured apparatus text that
            was fed to the grammar (pages marked ``%% page N``).
        outputs: paths of every file written under ``work_dir``/``output``.
    """
    entries: list
    artifact: Stage1Artifact
    report: list
    grammar_input: str
    outputs: dict


def run_pipeline(pdf_path: Path, backend: ModelBackend,
                 output: Path, work_dir: Path, *,
                 chapter: str = "1",
                 prompt: Optional[str] = None,
                 align: str = "global",
                 geometry: bool = True,
                 debug: bool = False,
                 force_inventory: bool = False,
                 artifact: Optional[Stage1Artifact] = None) -> PipelineResult:
    """Run all four stages on one PDF and write the parsed apparatus JSON.

    Args:
        pdf_path: scanned input PDF.
        backend: any configured backend (``mistral-api`` or ``local``).
        output: path of the parsed JSON output file.
        work_dir: scratch dir receiving all intermediate stage files.
        chapter: chapter number the apparatus starts at (Genesis px. 1).
        prompt: overrides the canonical apparatus prompt.
        align, geometry: reconciliation-route knobs (paper defaults).
        debug: write stage-2 annotated debug PNGs.
        force_inventory: replace expanded witnesses absent from the
            inventory with ``?`` (see the grammar's evaluation mode).
        artifact: pre-existing stage-1 artifact to reuse (resume path); by
            default stage 1 runs against ``backend``.

    Returns:
        A :class:`PipelineResult` with the parsed entries and every
        intermediate artifact.
    """
    pdf_path = Path(pdf_path)
    output = Path(output)
    work_dir = Path(work_dir)
    stem = pdf_path.stem

    # ---- Stage 1: VLM transcription (model-agnostic backend) --------------
    if artifact is None:
        artifact = transcribe_pdf(pdf_path, backend, work_dir, prompt=prompt)
    else:
        if not artifact.native_pages:
            artifact.native_pages = render_pdf(pdf_path, work_dir / "native")
    artifact.save(work_dir, stem)

    # ---- Stage 2: zoning + segmentation + glyph classification ------------
    report = run_chunk(artifact.model_pages, artifact.page_sizes,
                       artifact.native_pages, work_dir / "stage2",
                       debug=debug)
    (work_dir / f"{stem}_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Stage 3: reconciliation (write-back of the CNN readings) ---------
    reconciled = reconcile_chunk(artifact.model_pages, report,
                                 out_dir=work_dir / "stage3", stem=stem,
                                 align=align, geometry=geometry,
                                 report_stem=stem)
    corrected_pages = json.loads(Path(reconciled["corrected_latex_json"])
                                 .read_text(encoding="utf-8"))

    # ---- Stage 4: formal-grammar parse of the corrected transcription ----
    raw = corrected_transcription(artifact.model_pages, corrected_pages)
    grammar_input_path = work_dir / f"{stem}_apparatus.md"
    grammar_input_path.write_text(raw, encoding="utf-8")
    entries = parse_apparatus(raw, chapter=chapter,
                              force_inventory=force_inventory)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    outputs = {
        "parsed_json": output,
        "stage1_model_json": work_dir / f"{stem}_model.json",
        "stage1_page_sizes": work_dir / f"{stem}_page_sizes.json",
        "stage2_report": work_dir / f"{stem}_report.json",
        "stage2_zones_png_dir": work_dir / "stage2" / "debug",
        "stage3_corrected_latex": reconciled["corrected_latex_json"],
        "stage3_corrected_unicode": reconciled["corrected_unicode_md"],
        "stage3_writeback": reconciled["writeback_json"],
        "grammar_input": grammar_input_path,
    }
    return PipelineResult(entries=entries, artifact=artifact, report=report,
                          grammar_input=raw, outputs=outputs)
