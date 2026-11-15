"""Offline integration test: stages 2-4 of the pipeline on the sample PDF.

Stage 1 is stubbed with the archived VLM transcription of the same corpus
pages (``artifact_from_transcript``), so the test exercises the glyph
classifier, the reconciliation and the grammar against the *real scans*,
without any model call.  Marked slow: it loads the CNN checkpoint.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_PDF = REPO_ROOT / "example" / "sample_genesis.pdf"
GLM_TRANSCRIPT = REPO_ROOT / "evaluation" / "raw_datasets" / "glm.txt"

sample_missing = not (SAMPLE_PDF.is_file() and GLM_TRANSCRIPT.is_file())


def _first_pages_transcript(n_pages: int = 4) -> str:
    text = GLM_TRANSCRIPT.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^%%\s*page\s+(\d+)", text)
    out = []
    for i in range(1, len(parts), 2):
        if int(parts[i]) <= n_pages:
            out += [f"%% page {parts[i]}", parts[i + 1].strip()]
    return "\n\n".join(out)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    if sample_missing:
        pytest.skip("sample corpus not present in the repository")
    from vulgata_scan.pipeline import run_pipeline
    from vulgata_scan.vlm.transcribe import artifact_from_transcript

    tmp = tmp_path_factory.mktemp("vscan_run")
    work = tmp / "work"
    artifact = artifact_from_transcript(SAMPLE_PDF,
                                        _first_pages_transcript(), work)
    return run_pipeline(SAMPLE_PDF, None, tmp / "parsed.json", work,
                        chapter="1", artifact=artifact)


@pytest.mark.skipif(sample_missing, reason="sample corpus not present")
def test_stage2_finds_and_classifies_tokens(run):
    assert run.report, "stage-2 report must not be empty"
    tokens = [t for r in run.report for t in r["tokens"]]
    families = {t["prediction"]["family"] for t in tokens}
    assert "manuscript_siglum" in families
    assert "background" in families


@pytest.mark.skipif(sample_missing, reason="sample corpus not present")
def test_stage3_reconciles(run):
    corrected = json.loads(
        Path(run.outputs["stage3_corrected_latex"]).read_text(encoding="utf-8"))
    assert corrected, "corrected transcription must not be empty"
    text = json.dumps(corrected, ensure_ascii=False)
    assert "\\Sigma" in text or "\\Theta" in text


@pytest.mark.skipif(sample_missing, reason="sample corpus not present")
def test_grammar_parses_pipe_output(run):
    assert run.entries, "the grammar must produce verse entries"
    first = run.entries[0]
    assert first["chapter"] == "1"
    assert {"chapter", "verse", "apparatus"} <= set(first)
    types = [a["variant_type"] for e in run.entries for a in e["apparatus"]]
    assert "substitution" in types
    assert set(types) <= {"substitution", "addition", "omission", "note",
                         "positive_apparatus", "in_text_support",
                         "transposition"}
