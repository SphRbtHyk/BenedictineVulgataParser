# vulgata-scan — parsing the critical apparatus of the Benedictine Vulgate 📜✨

`vulgata-scan` is the reference implementation of the paper *From Printed
Critical Apparatus to Structured Data: A Case Study on the Benedictine
Vulgate*: it turns **scanned pages of the critical apparatus of the Vulgate
edition of Genesis** (*Biblia Sacra iuxta latinam vulgatam versionem*,
the "Roman" or "San Girolamo" Vulgate) into **structured JSON** of variant
readings, witnesses and citations.

This folder (`BenedictineVulgataParser/`) is a **self-contained,
installable consolidation** of the paper's codebase: every stage of the
pipeline, the frozen glyph classifier, the evaluation tooling and the
gold/silver datasets live together, with clean module boundaries, type
hints and a CLI. The legacy folders of the repository
(`pipeline_final_code/`, `parser/`, `evaluation/`, `dataset/`) are kept
untouched for provenance; this package incorporates their logic.

---

## 1. Library overview 🔭

The pipeline operates in **four stages** 🚂💨 — all aboard:

```
                    ┌──────────────────────────────────────────────┐
 scanned PDF  ──►   │ 1. VLM transcription        vulgata_scan.vlm │
 (apparatus   │    │    any vision-language model, pluggable       │
  pages)      │    │    ModelBackend (Mistral API | local server)  │
              └──►  │    → LaTeX-flavoured transcription + layout  │
                    └────────────────┬─────────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────────┐
                    │ 2. Glyph classification  vulgata_scan.classifier│
                    │    zoning → segmentation → CNN per siglum    │
                    │    → structured glyph readings (checkpoint)   │
                    └────────────────┬─────────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────────┐
                    │ 3. Reconciliation       vulgata_scan.reconcile│
                    │    alignment + write-back of the two         │
                    │    readings into one consensus transcription │
                    └────────────────┬─────────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────────┐
                    │ 4. Grammar parsing       vulgata_scan.grammar│
                    │    pyparsing formal grammar → flat JSON       │
                    │    model of readings (dataset format)        │
                    └────────────────┬─────────────────────────────┘
                                     ▼
                     parsed apparatus JSON 🎉
                     (verse, variant_type,
                     reading, witnesses[], missing[], citations…)
```

### Architecture 🏛️

| Module | Paper stage | Contents |
|---|---|---|
| `vulgata_scan.backends` | 1 | `ModelBackend` abstraction + the two reference backends: `MistralApiBackend` (Mistral API) and `LocalBackend` (any OpenAI-compatible local server or served checkpoint) |
| `vulgata_scan.vlm` | 1 | page rendering (330 PPI, background normalization, scan-border trim), apparatus-band detection (printed divider rules; robust to both raw gray renders and normalized white pages — see `classifier.segment.paper_level`), the canonical transcription prompt, the stage-1 artifact |
| `vulgata_scan.classifier` | 2 | the frozen multi-head CNN (torch), two-pass decoding, geometric segmentation, apparatus zoning, the stage-2 driver |
| `vulgata_scan.reconcile` | 3 | the write-back: sigla-span extraction, global alignment, vetoes, per-region correction drivers |
| `vulgata_scan.grammar` | 4 | the formal grammar of the apparatus and the `parse_apparatus()` entry point |
| `vulgata_scan.pipeline` | all | `run_pipeline()`: the four stages wired end-to-end |
| `vulgata_scan.cli` | — | the `vulgata-scan` command |
| `vulgata_scan.dataset` | — | loaders + pretty-printer for the gold/silver datasets |
| `vulgata_scan.evaluation` | — | the paper's metrics (ported `evaluate_parser`, `ablation_study`, `apparatus_cer`) |

The stage-1 **contract** between the VLM stage and stages 2-3 is one JSON
artifact (`<stem>_model.json` + `<stem>_page_sizes.json`: pages of regions
with `bbox_norm_1000` and LaTeX-flavoured `content`). Every backend produces
exactly that artifact — this is what makes the pipeline model-agnostic: the
backend can be the Mistral API, mlx-vlm, llama.cpp, vLLM, LM Studio… with
no change to stages 2-4.

### Backends (model-agnostic by design) 🔌

- **Mistral API (reference)** 🛰️ — `--backend mistral-api`; endpoint
  (default `https://api.mistral.ai/v1`), model (default
  `mistral-medium-latest`; any **image-capable** model the account serves —
  the Pixtral models used during the paper's runs are retired, and text-only
  models such as `mistral-large` are rejected with HTTP 422 because they
  cannot take the page image) and API key are configuration, never
  hardcoded. The key is read from an environment variable whose **name** you
  choose (`--api-key-env MISTRAL_API_KEY` by default), avoid putting secret
  material in command lines.
- **Local** 🖥️ — `--backend local --endpoint http://127.0.0.1:8080/v1
  --model <served-model>`: any OpenAI-compatible local server (the GLM-OCR
  path on Apple Silicon via `mlx-vlm`, llama.cpp's `server`, vLLM, LM
  Studio, Ollama's OpenAI endpoint…). Connection details are given at call
  time; a local checkpoint is used by serving it with one of those
  servers.

### Installation 📦

Python ≥ 3.10 (the paper's experiments used 3.12):

```bash
cd BenedictineVulgataParser
pip install -e .            # installs the vulgata_scan package + `vulgata-scan`
pip install -e ".[dev]"     # + pytest
```

Dependencies are pinned in `pyproject.toml` to the environment of the
paper's experiments (torch, Pillow, numpy, scipy, pyparsing, PyYAML,
requests, pymupdf). The frozen classifier checkpoint
(`classifier/checkpoints/best.pt`, 3 MB) and its `taxonomy.yaml` ship inside
the package.

### CLI usage ⌨️

```bash
vulgata-scan parse <input-file> \
  --backend mistral-api           # or: local \
  --endpoint <url> --model <model-name> --api-key-env MISTRAL_API_KEY \
  --output <output-file>
```

The canonical test input is `example/sample_genesis.pdf` 🧪 (4 scanned
apparatus pages of Genesis):

```bash
# with the Mistral API (reads the key from $MISTRAL_API_KEY)
# --model must be an image-capable model (default: mistral-medium-latest);
# text-only models (e.g. mistral-large) or retired ones (pixtral-*) fail
export MISTRAL_API_KEY=...
vulgata-scan parse ../example/sample_genesis.pdf \
    --backend mistral-api --model mistral-medium-latest \
    --output parsed_genesis.json

# with a local OpenAI-compatible server
vulgata-scan parse ../example/sample_genesis.pdf \
    --backend local --endpoint http://127.0.0.1:8080/v1 --model GLM-OCR \
    --output parsed_genesis.json
```

Useful options: `--chapter` (chapter at which the apparatus starts),
`--workdir` (keeps every intermediate artifact: stage-1 model JSON,
stage-2 region/token report + zones PNGs, stage-3 corrected
transcriptions, stage-4 grammar input), `--transcript FILE.md` (skip the
VLM call and reuse an existing LaTeX-flavoured transcription — the
regression route used in the tests), `--debug` (annotated segmentation
PNGs), `--align difflib` / `--no-geometry` (ablations).

**Outputs.** 🎁 The command writes the **parsed apparatus JSON** (the flat
reading model, same schema as the released datasets) and, under the work
dir, the stage-3 consensus transcription in the two native notations:

| File | Content |
|---|---|
| `<out>.json` | parsed apparatus — the final structured output |
| `<stem>_corrected_latex.json` | consensus transcription, LaTeX notation — the pipeline's canonical intermediate |
| `<stem>_corrected_unicode.md` | consensus transcription, human-readable Unicode (Σᵀ, 𝔞𝔤𝔯…) |
| `<stem>_report.json` | stage-2 glyphs: per-token CNN readings with bounding boxes |
| `<stem>_model.json`, `<stem>_page_sizes.json` | stage-1 VLM artifact |

**Library usage** mirrors the stages:

```python
from vulgata_scan import build_backend, run_pipeline, parse_apparatus

backend = build_backend("mistral-api", model="mistral-medium-latest")
result = run_pipeline(pdf, backend, out="parsed.json", work_dir="work/",
                      chapter="1")
result.entries  # the parsed apparatus, same schema as the datasets
```

---

## 2. Reproduction of results 🔬

The evaluation corpus of the paper is **the first 25 scanned pages of
Genesis** (front matter + Genesis 1: the pages carrying the hand-annotated
ground truth), hand-collated into a LaTeX-flavoured gold transcription. The
pipeline side is the archived full-pipeline output over the same pages.

Which backend/model reproduces the numbers: the staged results were
produced with the paper's reference VLM stage on Apple Silicon (the GLM-OCR
model served locally via mlx-vlm); **any backend works** for a fresh run —
the paper's claim — but exact numbers below correspond to the archived
corpus in `evaluation/`, not to a fresh VLM roll (VLM inference is not
deterministic).

Run everything with one script (from the repository root, package
installed):

```bash
bash BenedictineVulgataParser/scripts/reproduce_paper_results.sh
```

or command by command:

```bash
# full pipeline vs gold: census + paper metrics [1]-[4] + global CER [G]
python -m vulgata_scan.evaluation.evaluate_parser \
    --gt evaluation/parsed_datasets/ground_truth.json \
    --pred evaluation/parsed_datasets/glm_classifier.json \
    --gt-text evaluation/raw_datasets/ground_truth.txt \
    --pred-text evaluation/raw_datasets/glm_classifier.txt \
    --per-verse

# VLM-only (stage 1 alone) vs gold — the ablation the paper reports
python -m vulgata_scan.evaluation.evaluate_parser \
    --gt evaluation/parsed_datasets/ground_truth.json \
    --pred evaluation/parsed_datasets/glm.json \
    --gt-text evaluation/raw_datasets/ground_truth.txt \
    --pred-text evaluation/raw_datasets/glm.txt

# raw character error rate of the consensus transcription
python -m vulgata_scan.evaluation.apparatus_cer \
    --gt evaluation/raw_datasets/ground_truth.txt \
    --pred evaluation/raw_datasets/glm_classifier.txt --per-page

# three-way ablation (gold / naive VLM+CNN / full VLM+CNN)
python -m vulgata_scan.evaluation.ablation_study \
    --gold evaluation/raw_datasets/ground_truth.txt \
    --vlm evaluation/pipeline_output/output_glm_classifier_naive.md \
    --pipe evaluation/pipeline_output/output_glm_classifier.md
```

**Expected outputs** 📊 (full pipeline, `glm_classifier`, against the gold;
produced by the commands above):

| Paper metric | Value |
|---|---|
| ground truth census | 106 verses, 4 chapters (Genesis 1-4), 3281 witness sigla |
| [1] structure detection | entry precision **0.943**, recall **0.951**, F1 **0.947** |
| [2] witness extraction | sigla exact match **0.908**, precision **0.912**, F1 **0.910**; strict group match 0.854 (0.898 on aligned entries; F1 0.959) |
| [3] reading transcription | reading CER **0.129** (exact **0.894**), in-text CER 0.136, anchor CER 0.056, over-3-fields CER 0.127 |
| [4] agreement computation | mean per-witness Spearman over the witness-agreement grids **0.914** |
| [G] global CER (raw OCR vs annotated GT) | **0.0476** (27,149 chars, 1,292 edits) |

For comparison, **VLM only** (no classifier, no reconciliation):
entry precision 0.532 / recall 0.900 / F1 0.668; sigla F1 0.463 — the
numbers that motivate the glyph-classifier stage.

To regenerate the pipeline output itself (fresh VLM roll, numbers will
differ slightly), run the CLI over the corpus chunks and re-parse with the
grammar; the `--transcript` route reproduces stages 2-4 deterministically
from any LaTeX-flavoured transcription, and
`tests/test_pipeline_stages.py` exercises exactly that path on
`example/sample_genesis.pdf`.

---

## 3. The dataset — a major contribution of the paper 💎

The release ships **two datasets** of the same flat reading model — the
JSON schema the grammar emits and the whole pipeline evaluates against:

```json
[{
  "chapter": "1", "verse": "3",
  "apparatus": [{
      "variant_type": "substitution",
      "reference_reading": {"reading": "creavit", "witnesses": []},
      "alternative_reading": {"reading": "creabit", "witnesses": ["X", "B"]},
      "missing": ["G^{*}", "\\Sigma^T"], "versions": [...], "citations": [...]
  }]
}]
```

### Gold 🏆

- **`dataset/gold_data/gold_data.json`** — the **hand-verified** parsed
  apparatus of **Genesis 1-4** (chapters 1-4): **106 verse entries**,
  **936 apparatus units**, 141 distinct atomic witnesses.
- How it was built: the printed apparatus of the first 25 scanned pages
  was transcribed by hand into the LaTeX-flavoured notation
  (`evaluation/raw_datasets/ground_truth.txt`), reviewed against the
  printed edition (`ground_truth/reviewed_ground_truth/`), and parsed with
  the grammar; every verse-level entry and witness siglum was then
  audited field by field.
- Role: the **evaluation gold** of every metric in section 2 😇.

### Silver 🤖

- **`dataset/silver_data/parsed_data/genesis_p1_249.json`** — the
  **machine-produced** parsed apparatus of the **whole book** (**Genesis,
  ~249 scanned pages**, all 50 chapters): **357 verse entries**,
  **25,194 apparatus units**, 254 witnesses.
- How it was built: the four-stage pipeline (VLM → classifier →
  reconciliation → grammar) over the book, page-chunk by page-chunk;
  per-chunk raw artifacts are kept under
  `dataset/silver_data/raw_data/` (per 10-page chunk:
  `<chunk>_result_global.json` stage-2 report,
  `<chunk>_corrected_latex.json` consensus transcription,
  `<chunk>_corrected_unicode.md` human view, `<chunk>_writeback.json`
  per-region diff + review list).
- Differences: **gold = ground truth, reviewed by hand, Genesis 1-4;
  silver = the pipeline's own output over the entire book, no exhaustive
  human verification.** Never use silver to evaluate the pipeline that
  produced it.
- Role: the stand-against-the-world dataset that lets other pipelines
  (or humanists) be evaluated over the full apparatus, and the reference
  output any reproduction can be compared against.

### Loading and using the datasets 🛠️

`vulgata_scan.dataset` (small copies of both files also ship inside the
package, under `BenedictineVulgataParser/data/`, so the examples work
without the repository):

```python
from vulgata_scan.dataset import load_gold, load_silver, census, pretty_entry

gold = load_gold()                                    # 106 entries
silver = load_silver(path="dataset/silver_data/parsed_data/genesis_p1_249.json")
print(census(gold))       # {'verse_entries': 106, 'apparatus_units': 936, ...}
print(pretty_entry(gold[1]))
```

```bash
vulgata-scan dataset --samples 3            # census + pretty-printed entries
python scripts/demo_dataset.py --samples 2  # same, as a stand-alone demo
```

### Visualizing the datasets 🖼️

The release includes the **apparatus reader**, the interactive
visualization used to explore both datasets (taken from the paper's
`dataset/display.html` without changes):

- open **`dataset_visualization/apparatus_reader.html`** in a browser. It
  loads `apparatus.json` (a copy of the gold dataset) from its own folder
  by default, loads any other JSON via **?src=…** (try
  `apparatus_reader.html?src=genesis_p1_249.json` for the silver dataset),
  or via the **Open file** button;
- per verse: every unit with its reading and its witness sigla,
  color-coded with the three witness states of Quentin's method
  (**attests / lacunose (desunt) / silent**), the **lacuna map** view over
  the whole chapter, full-text search over readings, sigla and citations,
  an export of the current view, and a dark mode;
- the reader is a single self-contained HTML file (no build, no server):
  it can be opened directly from the file system.

A table of contents of the datasets at a glance 👀:

| | gold | silver |
|---|---|---|
| origin | hand-verified against the printed edition | four-stage pipeline output |
| extent | Genesis 1-4 (25 scanned pages) | Genesis, ~249 scanned pages (ch. 1-50) |
| verse entries | 106 | 357 |
| apparatus units | 936 | 25,194 |
| atomic witnesses | 141 | 254 |
| raw per-chunk artifacts | `evaluation/raw_datasets/ground_truth.txt`, `ground_truth/reviewed_ground_truth/` | `dataset/silver_data/raw_data/*` (result, corrected LaTeX, corrected Unicode, writeback) |
| packaged copy | `data/gold_data.json` | `data/genesis_p1_249.json` |

---

## Tests 🧪

```bash
pytest -m "not slow"   # grammar, backends (mock VLM server), dataset: fast
pytest                  # + offline pipeline integration on example/sample_genesis.pdf
```

(19 tests; the slow marker loads the CNN checkpoint and processes the real
sample scans, ~1.5 min on an M-series laptop.)

## Notes on this consolidation — deviations from the original layout 📝

- The legacy repository keeps its own untouched copy of everything; this
  package is the consolidated, installable artifact of the paper.
- `pipeline/grammar/` in the legacy repo holds only docstring stubs; the
  live grammar implementation it advertises is the (partly commented-out)
  `parser/parse_apparatus.py`, whose live part is ported unchanged here to
  `vulgata_scan/grammar/parser.py`.
- The canonical test input is `example/sample_genesis.pdf` (a PDF, not a
  `.py` file as the task list suggested).
- The GLM-OCR SDK path of the original stage 1 (`pipeline_final_code/glm/`,
  mlx-vlm-specific) is subsumed by the model-agnostic `vlm` module:
  the same OpenAI-compatible wire protocol covers both the reference
  Mistral backend and local servers such as mlx-vlm's. Any other provider
  that speaks the OpenAI chat-completions protocol with image parts works
  via the `local` backend by pointing `--endpoint` at it.
