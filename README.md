# Parser for the critical apparatus of the Benedictine Vulgate 📜✨

<p align="center">
  <img src="public/logo.jpg" alt="Parser logo" width="240">
</p>

This is the companion GitHub for the paper *From Printed
Critical Apparatus to Structured Data: A Case Study on the Benedictine
Vulgate*, now accepted at the *2027 Computational Humanities Conference* 🎉.
The goal of the paper is to transform **scanned pages of the critical apparatus of the Vulgate
edition of Genesis** (*Biblia Sacra iuxta latinam vulgatam versionem*,
the "Roman" or "San Girolamo" Vulgate) into **structured JSON** of variant
readings, witnesses and citations, to perform **further analysis** regarding the transmission of this text.

---

## 1. Pipeline overview 🔭

The pipeline operates in **four stages** 🚂💨:

```
                    ┌──────────────────────────────────────────────┐
 scanned PDF  ──►   │ 1. VLM transcription        vulgata_scan.vlm │
 (apparatus   │     │    any vision-language model, pluggable      │
  pages)      │     │    ModelBackend (Mistral API | local server) │
              └──►  │   
                    └────────────────┬─────────────────────────────┘
                                     ▼
                    ┌────────────────────────────────────────────────┐
                    │ 2. Glyph classification vulgata_scan.classifier│
                    │    zoning → segmentation → CNN per siglum      │
                    │    → structured glyph readings (checkpoint)    │
                    └────────────────┬───────────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────────┐
                    │ 3. Reconciliation      vulgata_scan.reconcile│
                    │    alignment + write-back of the two         │
                    │    readings into one consensus transcription │
                    └────────────────┬─────────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────────┐
                    │ 4. Grammar parsing       vulgata_scan.grammar│
                    │    pyparsing formal grammar → flat JSON      │
                    │    model of readings (dataset format)        │
                    └────────────────┬─────────────────────────────┘
                                     ▼
                     parsed apparatus JSON 🎉
                     (verse, variant_type,
                     reading, witnesses[], missing[], citations…)
```

### Selecting vision back-ends (model-agnostic by design) 🔌
The pipeline is relatively agnostic to any vision back-end:

- **Mistral API (reference)** 🛰️ — `--backend mistral-api`; endpoint
  (default `https://api.mistral.ai/v1`), model (default
  `mistral-medium-latest`; The key is read from an environment variable whose **name** you
  choose (`--api-key-env MISTRAL_API_KEY` by default).
- **Local** 🖥️ — `--backend local --endpoint http://127.0.0.1:8080/v1
  --model <served-model>`: any OpenAI-compatible local server running on your machine.

### Installation 📦

Python ≥ 3.10 (the paper's results ran 3.12, but the library should be compatible).
After downloading the source code from GitHub:

```bash
cd BenedictineVulgataParser
pip install -e .
pip install -e ".[dev]"     # possibility to run the unittests through pytest
```

### CLI usage ⌨️

```bash
vulgata-scan parse <input-file> \
  --backend mistral-api --api-key-env MISTRAL_API_KEY
  # or: local \
  --endpoint <url> --model <model-name> \
  --output <output-file>
```

To test your proper installation, run `example/sample_genesis.pdf` 🧪 (4 scanned
apparatus pages of Genesis):

```bash
# with the Mistral API (reads the key from $MISTRAL_API_KEY)
# --model must be an image-capable model (default: mistral-medium-latest);
# text-only models (e.g. mistral-large) or retired ones (pixtral-*) fail
export MISTRAL_API_KEY=...
vulgata-scan parse example/sample_genesis.pdf \
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
PNGs).

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

If you want to run it directly in Python:

```python
from vulgata_scan import build_backend, run_pipeline, parse_apparatus

backend = build_backend("mistral-api", model="mistral-medium-latest")
result = run_pipeline(pdf, backend, out="parsed.json", work_dir="work/",
                      chapter="1")
result.entries  # the parsed apparatus
```

---

## 2. Reproduction of results 🔬

The evaluation corpus of the paper is **the first 25 scanned pages of
Genesis** (front matter + Genesis 1: the pages carrying the hand-annotated
ground truth), hand-annotated.

The paper results were produced with the VLM GLM-OCR model served locally via mlx-vlm;
**Any backend works** for a new run, but keep in mind that VLM inference is not
deterministic and you might not be able to reproduce exactly the same results!

To run the results of the paper in one script:

```bash
bash BenedictineVulgataParser/scripts/reproduce_paper_results.sh
```
which should return the following outputs** 📊:

| Paper metric | Value |
|---|---|
| ground truth census | 106 verses, 4 chapters (Genesis 1-4), 3281 witness sigla |
| [1] structure detection | entry precision **0.943**, recall **0.951**, F1 **0.947** |
| [2] witness extraction | sigla exact match **0.908**, precision **0.912**, F1 **0.910**; strict group match 0.854 (0.898 on aligned entries; F1 0.959) |
| [3] reading transcription | reading CER **0.129** (exact **0.894**), in-text CER 0.136, anchor CER 0.056, over-3-fields CER 0.127 |
| [4] agreement computation | mean per-witness Spearman over the witness-agreement grids **0.914** |
| [G] global CER (raw OCR vs annotated GT) | **0.0476** (27,149 chars, 1,292 edits) |

---

## 3. Using our datasets 💎

The release ships **two datasets** for the study of the transmission of the Vulgata:


### Gold 🏆

**`dataset/gold_data/gold_data.json`** contains the **hand-verified** parsed apparatus of **Genesis 1-4** (chapters 1-4, first 25 scanned pages): **106 verse entries**, **936 apparatus units**, 141 distinct atomic witnesses.

### Silver 🤖

- **`dataset/silver_data/parsed_data/genesis_p1_249.json`** — the
  **machine-produced** parsed apparatus of the **whole book** (**Genesis,
  ~249 scanned pages**, all 50 chapters).
  **This dataset will be regularly updated as the quality of our method improves and as the project advances, stay-tuned!** 😇


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

The release includes the **apparatus reader**, an interactive
visualization used to explore both datasets (taken from the paper's
`dataset/display.html` without changes. To use it, open **`dataset_visualization/apparatus_reader.html`** in a browser. It
  loads `apparatus.json` (a copy of the gold dataset) from its own folder
  by default but can load anyfile respecting the (*i.e.* the silver dataset) the **Open file** button.

---

## Tests 🧪

```bash
pytest -m "not slow"   # grammar, backends (mock VLM server), dataset: fast
pytest                  # + offline pipeline integration on example/sample_genesis.pdf
```

## Quote us

You can quote this paper as:

```
@inproceedings{whatever_you_want,
  title     = {From Printed Critical Apparatus to Structured Data: A Case Study on the Benedictine Vulgate},
  author    = {Robert-Hayek, Sophie and Chen, Siying},
  booktitle = {Proceedings of the Computational Humanities Research Conference (CHR 2027)},
  year      = {2027},
  series    = {Anthology of Computer and the Humanities},
}
```

You can already find the pre-print in public/preprint.pdf 🤓.
