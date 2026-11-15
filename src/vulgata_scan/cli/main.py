"""``vulgata-scan`` command-line interface.

Usage::

    vulgata-scan parse <input-file> \\
        --backend mistral-api        # or: local \\
        --endpoint <url> --model <model-name> --api-key-env MISTRAL_API_KEY \\
        --output <output-file>

Subcommands:

- ``parse``    -- run the full four-stage pipeline on one PDF (or, with
  ``--transcript``, on an existing stage-1 transcription artifact);
- ``dataset``  -- explore the packaged gold/silver datasets (census +
  pretty-printed samples).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parse_parser(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("input", help="input PDF (scanned pages of the edition)")
    ap.add_argument("--backend", default="mistral-api",
                    choices=["mistral-api", "local"],
                    help="model backend: 'mistral-api' (reference, any Mistral "
                         "vision model) or 'local' (any OpenAI-compatible "
                         "local server / served checkpoint)")
    ap.add_argument("--endpoint", default=None,
                    help="backend base URL, e.g. https://api.mistral.ai/v1 or "
                         "http://127.0.0.1:8080/v1 (default: the Mistral API "
                         "for mistral-api; required for local)")
    ap.add_argument("--model", default=None,
                    help="model name as known to the backend; must be "
                         "image-capable for the VLM stage (default: "
                         "mistral-medium-latest for mistral-api; the Pixtral "
                         "models are retired)")
    ap.add_argument("--api-key-env", default="MISTRAL_API_KEY",
                    help="environment variable holding the API key")
    ap.add_argument("--api-key", default=None,
                    help="explicit API key (overrides --api-key-env)")
    ap.add_argument("--output", "-o", required=True,
                    help="output file for the parsed apparatus JSON")
    ap.add_argument("--workdir", default=None,
                    help="scratch dir for all intermediate stage files "
                         "(default: <output-dir>/vulgata_scan_work)")
    ap.add_argument("--chapter", default="1",
                    help="chapter the apparatus starts at (default: 1)")
    ap.add_argument("--align", default="global", choices=["global", "difflib"],
                    help="reconciliation alignment route (paper default: global)")
    ap.add_argument("--no-geometry", action="store_true",
                    help="disable word-bbox geometry in the global alignment "
                         "(ablation)")
    ap.add_argument("--debug", action="store_true",
                    help="write stage-2 annotated debug PNGs")
    ap.add_argument("--force-inventory", action="store_true",
                    help="grammar check: replace witnesses outside the "
                         "inventory with '?'")
    ap.add_argument("--transcript", default=None,
                    help="reuse a saved stage-1 artifact (<stem>_model.json + "
                         "<stem>_page_sizes.json in --workdir) instead of "
                         "calling the backend; with a .md/.txt argument the "
                         "text is used as the whole transcription directly")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="backend request timeout in seconds")


def _build_dataset_parser(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("which", nargs="?", default="gold",
                   choices=["gold", "silver", "both"],
                   help="which dataset to show")
    ap.add_argument("--samples", type=int, default=3,
                   help="number of entries to pretty-print")
    ap.add_argument("--path", default=None,
                    help="explicit JSON path (else the packaged copies)")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vulgata-scan",
        description="Parse the critical apparatus of the Benedictine Vulgate "
                    "(VLM transcription + glyph classifier + reconciliation "
                    "+ formal grammar).")
    sub = ap.add_subparsers(dest="command", required=True)
    _build_parse_parser(sub.add_parser(
        "parse", help="run the four-stage pipeline on one file"))
    _build_dataset_parser(sub.add_parser(
        "dataset", help="explore the packaged gold/silver datasets"))
    return ap


def _cmd_parse(args: argparse.Namespace) -> int:
    # Import lazily so `vulgata-scan dataset` works without torch etc.
    from ..backends import build_backend
    from ..pipeline import run_pipeline
    from ..vlm.transcribe import Stage1Artifact

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2

    work_dir = (Path(args.workdir) if args.workdir
                else Path(args.output).resolve().parent / "vulgata_scan_work")

    artifact = None
    if args.transcript:
        transcript = Path(args.transcript)
        if transcript.suffix in {".md", ".txt"}:
            from ..vlm.transcribe import artifact_from_transcript
            artifact = artifact_from_transcript(
                input_path, transcript.read_text(encoding="utf-8"), work_dir)
        else:
            native_glob = sorted(
                (work_dir / "native").glob("page-*.png"))
            artifact = Stage1Artifact.load(
                transcript.parent, args.transcript if transcript.suffix
                else transcript.stem,
                native_pages=native_glob or None)
            artifact.native_pages = (native_glob or artifact.native_pages)

    if artifact is None:
        backend = build_backend(args.backend, endpoint=args.endpoint,
                                model=args.model, api_key=args.api_key,
                                api_key_env=args.api_key_env,
                                timeout=args.timeout)
    else:
        backend = None  # type: ignore[assignment]

    result = run_pipeline(
        input_path, backend, args.output, work_dir,
        chapter=args.chapter, align=args.align,
        geometry=not args.no_geometry, debug=args.debug,
        force_inventory=args.force_inventory, artifact=artifact)
    print(f"\nparsed {len(result.entries)} verse entries "
          f"-> {args.output}")
    for name, path in result.outputs.items():
        print(f"  {name}: {path}")
    return 0


def _cmd_dataset(args: argparse.Namespace) -> int:
    from .. import dataset as ds
    which = ["gold", "silver"] if args.which == "both" else [args.which]
    for name in which:
        ds.show_samples(name, n=args.samples, path=args.path)
    return 0


def main(argv: list | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "parse":
        return _cmd_parse(args)
    if args.command == "dataset":
        return _cmd_dataset(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
