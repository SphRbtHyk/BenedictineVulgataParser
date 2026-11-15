#!/usr/bin/env bash
# Reproduce the paper's numbers with the packaged evaluation tools, against
# the archived corpus (hand-annotated gold + archived pipeline outputs).
#
# Run from anywhere with the package installed; the script locates the
# REPOSITORY ROOT (the parent of this folder) itself.  Each command prints
# the "[1] [2] [3] [4] [G]" sections the paper's results section reports.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

EB="evaluation/parsed_datasets"   # archived PARSED files (grammar output)
ER="evaluation/raw_datasets"      # archived RAW transcriptions

echo "== [Census + metrics 1-4 + global CER]  full pipeline (VLM + classifier + reconcile) vs gold =="
$PY -m vulgata_scan.evaluation.evaluate_parser \
    --gt "$EB/ground_truth.json" --pred "$EB/glm_classifier.json" \
    --gt-text "$ER/ground_truth.txt" --pred-text "$ER/glm_classifier.txt" \
    --per-verse

echo
echo "== Same, VLM only (stage 1 alone) =="
$PY -m vulgata_scan.evaluation.evaluate_parser \
    --gt "$EB/ground_truth.json" --pred "$EB/glm.json" \
    --gt-text "$ER/ground_truth.txt" --pred-text "$ER/glm.txt"

echo
echo "== Character error rate, raw transcription vs the annotated ground truth =="
$PY -m vulgata_scan.evaluation.apparatus_cer \
    --gt "$ER/ground_truth.txt" --pred "$ER/glm_classifier.txt" --per-page

echo
echo "== Three-way ablation (gold / naive VLM+CNN / full VLM+CNN) =="
echo "   (several minutes; needs the region-segmented markdown views)"
if [ -f "evaluation/pipeline_output/output_glm_classifier_naive.md" ]; then
    $PY -m vulgata_scan.evaluation.ablation_study \
        --gold "$ER/ground_truth.txt" \
        --vlm "evaluation/pipeline_output/output_glm_classifier_naive.md" \
        --pipe "evaluation/pipeline_output/output_glm_classifier.md"
else
    echo "   (skip: evaluation/pipeline_output/*.md not present)"
fi