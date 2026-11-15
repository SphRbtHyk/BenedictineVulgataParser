"""Apparatus-region selection: flag which OCR-detected regions belong to the critical apparatus,
so segmentation and classification only run where needed.

Signal: Greek capitals, Modifier Letters, and Mathematical Fraktur/Italic codepoints
never appear in this book's running Latin text, only in sigla,
so any region containing one is apparatus material with very high precision.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
from PIL import Image

from .segment import measure_skew, paper_level

# Greek bases from the taxonomy, the Modifier Letters and Mathematical Fraktur blocks,
# and the declaratio symbols that are unambiguous alone (pilcrow, dagger, angle bracket;
# section mark and asterisk are too common in body text to use as a signal).
_GREEK_BASES = "ΘΦΨΩΛΣΠ"
_MODIFIER_LETTERS = re.compile(r"[ᴬ-ᵪⱽꟳ]")
_FRAKTUR_OR_ITALIC_MATH = re.compile(r"[\U0001D400-\U0001D7FF]")
_UNAMBIGUOUS_DECLARATIO = "¶†⟨"
# GLM emits superscripts as <sup>ASCII</sup> or inline LaTeX ($\Sigma^{\mathrm{M}}$),
# so a literal-codepoint scan alone misses most apparatus regions.
# GLM never emits the main text's verse numerals as <sup> (checked), so the tag is safe.
_MARKUP_SIGLA = re.compile(r"<su[pb]>|\\(Theta|Phi|Psi|Omega|Lambda|Sigma|Pi|mathfrak|dagger)\b")

_SIGLA_PATTERN = re.compile(
    f"[{_GREEK_BASES}{_UNAMBIGUOUS_DECLARATIO}]|{_MODIFIER_LETTERS.pattern}"
    f"|{_FRAKTUR_OR_ITALIC_MATH.pattern}|{_MARKUP_SIGLA.pattern}"
)


def is_apparatus_text(text: str) -> bool:
    return bool(_SIGLA_PATTERN.search(text))


_NON_LETTERS = re.compile(r"[^A-Za-z]")


def desunt_line_index(text: str) -> int | None:
    """Index of the line opening with 'Desunt' (or singular 'Deest', printed when
    exactly one witness is absent, e.g. p182), the witness-absence header
    that starts the second apparatus; None if absent.
    Matched on the letters-only prefix so GLM markup variants ('*Desunt G*...') all hit.
    """
    for i, line in enumerate(text.splitlines()):
        if _NON_LETTERS.sub("", line).lower().startswith(("desunt", "deest")):
            return i
    return None


def filter_apparatus_regions(glm_output: dict) -> list[dict]:
    """Subset of GLM `regions` whose content contains sigla-family characters,
    as the original region dicts (bbox in `page_image_size` pixel space, ~200 DPI).
    Scans `content` (HTML), not `content_raw`:
    `content_raw` encodes Greek letters as LaTeX macros, which the codepoint scan misses.
    """
    return [r for r in glm_output["regions"] if is_apparatus_text(r.get("content", ""))]


# ---------------------------------------------------------------------------
# Page zoning by the printed divider rules
# ---------------------------------------------------------------------------
# Each page is three bands separated by two printed horizontal rules:
# main text / the apparatus we parse / a footnote-like bottom apparatus.
# The rules are detected on the NATIVE page image, independently of GLM,
# so a region is still processed when GLM normalized away every siglum in it;
# the content scan above is then a cross-check, not the gate.

ZONES = ("main", "apparatus", "footnote")


def detect_rules(native_page, thr_offset: int = 45, min_row_px: int = 60,
                 max_band_h: int = 18, min_cover: float = 0.22,
                 y_min_frac: float = 0.3, y_max_frac: float = 0.95,
                 deskew_deg: float | None = None) -> list[int]:
    """y centers (native px) of the printed horizontal divider rules.

    A rule is a thin row band (<= max_band_h) whose ink columns cover >= min_cover of the page width.
    Detection runs in DESKEWED geometry:
    a ~1700 px rule smears across ~26 rows at only 0.75 deg of skew,
    while after deskewing it collapses to its true 4-15 px thickness
    and the nearest text-line impostor is 23 px, so max_band_h=18 separates cleanly.
    Returned y values are in the deskewed frame (the rule's y at the page's center column);
    across the width the physical rule deviates by up to ~13 px,
    so callers comparing against unrotated bbox coordinates must tolerate that.

    Ink is thresholded against the estimated paper level (segment.paper_level,
    robust to both raw gray renders and background-normalized white pages),
    not via segment.ink_mask: the white pdftoppm margin outnumbers the scan
    paper on a full page and flips ink_mask's polarity.
    y_max_frac drops the dark edge band the rotation leaves at the page bottom.
    deskew_deg: page skew from segment.measure_skew(); measured here when None.
    """
    native_page = native_page.convert("L")
    if deskew_deg is None:
        deskew_deg = measure_skew(native_page)
    if abs(deskew_deg) >= 0.1:
        native_page = native_page.rotate(deskew_deg, resample=Image.BILINEAR,
                                         fillcolor=255)
    arr = np.asarray(native_page, dtype=np.uint8)
    paper = paper_level(arr)
    mask = arr < paper - thr_offset
    h, w = mask.shape
    counts = mask.sum(axis=1)
    bands: list[tuple[int, int]] = []
    start = None
    for y, ok in enumerate(counts >= min_row_px):
        if ok and start is None:
            start = y
        elif not ok and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, h))
    merged = bands[:1]
    for a, b in bands[1:]:
        if a - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(y0 + y1) // 2 for y0, y1 in merged
            if y1 - y0 <= max_band_h and y0 >= y_min_frac * h
            and (y0 + y1) / 2 <= y_max_frac * h
            and mask[y0:y1].any(axis=0).sum() >= min_cover * w]


def assign_zone(bbox_native: tuple[float, float, float, float],
                rule_ys: list[int]) -> str:
    """Zone of a region by its vertical center vs the two divider rules."""
    y1, y2 = rule_ys
    yc = (bbox_native[1] + bbox_native[3]) / 2
    return "main" if yc < y1 else "apparatus" if yc < y2 else "footnote"


def region_band_gaps(bboxes_native: list[list[float]],
                     min_gap_px: int = 50) -> list[tuple[int, int]]:
    """Page-wide horizontal gaps (y0, y1) in the union of region y-intervals,
    i.e. the layout detector's own view of the page's band structure.
    Used as a second opinion for zoning: every printed rule must fall inside a gap,
    and when rule detection fails the gaps serve as pseudo-rules.
    Never the primary signal: a region merged across a divider silently erases the gap,
    whereas the printed rule survives.
    """
    ivs = sorted((b[1], b[3]) for b in bboxes_native)
    if not ivs:
        return []
    merged = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(int(p[1]), int(n[0])) for p, n in zip(merged, merged[1:])
            if n[0] - p[1] >= min_gap_px]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("glm_output_json")
    args = ap.parse_args()

    with open(args.glm_output_json, encoding="utf-8") as f:
        data = json.load(f)

    regions = filter_apparatus_regions(data)
    print(f"{len(regions)}/{len(data['regions'])} regions flagged as apparatus-criticus")
    for r in regions:
        print(f"  region {r['index']:3d}  bbox={r['bbox']}  {r['content'][:70]!r}")
