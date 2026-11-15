"""Regression tests for divider-rule detection (zone detection).

Bug history: ``detect_rules`` estimated the paper level as
``median(arr[arr < 250])``, which is only correct for raw gray pdftoppm
renders.  On background-normalized pages (paper pushed to white) the
sub-250 pixels are *ink only*, the estimate collapsed to the ink median,
and the divider rules of the apparatus band were no longer detected --
the zones PNG showed no rules and stage 1 fell back to a whole-page band.
``paper_level`` (histogram-bright-class estimate) fixes both flavours.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from vulgata_scan.classifier.apparatus_filter import detect_rules
from vulgata_scan.classifier.segment import measure_skew, paper_level


def _synthetic_page(paper_gray: int, w: int = 1200, h: int = 1700,
                    rules=(700, 1300)) -> Image.Image:
    """A page with a white render margin, gray paper, fake text lines and
    two full-width printed divider rules."""
    arr = np.full((h, w), 255, dtype=np.uint8)          # white margin
    m = 60                                               # margin width
    arr[m:-m, m:-m] = paper_gray                         # the scan paper
    rng = np.random.default_rng(7)
    # 9 px lines on an 11 px pitch: adjacent lines merge into one thick band
    # (> max_band_h), exactly like real text lines -- only the 5 px rules stay thin.
    for y in range(m + 30, rules[0] - 40, 11):           # fake text above rule 1
        x0 = m + 40
        while x0 < w - m - 200:
            x1 = min(x0 + rng.integers(8, 30), w - m - 40)
            arr[y:y + 9, x0:x1] = paper_gray - 110
            x0 = x1 + rng.integers(4, 14)
    for y in range(rules[0] + 40, rules[1] - 40, 11):    # text between the rules
        x0 = m + 40
        while x0 < w - m - 200:
            x1 = min(x0 + rng.integers(8, 30), w - m - 40)
            arr[y:y + 9, x0:x1] = paper_gray - 110
            x0 = x1 + rng.integers(4, 14)
    for ry in rules:                                     # the printed rules
        arr[ry:ry + 5, m:w - m] = min(paper_gray - 130, 40)
    return Image.fromarray(arr, mode="L")


def test_paper_level_gray_scan_with_white_margin():
    arr = np.asarray(_synthetic_page(paper_gray=150))
    assert paper_level(arr) == 150  # gray paper, not the white margin


def test_paper_level_normalized_white_page():
    arr = np.asarray(_synthetic_page(paper_gray=150)).astype(np.float32)
    norm = np.clip(arr * 255.0 / 150.0, 0, 255).astype(np.uint8)  # what normalize_background does
    assert paper_level(norm) == 255.0


def test_detect_rules_on_raw_gray_scan():
    im = _synthetic_page(paper_gray=150)
    rules = detect_rules(im, deskew_deg=measure_skew(im))
    assert len(rules) == 2
    assert abs(rules[0] - 700) <= 10 and abs(rules[1] - 1300) <= 10


def test_detect_rules_on_normalized_page():
    # the regression: after background normalization the rules must survive
    im = _synthetic_page(paper_gray=150)
    arr = np.asarray(im).astype(np.float32)
    norm = Image.fromarray(
        np.clip(arr * 255.0 / 150.0, 0, 255).astype(np.uint8), mode="L")
    rules = detect_rules(norm, deskew_deg=measure_skew(norm))
    assert len(rules) == 2
    assert abs(rules[0] - 700) <= 10 and abs(rules[1] - 1300) <= 10
