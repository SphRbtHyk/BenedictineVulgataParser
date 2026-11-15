"""Geometric segmentation: crop an apparatus region from the native 330 PPI page image,
split it into lines, then split each line into per-token crops.

Adjacent sigla are often printed without spaces (e.g. "CΛᴸTMO" is five sigla),
so tokens cannot be found by whitespace.
Instead we segment at the glyph level via connected components,
then regroup glyphs into tokens by size and vertical position:
a small, raised component following a normal-sized one is a superscript, not a new token.
The geometry thresholds were measured on the scans (see measure_scale.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

SUPERSCRIPT_HEIGHT_RATIO = 0.75   # a component shorter than this * line's base height is considered "small"
SUPERSCRIPT_TOP_MARGIN = 0.15     # and must also start at least this * base height above the baseline
TOKEN_GAP_RATIO = 0.6             # an x-gap bigger than this * base height starts a new token


def paper_level(arr: np.ndarray) -> float:
    """Estimate the paper (background) gray level of a page or crop.

    Robust to both render flavours the pipeline meets:

    - raw ``pdftoppm`` renders of gray scans: a white render margin coexists
      with gray paper, so the sub-250 pixels are mostly paper and their
      median is the paper level;
    - background-normalized (or clean white) pages: almost every pixel is
      >= 250, so the paper level is simply white.

    The legacy ``median(arr[arr < 250])`` only handles the first case: on a
    normalized page the sub-250 pixels are *ink only*, so the "paper" estimate
    collapses to the ink median and every threshold derived from it is wrong
    (this is what made divider-rule detection fail on normalized pages).
    """
    bright = arr >= 250
    if bright.mean() >= 0.5:
        return 255.0
    sub = arr[~bright]
    return float(np.median(sub)) if sub.size else 255.0


def otsu_threshold(arr: np.ndarray) -> int:
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0
    best_t, best_var = 0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var, best_t = var_between, t
    return best_t


def ink_mask(arr: np.ndarray) -> np.ndarray:
    """Generates a boolean mask where True represents ink.

    Polarity-robust: ink is assumed to be the minority class in a region crop,
    so both white-paper renders (the production `pdftoppm -r 330 -gray` input)
    and inverted scans threshold correctly.
    """
    t = otsu_threshold(arr)
    below = arr <= t
    return below if below.sum() < (arr.size - below.sum()) else ~below


def measure_skew(native_page: Image.Image, max_deg: float = 1.5, step_deg: float = 0.05,
                 downsample: int = 4, thr_offset: int = 45) -> float:
    """Calculates the text-line skew of a scanned page in degrees.

    Shear-histogram search: every ink pixel is projected onto y - x*tan(θ),
    and the θ with the sharpest row profile (sum of squared bin counts) wins.
    Positive means the lines drop toward the right (image y points downward).
    Pass the result to `crop_region_from_native_page(deskew_deg=...)`.
    """
    arr = np.asarray(native_page.convert("L").reduce(downsample), dtype=np.uint8)
    paper = paper_level(arr)
    ys, xs = np.nonzero(arr < paper - thr_offset)
    if len(ys) < 1000:
        return 0.0
    best_deg, best_score = 0.0, -1.0
    for deg in np.arange(-max_deg, max_deg + 1e-9, step_deg):
        b = (ys - xs * np.tan(np.radians(deg))).astype(np.int32)
        prof = np.bincount(b - b.min()).astype(np.float64)
        score = float((prof ** 2).sum())
        if score > best_score:
            best_score, best_deg = score, deg
    return round(float(best_deg), 3)


def crop_region_from_native_page(native_page: Image.Image, bbox_200dpi: tuple[float, float, float, float],
                                  scale_to_native: float, deskew_deg: float = 0.0) -> Image.Image:
    """Extracts a deskewed region crop from the native page image.

    `bbox_200dpi` is a GLM/PP-DocLayout-V3 box in its ~200 DPI page space;
    `scale_to_native` is the native/GLM width ratio computed by the caller from actual image sizes.
    The crop (not the page) is rotated by `deskew_deg` so the GLM boxes stay valid.
    Every downstream consumer of these pixels must use the same angle,
    or token bounding boxes will misalign with the ink.
    """
    x0, y0, x1, y1 = [round(v * scale_to_native) for v in bbox_200dpi]
    crop = native_page.crop((x0, y0, x1, y1))
    if abs(deskew_deg) >= 0.1:  # below that, resampling costs more than it fixes
        fill = int(np.median(np.asarray(crop.convert("L"))))
        crop = crop.rotate(deskew_deg, resample=Image.BICUBIC, fillcolor=fill)
    return crop


def _merge_close_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for a, b in runs[1:]:
        pa, pb = merged[-1]
        if a - pb <= max_gap:
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))
    return merged


def segment_lines(mask: np.ndarray, min_gap_px: int = 4,
                  max_line_h: int | None = None) -> list[tuple[int, int]]:
    """Converts a row-projection profile into a list of (y0, y1) line bands.

    If `max_line_h` is given (~1.6x the expected line height),
    any taller band is recursively force-split at its weakest ink row:
    touching descenders/ascenders otherwise merge two real lines into one band.
    """
    row_has_ink = mask.any(axis=1)
    runs = []
    start = None
    for y, v in enumerate(row_has_ink):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, len(row_has_ink)))
    runs = _merge_close_runs(runs, min_gap_px)
    if max_line_h is not None:
        row_ink = mask.sum(axis=1)
        runs = [r for band in runs for r in _split_tall_band(row_ink, *band, max_line_h)]
    return runs


def _split_tall_band(row_ink: np.ndarray, y0: int, y1: int, max_line_h: int) -> list[tuple[int, int]]:
    """Splits an over-tall band at the least-inked row of its middle region,
    even if that row has ink: a clipped descender beats a merged two-line token.
    """
    if y1 - y0 <= max_line_h:
        return [(y0, y1)]
    quarter = (y1 - y0) // 4
    lo, hi = y0 + quarter, y1 - quarter
    if hi - lo < 2:
        return [(y0, y1)]
    cut = lo + int(np.argmin(row_ink[lo:hi]))
    return _split_tall_band(row_ink, y0, cut, max_line_h) + _split_tall_band(row_ink, cut, y1, max_line_h)


@dataclass
class Component:
    bbox: tuple[int, int, int, int]  # x0,y0,x1,y1 within the line crop
    is_small: bool = False


@dataclass
class Token:
    bbox: tuple[int, int, int, int]        # union bbox within the line crop
    components: list[Component]


def segment_line_into_tokens(line_mask: np.ndarray) -> list[Token]:
    """Performs connected-component glyph segmentation and groups the results into tokens.

    The reference height is tracked sequentially:
    the component that last started a token is the anchor,
    and only components smaller and raised relative to that anchor attach as superscripts.
    (A global per-line median fails because a real line mixes x-height letters, capital bases,
    superscripts and punctuation, so the median represents no category.)
    Comparing against the anchor also handles stacked superscripts:
    in ΘᴴᴳΨ, both H and G are compared against Θ, and Ψ starts a fresh token.

    Plain Latin words are deliberately NOT merged into word tokens:
    intra-word letter gaps and gaps between back-to-back bare-letter sigla are both 1-4 px,
    so every normal-sized component starts its own token.
    The resulting over-segmentation is harmless; those crops only need to classify as background.
    """
    labeled, n = ndimage.label(line_mask, structure=np.ones((3, 3)))
    if n == 0:
        return []
    objs = ndimage.find_objects(labeled)
    comps = []
    for sl in objs:
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        comps.append(Component(bbox=(x0, y0, x1, y1)))
    comps.sort(key=lambda c: c.bbox[0])  # left to right

    # Semicolon guard: the upper dot of a semicolon/colon is small and raised,
    # so it passes every attachment test (observed: '𝔞' + dot decoded as {a,c}).
    # Real superscripts never have ink directly below them, punctuation stacks always do,
    # so a component sitting above another one is punctuation and must not attach.
    def _has_ink_below(c: Component) -> bool:
        x0, y0, x1, y1 = c.bbox
        for o in comps:
            if o is c:
                continue
            ox0, oy0, ox1, oy1 = o.bbox
            if oy0 >= y1 and min(x1, ox1) - max(x0, ox0) >= 0.5 * (x1 - x0):
                return True
        return False

    # Line baseline = 75th percentile of component bottoms:
    # most glyphs rest on it, superscripts sit above, only descenders reach below.
    # With fewer than 4 components the percentile is meaningless,
    # so no baseline is set and the descender guard below stays off.
    bottoms = sorted(c.bbox[3] for c in comps)
    baseline = bottoms[int(0.75 * (len(bottoms) - 1))] if len(bottoms) >= 4 else None

    tokens: list[Token] = []
    current_base_height = None
    current_base_bottom = None

    for c in comps:
        x0, y0, x1, y1 = c.bbox
        h = y1 - y0

        if tokens and current_base_height is not None:
            small = h < SUPERSCRIPT_HEIGHT_RATIO * current_base_height
            raised = (current_base_bottom - y1) > SUPERSCRIPT_TOP_MARGIN * current_base_height
            # Dash guard: an em-dash is short and raised, so it passes the size tests,
            # but no genuine superscript is that flat, so extreme-aspect components are never attached.
            # Exception: a NARROW flat mark is a macron (ā), which must stay attached.
            flat = (x1 - x0) > 3 * h and (x1 - x0) > 0.6 * current_base_height
            gap = _gap_before(tokens[-1], c)
            if small and raised and not flat and gap < TOKEN_GAP_RATIO * current_base_height \
                    and not _has_ink_below(c):
                c.is_small = True
                prev = tokens[-1]
                px0, py0, px1, py1 = prev.bbox
                tokens[-1] = Token(bbox=(min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1)),
                                    components=prev.components + [c])
                continue

        tokens.append(Token(bbox=c.bbox, components=[c]))
        # Descender guard: marks like '|', ']', '¶' and descender letters reach well below the baseline.
        # Anchored on one, every following glyph looks small and raised and is absorbed as a superscript,
        # producing phantom sigla ("] in" -> Λ¹¹ᶻ, "¶ ā" -> Λᴴ).
        # A component is barred from anchoring iff it dips below the baseline AND is not a wide capital;
        # both clauses are needed (dip alone breaks 8 genuine wide sigla,
        # shape alone breaks thin honest bases and fraktur '𝔩').
        narrow = 10 * h >= 13 * (x1 - x0)
        dips = baseline is not None and (y1 - baseline) > 0.10 * h
        current_base_height = None if (narrow and dips) else h
        current_base_bottom = y1

    return tokens


def _gap_before(prev_token: Token, comp: Component) -> float:
    return comp.bbox[0] - prev_token.bbox[2]


def crop_token_image(line_image: Image.Image, token: Token, pad: int = 4) -> Image.Image:
    x0, y0, x1, y1 = token.bbox
    w, h = line_image.size
    return line_image.crop((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))


if __name__ == "__main__":
    import sys

    im = Image.open(sys.argv[1]).convert("L")
    arr = np.array(im)
    mask = ink_mask(arr)
    lines = segment_lines(mask)
    print(f"{len(lines)} lines")
    for i, (y0, y1) in enumerate(lines):
        line_mask = mask[y0:y1]
        tokens = segment_line_into_tokens(line_mask)
        print(f"  line {i} (y={y0}-{y1}): {len(tokens)} tokens")
