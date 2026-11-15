"""Rendering: PDF page -> native 330 PPI grayscale page images.

The glyph classifier is trained on the edition's scans rendered at 330 DPI
grayscale, so the pipeline always renders its own native images from the
input PDF, independently of the VLM.  Poppler (``pdftoppm``) is used when
available; otherwise PyMuPDF renders the same images.

Scans of the edition carry solid black border stripes around the paper;
they are trimmed here so that the segmentation never sees them (a border
stripe is one giant connected component that would swallow every line).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

#: Native render resolution (matches the classifier training resolution).
NATIVE_DPI = 330

#: Column/row is border when more than this fraction of it is near-black.
_BORDER_INK_FRACTION = 0.5
# a shade dark enough to be the scan border but not print grayshades?  no:
# print ink is also dark -- the discriminator is the *fraction*, not value.
_BORDER_DARK = 100

#: Window (pixels, ~ the size of the largest printed glyph) of the median
#: filter used to estimate the scan's illumination background.
_BACKGROUND_FILTER_SIZE = 71


def _render_with_poppler(pdf_path: Path, out_dir: Path) -> list[Path]:
    """Render with ``pdftoppm`` into ``out_dir``; returns the page paths."""
    prefix = out_dir / "page"
    subprocess.run(
        ["pdftoppm", "-r", str(NATIVE_DPI), "-gray", "-png",
         str(pdf_path), str(prefix)],
        check=True, capture_output=True)
    return sorted(out_dir.glob(prefix.name + "-*.png"))


def _render_with_pymupdf(pdf_path: Path, out_dir: Path) -> list[Path]:
    """Render with PyMuPDF (no Poppler required); returns the page paths."""
    import pymupdf  # local import: keeps PyMuPDF optional at import time

    paths: list[Path] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=NATIVE_DPI, colorspace=pymupdf.csGRAY)
            out = out_dir / f"page-{i:02d}.png"
            pix.save(str(out))
            paths.append(out)
    return paths


def trim_scan_border(image: Image.Image) -> Image.Image:
    """Trim the solid near-black border stripes of a scanned page render.

    A column (or row) counts as border when at least
    ``_BORDER_INK_FRACTION`` of it is darker than ``_BORDER_DARK``: printed
    text never fills half a column, a scan border always does.
    """
    arr = np.array(image.convert("L"))
    dark = arr < _BORDER_DARK
    col_frac = dark.mean(axis=0)
    row_frac = dark.mean(axis=1)
    w, h = arr.shape[1], arr.shape[0]

    def _run_left(frac: np.ndarray) -> int:
        i = 0
        while i < len(frac) and frac[i] > _BORDER_INK_FRACTION:
            i += 1
        return i

    def _run_right(frac: np.ndarray) -> int:
        i = len(frac)
        while i > 0 and frac[i - 1] > _BORDER_INK_FRACTION:
            i -= 1
        return i

    x0, x1 = _run_left(col_frac), _run_right(col_frac)
    y0, y1 = _run_left(row_frac), _run_right(row_frac)
    # degenerate (a fully dark "page"): keep the original
    if x1 - x0 < 0.5 * w or y1 - y0 < 0.5 * h:
        return image
    return image.crop((x0, y0, x1, y1))


def normalize_background(image: Image.Image) -> Image.Image:
    """Flatten a scan's gray illumination background to clean white paper.

    The classifier is trained on white-background crops, and the stage-2 ink
    mask (Otsu) assumes a bright page background.  Scans whose paper reads
    gray (photographed pages, dark microfilm) need this division by the
    estimated background; the estimate is a median filter larger than any
    printed glyph (so glyphs vanish from it), computed on an eight-fold
    downscale because the illumination field is smooth.  No-op on
    already-clean renders (background stays 255).
    """
    from PIL import Image as _Image
    from scipy import ndimage

    im = image.convert("L")
    arr = np.asarray(im).astype(np.float32)
    if np.percentile(arr, 25) > 245:  # already clean white paper
        return image
    zoom = 8
    small = arr[::zoom, ::zoom]
    bg_small = ndimage.median_filter(small,
                                     size=_BACKGROUND_FILTER_SIZE // zoom + 1)
    bg = ndimage.zoom(bg_small, zoom, order=1)[:arr.shape[0], :arr.shape[1]]
    norm = np.clip(arr * 255.0 / np.maximum(bg, 1.0), 0, 255).astype(np.uint8)
    return _Image.fromarray(norm)


def render_pdf(pdf_path: Path, out_dir: Path, trim: bool = True,
               normalize: bool = True) -> list[Path]:
    """Render every page of ``pdf_path`` as a native 330 PPI grayscale PNG.

    Args:
        pdf_path: input PDF (scanned pages of the edition).
        out_dir: directory receiving ``page-<n>.png``. Created if missing.
        trim: trim the solid scan-border stripes (default on).
        normalize: flatten gray scan backgrounds to white (default on).

    Returns:
        The page image paths, in document order.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("pdftoppm"):
        pages = _render_with_poppler(pdf_path, out_dir)
    else:
        pages = _render_with_pymupdf(pdf_path, out_dir)
    if not pages:
        raise RuntimeError(f"no pages rendered from {pdf_path}")
    for p in pages:
        with Image.open(p) as im:
            im.load()
            out = trim_scan_border(im) if trim else im
            out = normalize_background(out) if normalize else out
        out.save(p)
    return pages


def image_size(page_path: Path) -> tuple[int, int]:
    """Pixel ``(width, height)`` of a rendered page."""
    with Image.open(page_path) as im:
        return im.size
