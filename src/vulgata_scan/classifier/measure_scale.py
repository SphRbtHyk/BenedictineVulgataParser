"""
Measures the actual pixel height of a text line on a native-resolution page render,
using the row-intensity projection profile.
The results provide the `measured_scale_px_native` values in `config/taxonomy.yaml`.

Usage:
    python measure_scale.py page1.png --y0 3140 --y1 3230

`y0`/`y1` delimit a vertical band known to contain exactly one apparatus text line;
identify the coordinates visually beforehand on a full-page render.

IMPORTANT: render the page with `pdftoppm -png -r <dpi>`, never `pdfimages`.
`pdftoppm` renders the full page, so one clean scale factor maps its pixels
onto GLM/PP-DocLayout-V3 bounding boxes (verified: bbox * 330/200 aligns perfectly).
`pdfimages` dumps the raw embedded XObject without page placement,
and crops taken from it land on the wrong paragraph.
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image


def measure_line_height(image_path: str, y0: int, y1: int, row_threshold_frac: float = 0.05) -> dict:
    im = Image.open(image_path).convert("L")
    arr = np.array(im.crop((0, y0, im.width, y1))).astype(float)
    if arr.mean() > 127:
        # Invert white-background images: the row sums below expect ink to be the high values.
        arr = 255.0 - arr
    row_sum = arr.sum(axis=1)
    ink_rows = np.where(row_sum > row_sum.max() * row_threshold_frac)[0]
    if ink_rows.size == 0:
        raise ValueError(
            "No text found in the specified area. Please check your y0 and y1 coordinates. "
            "The code already accounts for white/black backgrounds automatically, "
            "so this error usually means the selected region is completely blank."
        )
    return {
        "band_top": int(y0 + ink_rows.min()),
        "band_bottom": int(y0 + ink_rows.max()),
        "full_line_height_px": int(ink_rows.max() - ink_rows.min()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_path", help="native-resolution page PNG from `pdftoppm -png -r 330` "
                    "(NOT pdfimages, see module docstring)")
    ap.add_argument("--y0", type=int, required=True)
    ap.add_argument("--y1", type=int, required=True)
    args = ap.parse_args()

    result = measure_line_height(args.image_path, args.y0, args.y1)
    print(result)
    print(
        "\nNote: This tool provides the FULL line height, spanning from ascenders and superscripts down to descenders.\n"
        "If you need the base-character cap-height or the superscript-glyph height, you must use a tighter crop.\n"
        "To do this, select a single known token and rerun the script.\n"
        "Adjust your y0 and y1 coordinates to delimit only that specific token's base or superscript band.\n"
        "For reference, this is usually around 20-24px for the base and 10-15px for superscripts at 330 PPI on this scan."
    )
