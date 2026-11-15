"""The one resize step shared between training and inference.

CONTRACT: `to_cnn_input` must stay byte-identical to the version used to train the released checkpoint.
It is the last transform every training image went through;
changing it silently shifts the input distribution under the frozen model.
"""
from __future__ import annotations

from PIL import Image

CNN_INPUT_SIZE = 64


def to_cnn_input(im: Image.Image, size: int = CNN_INPUT_SIZE) -> Image.Image:
    """Pad to square (so aspect ratio is preserved), then resize to the CNN input size."""
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), color=255)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), resample=Image.BILINEAR)


def preprocess_for_inference(crop: Image.Image, size: int = CNN_INPUT_SIZE) -> Image.Image:
    """Prepare a real crop for the CNN: grayscale conversion plus the shared final resize."""
    return to_cnn_input(crop.convert("L"), size)
