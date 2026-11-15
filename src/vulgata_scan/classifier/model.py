"""Multi-head CNN for siglum classification, built as a shared backbone with one linear head per label family.

The backbone is deliberately shallow: the real signal is only a ~10-24 px glyph on the 64x64 canvas,
and a deeper backbone would overfit a training set built from ~200 reference images.
The family head gates the others.
Training masks each head's loss by the sample's family,
and inference reads only the heads selected by the predicted family (infer.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .labelspace import LabelSpace


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SiglaCNN(nn.Module):
    def __init__(self, label_space: LabelSpace, in_ch: int = 1, width: int = 32):
        super().__init__()
        self.ls = label_space

        # Resolution 64 -> 32 -> 16 -> 8 -> 4, width doubling each stage.
        self.backbone = nn.Sequential(
            conv_block(in_ch, width),
            conv_block(width, width * 2),
            conv_block(width * 2, width * 4),
            conv_block(width * 4, width * 4),
        )
        feat_dim = width * 4 * 4 * 4  # channels * H * W at the 4x4 stage

        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.head_family = nn.Linear(256, len(label_space.families))
        self.head_base = nn.Linear(256, len(label_space.bases))
        self.head_superscript = nn.Linear(256, len(label_space.superscripts))   # multi-label -> sigmoid at loss time
        self.head_numeral = nn.Linear(256, len(label_space.numeral_classes()))
        self.head_fraktur = nn.Linear(256, len(label_space.fraktur_letters))    # multi-label -> sigmoid at loss time
        self.head_declaratio = nn.Linear(256, len(label_space.declaratio_symbols))
        self.head_latin = nn.Linear(256, len(label_space.latin_letters))
        # The per-component head decodes one small crop at a time, so glyph order
        # is observed rather than reconstructed. It also carries the "reject" class.
        self.head_glyph = nn.Linear(256, len(label_space.glyph_classes()))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.trunk(self.backbone(x))
        return {
            "family": self.head_family(feat),
            "base": self.head_base(feat),
            "superscript": self.head_superscript(feat),
            "numeral": self.head_numeral(feat),
            "fraktur": self.head_fraktur(feat),
            "declaratio": self.head_declaratio(feat),
            "latin": self.head_latin(feat),
            "glyph": self.head_glyph(feat),
        }


if __name__ == "__main__":
    from taxonomy import load_taxonomy

    ls = LabelSpace.from_taxonomy(load_taxonomy())
    model = SiglaCNN(ls)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} parameters")
    x = torch.randn(4, 1, 64, 64)
    out = model(x)
    for k, v in out.items():
        print(k, tuple(v.shape))
