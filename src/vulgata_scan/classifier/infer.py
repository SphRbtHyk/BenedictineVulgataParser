"""Inference: decode one already-segmented token crop into a structured sigla label.

Crops must come from the native 330 PPI scans,
because the classifier was trained at that resolution
and the ~12 px superscript glyphs do not survive further downsampling.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from .labelspace import LabelSpace
from .model import SiglaCNN
from .preprocess import preprocess_for_inference
from .taxonomy import load_taxonomy

SIGMOID_THRESHOLD = 0.5
_TAX = load_taxonomy()  # parsed once at import; the yaml never changes within a run


class SiglaClassifier:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        ls_dict = ckpt["label_space"]
        self.ls = LabelSpace(**ls_dict)
        self.model = SiglaCNN(self.ls).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, crop: Image.Image) -> dict:
        im = preprocess_for_inference(crop)
        x = torch.from_numpy(np.array(im)).float().unsqueeze(0).unsqueeze(0) / 255.0
        x = x.to(self.device)
        pred = self.model(x)

        family_idx = pred["family"].argmax(1).item()
        family = self.ls.families[family_idx]
        family_confidence = torch.softmax(pred["family"], dim=1)[0, family_idx].item()

        result = {"family": family, "family_confidence": family_confidence}

        if family == "manuscript_siglum":
            base_idx = pred["base"].argmax(1).item()
            base = self.ls.bases[base_idx]
            base_confidence = torch.softmax(pred["base"], dim=1)[0, base_idx].item()
            sup_probs = torch.sigmoid(pred["superscript"])[0]
            superscripts = [self.ls.superscripts[i] for i, p in enumerate(sup_probs) if p > SIGMOID_THRESHOLD]
            numeral_idx = pred["numeral"].argmax(1).item()
            numeral_classes = self.ls.numeral_classes()
            numeral = None if numeral_classes[numeral_idx] == "none" else numeral_classes[numeral_idx]
            result.update({
                "base": base, "base_confidence": base_confidence,
                "superscripts": superscripts,
                "superscript_probs": {self.ls.superscripts[i]: float(p) for i, p in enumerate(sup_probs)},
                "trailing_numeral": numeral,
                "_consistency_warning": self._check_consistency(base, superscripts),
            })
        elif family == "fraktur_cluster":
            fr_probs = torch.sigmoid(pred["fraktur"])[0]
            letters = [self.ls.fraktur_letters[i] for i, p in enumerate(fr_probs) if p > SIGMOID_THRESHOLD]
            result["letters"] = letters
        elif family == "declaratio_signorum":
            idx = pred["declaratio"].argmax(1).item()
            result["symbol"] = self.ls.declaratio_symbols[idx]
        elif family == "latin_letter":
            idx = pred["latin"].argmax(1).item()
            result["letter"] = self.ls.latin_letters[idx]

        return result

    @torch.no_grad()
    def predict_glyph(self, crop: Image.Image, allowed: list[str] | None = None,
                      allow_reject: bool = True) -> tuple[str, float]:
        """Classify one small component crop; returns (glyph class, confidence).
        `allowed` restricts the candidates to a subset of glyph classes.
        Unless allow_reject is False, "reject" always stays a candidate regardless of `allowed`,
        otherwise the head would be forced to call a punctuation fragment some letter."""
        im = preprocess_for_inference(crop)
        x = torch.from_numpy(np.array(im)).float().unsqueeze(0).unsqueeze(0) / 255.0
        logits = self.model(x.to(self.device))["glyph"][0]
        classes = self.ls.glyph_classes()
        probs = torch.softmax(logits, dim=0)
        if allowed is not None:
            cand = {*allowed, "reject"} if allow_reject else set(allowed) - {"reject"}
            idxs = [classes.index(a) for a in cand]
            best = max(idxs, key=lambda i: probs[i].item())
        else:
            best = int(probs.argmax())
        return classes[best], float(probs[best])

    def _check_consistency(self, base: str, superscripts: list[str]) -> str | None:
        """Flag superscripts outside this base's known candidate set.
        The superscript head shares one alphabet across all bases,
        so an out-of-set prediction means either a needed taxonomy update or a genuinely ambiguous crop."""
        tax = _TAX
        allowed = set(tax.bases[base].superscripts) if base in tax.bases else set()
        unexpected = [s for s in superscripts if s not in allowed]
        if unexpected:
            return f"base {base!r} does not list {unexpected} as a known candidate superscript, treat as low confidence"
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("image", help="a single already-segmented token crop")
    args = ap.parse_args()

    clf = SiglaClassifier(args.checkpoint)
    crop = Image.open(args.image)
    result = clf.predict(crop)
    for k, v in result.items():
        print(f"{k}: {v}")
