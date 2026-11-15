"""Label vocabularies for the sigla CNN.

The LabelSpace fixes the output dimension of every head in model.py.
At inference it is reconstructed from the dict stored inside the checkpoint, not from the taxonomy,
so a checkpoint always decodes with the exact vocabulary it was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import Taxonomy, load_taxonomy

# Italic abbreviations are deliberately not a family,
# because GLM-OCR reads them correctly and what the VLM already gets right stays background.
FAMILIES = ["manuscript_siglum", "fraktur_cluster", "declaratio_signorum",
            "latin_letter", "background"]


@dataclass
class LabelSpace:
    """Label vocabularies derived from the taxonomy, fixing every head's output dimension.
    Built from the Taxonomy rather than the dataset,
    so the vocabulary does not shift when a rare combination is missing from a generation run."""
    families: list[str]
    bases: list[str]
    superscripts: list[str]        # union alphabet, Head-B label space
    fraktur_letters: list[str]
    declaratio_symbols: list[str]
    latin_letters: list[str]
    trailing_numerals: list[str]   # + implicit "none" handled as its own class

    @classmethod
    def from_taxonomy(cls, tax: Taxonomy) -> "LabelSpace":
        return cls(
            families=FAMILIES,
            bases=tax.base_labels,
            superscripts=tax.all_superscript_letters,
            fraktur_letters=list(tax.fraktur_letters),
            declaratio_symbols=sorted(tax.declaratio_signorum.keys()),
            latin_letters=list(tax.latin_letters),
            trailing_numerals=list(tax.trailing_numerals),
        )

    def numeral_classes(self) -> list[str]:
        return ["none", *self.trailing_numerals]

    def glyph_classes(self) -> list[str]:
        """The glyph head label space covers everything a single small crop can be.
        Superscript letters are uppercase and fraktur lowercase, so the strings never collide.
        "reject" is the escape hatch for punctuation fragments wrongly attached by segmentation."""
        return [*self.superscripts, *self.trailing_numerals, "*", "†",
                *self.fraktur_letters, "reject"]


if __name__ == "__main__":
    ls = LabelSpace.from_taxonomy(load_taxonomy())
    print("families:", ls.families)
    print("bases:", ls.bases)
    print("superscripts (union):", ls.superscripts)
    print("fraktur_letters:", ls.fraktur_letters)
    print("declaratio_symbols:", ls.declaratio_symbols)
    print("glyph_classes:", ls.glyph_classes())
