"""Loads config/taxonomy.yaml into the structured objects used by every other module.

The taxonomy is data-driven,
so adding a base character, superscript, or symbol means editing the yaml, never the model code.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Unicode modifier-letter (superscript capital) code points, keyed by the plain letter.
_SUPERSCRIPT_LETTER_TO_CHAR = {
    "A": "ᴬ", "B": "ᴮ", "C": "ᶜ", "D": "ᴰ", "F": "ꟳ",
    "G": "ᴳ", "H": "ᴴ", "J": "ᴶ", "L": "ᴸ", "M": "ᴹ",
    "O": "ᴼ", "P": "ᴾ", "R": "ᴿ", "S": "ˢ", "T": "ᵀ",
    "V": "ⱽ", "Z": "ᶻ",
}

# Mathematical Fraktur lowercase (U+1D51E..).
_FRAKTUR_LOWER = {
    "a": "\U0001D51E", "b": "\U0001D51F", "c": "\U0001D520", "d": "\U0001D521",
    "e": "\U0001D522", "f": "\U0001D523", "g": "\U0001D524", "h": "\U0001D525",
    "i": "\U0001D526", "j": "\U0001D527", "k": "\U0001D528", "l": "\U0001D529",
    "m": "\U0001D52A", "n": "\U0001D52B", "o": "\U0001D52C", "p": "\U0001D52D",
    "q": "\U0001D52E", "r": "\U0001D52F", "s": "\U0001D530", "t": "\U0001D531",
    "u": "\U0001D532", "v": "\U0001D533", "w": "\U0001D534", "x": "\U0001D535",
    "y": "\U0001D536", "z": "\U0001D537",
}

TRAILING_NUMERAL_TO_CHAR = {"1": "¹", "2": "²", "3": "³"}

DEFAULT_TAXONOMY_PATH = Path(__file__).resolve().parent / "config" / "taxonomy.yaml"


@dataclass
class BaseSpec:
    letter: str
    superscripts: list[str] = field(default_factory=list)
    plain_latin: bool = False  # bare glyph == ordinary Latin capital; see taxonomy.yaml


@dataclass
class Taxonomy:
    bases: dict[str, BaseSpec]
    trailing_numerals: list[str]
    fraktur_letters: list[str]
    declaratio_signorum: dict[str, str]
    latin_letters: list[str]
    reference_dirs: dict[str, Path]
    native_dpi: int
    measured_scale_px_native: dict[str, int]

    # ---- derived, convenience views -------------------------------------------------

    @property
    def all_superscript_letters(self) -> list[str]:
        """Union of every base's candidate superscript alphabet (Head-B label space)."""
        seen: set[str] = set()
        for spec in self.bases.values():
            seen.update(spec.superscripts)
        return sorted(seen)

    @property
    def base_labels(self) -> list[str]:
        return sorted(self.bases.keys())

    def fraktur_char(self, letter: str) -> str:
        return _FRAKTUR_LOWER[letter]

    def superscript_char(self, letter: str) -> str:
        return _SUPERSCRIPT_LETTER_TO_CHAR[letter]

    def trailing_numeral_char(self, digit: str) -> str:
        return TRAILING_NUMERAL_TO_CHAR[digit]


def load_taxonomy(path: Path | str = DEFAULT_TAXONOMY_PATH) -> Taxonomy:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    bases = {
        letter: BaseSpec(
            letter=letter,
            superscripts=list(spec.get("superscripts", [])),
            plain_latin=bool(spec.get("plain_latin", False)),
        )
        for letter, spec in raw["bases"].items()
    }

    return Taxonomy(
        bases=bases,
        trailing_numerals=list(raw["trailing_numerals"]),
        fraktur_letters=list(raw["fraktur_letters"]),
        declaratio_signorum=dict(raw["declaratio_signorum"]),
        # NFKC because macOS filenames and yaml text can differ in composed/decomposed form
        latin_letters=[unicodedata.normalize("NFKC", x) for x in raw.get("latin_letters", [])],
        # Relative reference_dirs resolve against the repository root,
        # so the training scripts work regardless of the current working directory.
        reference_dirs={k: Path(v) if Path(v).is_absolute()
                        else path.resolve().parent.parent.parent / v
                        for k, v in raw["reference_dirs"].items()},
        native_dpi=int(raw["native_dpi"]),
        measured_scale_px_native=dict(raw["measured_scale_px_native"]),
    )


if __name__ == "__main__":
    tax = load_taxonomy()
    print(f"{len(tax.bases)} base characters, "
          f"{len(tax.all_superscript_letters)} distinct superscript letters union, "
          f"{len(tax.fraktur_letters)} fraktur letters, "
          f"{len(tax.declaratio_signorum)} declaratio-signorum symbols.")
    for name, d in tax.reference_dirs.items():
        n = len(list(d.glob("*.png"))) if d.is_dir() else "MISSING DIR"
        print(f"reference_dirs[{name}] -> {n} png files")
