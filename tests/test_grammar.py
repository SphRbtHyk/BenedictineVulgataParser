"""Unit tests of the grammar stage (the formal apparatus parser).

The grammar is the ported implementation of the paper's parser; these tests
pin the behavior on small, hand-checked inputs.
"""

from __future__ import annotations

import pytest

from vulgata_scan.grammar import parse_apparatus, preprocess


def test_preprocess_marks_verses_and_chapters():
    s = preprocess("1. creavit $XB$.\n\nII. 1. in principio")
    assert "@@V1@@" in s
    assert "@@VXB" not in s  # sigla macro not a verse marker


def test_parse_simple_substitution():
    raw = "Desunt $G^{*}$.\n\n1. inan*s T | tenebrae + erant $G^{\\mathrm{C}}\\Lambda^{H}$."
    entries = parse_apparatus(raw, chapter="1")
    assert entries, "one verse entry expected"
    ch, verse = entries[0]["chapter"], entries[0]["verse"]
    assert (ch, verse) == ("1", "1")
    apps = entries[0]["apparatus"]
    types = [a["variant_type"] for a in apps]
    assert "substitution" in types
    assert "addition" in types
    sub = next(a for a in apps if a["variant_type"] == "substitution")
    assert sub["alternative_reading"]["witnesses"] == ["T"]
    add = next(a for a in apps if a["variant_type"] == "addition")
    assert "\\Lambda" in " ".join(add["alternative_reading"]["witnesses"])


def test_desunt_line_carries_missing_witnesses():
    raw = ("Desunt $\\Sigma^{T}\\Theta^{H} \\Psi^{B^*}$.\n\n"
           "2. deus $\\Lambda B$.")
    entries = parse_apparatus(raw, chapter="1")
    apps = entries[0]["apparatus"]
    assert apps and all("\\Sigma^T" in a["missing"] for a in apps)


def test_chapter_marker_switches_chapter():
    raw = "1. a $B$.\n\nII. 3. b $T$."
    entries = parse_apparatus(raw, chapter="1")
    chapters = [e["chapter"] for e in entries]
    assert chapters == ["1", "2"]


def test_witness_families_expand_to_atomic_members():
    # the released datasets store Greek families expanded to their members
    raw = "3. dixit $\\Phi^{RAZG}$."
    entries = parse_apparatus(raw, chapter="1")
    wits = entries[0]["apparatus"][0]["alternative_reading"]["witnesses"]
    assert wits == ["\\Phi^R", "\\Phi^A", "\\Phi^Z", "\\Phi^G"]
