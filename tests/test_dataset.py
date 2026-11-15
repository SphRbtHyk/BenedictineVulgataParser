"""Unit tests of the packaged dataset loader (gold + silver)."""

from __future__ import annotations

import pytest

from vulgata_scan.dataset import census, load_gold, load_silver, pretty_entry


def test_gold_loads():
    entries = load_gold()
    assert isinstance(entries, list) and len(entries) > 100
    first = entries[0]
    assert {"chapter", "verse", "apparatus"} <= set(first)
    unit = first["apparatus"][0]
    assert {"variant_type"} <= set(unit)


def test_silver_loads():
    entries = load_silver()
    assert len(entries) > 350
    # silver covers the whole book (Genesis 1-50), gold the reviewed chapters
    gold_ch = {e["chapter"] for e in load_gold()}
    assert gold_ch < {e["chapter"] for e in entries}


def test_census_counts():
    c = census(load_gold())
    assert c["verse_entries"] == 106
    assert "substitution" in c["variant_types"]
    assert "addition" in c["variant_types"]


def test_pretty_entry_renders():
    entry = {
        "chapter": "1", "verse": "3",
        "apparatus": [{"variant_type": "substitution",
                       "reference_reading": {"reading": "deus", "witnesses": []},
                       "alternative_reading": {"reading": "domino",
                                               "witnesses": ["B", "T"]},
                       "missing": ["G^{*}"]}],
    }
    out = pretty_entry(entry)
    assert "chapter 1, verse 3" in out
    assert "deus" in out and "domino" in out
    assert "B T" in out
    assert "desunt" in out


def test_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no repo-level dataset/ candidates either
    with pytest.raises(FileNotFoundError):
        load_gold(tmp_path / "nope.json")
