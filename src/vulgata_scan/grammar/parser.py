"""
apparatus_grammar.py
====================

Parse the critical apparatus of the Roman ("San Girolamo") Vulgate edition
(*Biblia Sacra iuxta latinam vulgatam versionem*), Genesis, from a lightly
LaTeX-flavoured transcription into a flat JSON model.

Pipeline
--------
1. ``preprocess(raw)``  canonicalises the LaTeX flavour to a plain token stream
   (see docs/grammar.md).  Line-initial verse numbers are marked ``@@Vn@@``;
   line-initial Roman chapter markers ``@@Cn@@``; ``Desunt`` lacuna declarations
   are kept inline as ``@@DESUNT@@ ... @@ENDDESUNT@@`` rather than dropped.
2. a **pyparsing** grammar (``segment_expr``) tokenises and *types* each
   apparatus segment (words, sigla, editions, versions, authors, signs).
3. ``classify_segment`` maps the typed tokens onto the reading model
   (``variant_type`` + flat ``witnesses`` list + ``reading``).

State carried across the stream
-------------------------------
``parse_apparatus`` walks the sentinel stream keeping two running values:

* **chapter** — updated by each ``@@Cn@@`` marker (default "1", since Genesis 1
  is unmarked in the source).
* **missing** — the set of witnesses declared lacunose by the most recent
  ``Desunt`` line, expanded to atomic members and carried forward until the
  next ``Desunt`` re-declares it (each ``Desunt`` *replaces* the set; a page
  with no ``Desunt`` inherits the previous one).  Attached to every reading as
  ``missing`` so that a witness's silence can be read three-valued
  (attests / dissents / inactive) rather than as plain negation.

The grammar deliberately keeps **witnesses flat and verbatim**: ``\\Phi^{RAZG}``
is one witness, not a family of four.
"""

from __future__ import annotations




import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pyparsing as pp

pp.ParserElement.enable_packrat()

# ---------------------------------------------------------------------------
# 0. Preprocessing: LaTeX-flavoured transcription -> canonical token stream
# ---------------------------------------------------------------------------

GREEK_UNICODE = {
    "Λ": r"\Lambda", "Π": r"\Pi", "Σ": r"\Sigma", "Φ": r"\Phi",
    "Θ": r"\Theta", "Ψ": r"\Psi", "Ω": r"\Omega", "Γ": r"\Gamma",
    "Δ": r"\Delta",
}

HOMOGLYPHS = {
    "Н": "H", "О": "O", "с": "c", "е": "e", "а": "a",
    "。": "", "＊": "*", "»": "»", "«": "«",
}

UNICODE_SUP = {
    "ᵢ": "i", "ᵇ": "b", "ᵃ": "a", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e",
    "¹": "1", "²": "2", "³": "3",
}

_FRAK_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Roman chapter numerals -> Arabic.  Genesis has 50 chapters; the table is
# generous so the same preprocessor serves later books.
_ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
    "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12, "L": 50,
}


def _chapter_sentinel(m: "re.Match") -> str:
    n = _ROMAN.get(m.group(1))
    if n is None:
        return m.group(0)
    return f" @@C{n}@@ @@V{m.group(2)}@@ "


def _desunt_sentinel(m: "re.Match") -> str:
    # keep the witness content inline so it is canonicalised by the steps
    # below; strip only the trailing sentence period / stray "$" / whitespace.
    content = re.sub(r"[.$\s]+$", "", m.group(1).strip())
    return f" @@DESUNT@@ {content} @@ENDDESUNT@@ "


def preprocess(raw: str) -> str:
    s = raw

    # 0. LaTeX line-breaks and transcriber stop-markers -> drop
    s = s.replace(r"\\", " ")
    s = re.sub(r"%+[^\n]*", " ", s)

    # 1. verse sentinels — line-initial "N." only (before newlines collapse)
    s = re.sub(r"(?m)^[ \t]*(\d+)\.[ \t]", r" @@V\1@@ ", s)

    # 2. chapter sentinels — line-initial Roman numeral + verse number
    #    ("II. 1." -> "@@C2@@ @@V1@@").  Runs after the verse step, which never
    #    matches these lines (they do not start with a digit).
    s = re.sub(
        r"(?m)^[ \t]*(VIII|VII|XII|III|XI|IX|IV|VI|II|I|V|X|L)\.[ \t]+(\d+)\.[ \t]",
        _chapter_sentinel, s)

    # 3. page markers -> drop (colon inside or outside the braces, or absent)
    s = re.sub(r"\\emph\{Page[^}]*\}:?", " ", s)

    # 4. "Desunt ... ." lacuna lines -> inline @@DESUNT@@ ... @@ENDDESUNT@@.
    #    Terminated at end-of-line (some notes contain "v." periods), matching
    #    plain "Desunt" and "\emph{Desunt}".  The witness content is kept and
    #    canonicalised by the steps below.
    s = re.sub(r"(?m)^[ \t]*(?:\\emph\{)?Desunt\}?[ \t]*(.*)$",
               _desunt_sentinel, s)

    # 5. homoglyphs and stray editorial word-markers
    for a, b in HOMOGLYPHS.items():
        s = s.replace(a, b)
    s = re.sub(r"([A-Za-zçœæ])\*\*+", r"\1", s)        # semetipso**

    # 6. unicode superscripts -> ^{x}
    for a, b in UNICODE_SUP.items():
        s = s.replace(a, "^{" + b + "}")

    # 7. unicode Greek -> macros
    for a, b in GREEK_UNICODE.items():
        s = s.replace(a, b)

    # 8. misc commands
    s = s.replace(r"\dagger", "†")
    s = re.sub(r"\\c\{e\}", "ę", s)

    # 9. unwrap \mathrm / \mathbf / \mathit
    for _ in range(3):
        s = re.sub(r"\\math(?:rm|bf|it)\{([^{}]*)\}", r"\1", s)

    # 10. \mathfrak{...} -> \frak{x}\frak{y}...   (expand runs, drop spaces)
    def _frak(m: re.Match) -> str:
        return "".join(rf"\frak{{{c}}}" for c in m.group(1) if c in _FRAK_LETTERS)
    s = re.sub(r"\\mathfrak\{([^{}]*)\}", _frak, s)

    # 11. merge chained brace superscripts: ^{A}^{M}^{*} -> ^{AM*}
    while re.search(r"\^\{([^{}]*)\}\^\{([^{}]*)\}", s):
        s = re.sub(r"\^\{([^{}]*)\}\^\{([^{}]*)\}", r"^{\1\2}", s)

    # 12. structural signs
    s = s.replace(r"\mid", "|").replace(r"\sim", "~")
    s = re.sub(r"\\P(?![A-Za-z])", "¶", s)

    # 13. LXX -> \LXX so it is never split into sigla L,X,X
    s = re.sub(r"\bLXX\b", r"\\LXX", s)

    # 14. drop math delimiters, collapse whitespace
    s = s.replace("$", " ")
    s = re.sub(r"[ \t\r\n]+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# 1. pyparsing grammar for a single segment
# ---------------------------------------------------------------------------

# --- lexical building blocks ----------------------------------------------
# A superscript may itself contain simple braced modifiers, as in
# ``\Pi^{C^{*}D^*}``.
_SUP = r"(?:\^\{(?:[^{}]|\{[^{}]*\})*\}|\^[A-Za-z0-9*†])"
_GREEK = r"\\(?:Lambda|Pi|Sigma|Phi|Theta|Psi|Omega|Gamma|Delta)"
_FRAK = r"\\frak\{[a-z]\}"
_LATIN = r"[ABCGMOPTX](?![a-z])"
_SIGLUM = rf"(?:{_GREEK}|{_FRAK}|{_LATIN}){_SUP}?\*?"

# indirect (non-manuscript) witnesses: ancient versions
_VERSION_NORM = {
    "hebr.": "hebrew",
    "\\LXX": "LXX",
    "graeci": "greek",
    "antiqui": "OL",
    "ant.": "OL",
    "vers.": "OL",
}


def _canon_witness(raw: str) -> str:
    def canon_superscript(value: str) -> str:
        """Uppercase sigla and put a hand star in its own superscript."""
        # Nested braces around a star carry no additional information:
        # C^{*} and C^* have the same canonical representation.
        value = value.replace("^{*}", "^*")
        # Uppercase letters belonging to the siglum, but leave LaTeX command
        # names such as ``\dagger`` untouched.
        value = re.sub(
            r"\\[A-Za-z]+|[A-Za-z]",
            lambda match: (
                match.group(0)
                if match.group(0).startswith("\\")
                else match.group(0).upper()
            ),
            value,
        )
        if value != "*" and value.endswith("*") and not value.endswith("^*"):
            return value[:-1] + "^*"
        return value

    raw = raw.strip()
    m = re.match(rf"({_GREEK}|{_FRAK}|[ABCGMOPTX])(.*)$", raw)
    if not m:
        return raw
    base, rest = m.group(1), m.group(2).strip()
    if rest == "":
        return base
    if rest == "*":
        return base + "^{*}"
    if rest.startswith("^{"):
        match = re.fullmatch(r"\^\{(.*)\}(\*)?", rest)
        if match:
            superscript, trailing_star = match.groups()
            if trailing_star:
                superscript += "*"
            return base + "^{" + canon_superscript(superscript) + "}"
        return base + rest
    if rest.startswith("^"):
        return base + "^{" + canon_superscript(rest[1:]) + "}"
    return base + rest


_GREEK_BASE = re.compile(
    r"^(\\(?:Lambda|Pi|Sigma|Phi|Theta|Psi|Omega|Gamma|Delta))\^\{(.+)\}$")

# Atomic witness identifiers permitted by the edition.
VALID_WITNESS_INVENTORY = set([
    "G",
    "\\G^{C}",
    "C",
    "\\Lambda",
    "\\Lambda^L",
    "\\Lambda^H",
    "X",
    "\\Pi",
    "\\Pi^C",
    "\\Pi^D",
    "\\Sigma",
    "\\Sigma^T",
    "\\Sigma^O",
    "\\Sigma^M",
    "B",
    "A",
    "T",
    "M",
    "\\Phi",
    "\\Phi^R",
    "\\Phi^A",
    "\\Phi^Z",
    "\\Phi^G",
    "\\Phi^V",
    "\\Phi^P",
    "O",
    "\\Theta",
    "\\Theta^H",
    "\\Theta^A",
    "\\Theta^M",
    "\\Theta^G",
    "P",
    "\\Psi",
    "\\Psi^B",
    "\\Psi^D",
    "\\Psi^F",
    "\\Psi^M",
    "\\Psi^Z",
    "\\Omega",
    "\\Omega^S",
    "\\Omega^M",
    "\\Omega^J",
    "\\frak{a}",
    "\\frak{g}",
    "\\frak{r}",
    "\\frak{e}",
    "\\frak{l}",
    "\\frak{v}",
    "\\frak{w}",
    "\\frak{s}",
    "\\frak{c}",
])


def expand_witness(w: str) -> List[str]:
    """Expand a Greek family siglum whose superscript lists member manuscripts.

    ``\\Theta^{AB}`` -> ``\\Theta^A``, ``\\Theta^B``.  A member letter is an
    uppercase character; any following hand modifier (``*``, digits, ``^n``,
    ``\u2020``, or a lowercase corrector letter) stays attached to it and is
    *not* split off: ``\\Theta^{AB1}`` -> ``\\Theta^A``, ``\\Theta^{B1}``.

    Latin sigla (``P^{2}``), bare families (``\\Phi``) and editions
    (``\\frak{a}``) are single manuscripts and are returned unchanged.
    """
    m = _GREEK_BASE.match(w)
    if not m:
        return [w]
    base, sup = m.group(1), m.group(2)
    out = []
    pos = 0
    nested_family = re.compile(rf"{_GREEK}{_SUP}?")
    member = re.compile(
        r"[A-Z](?:(?:\^(?:\*|\d+|\\dagger|†))|\*|\d+|\\dagger|†|[a-z])*"
    )

    while pos < len(sup):
        # A Greek family appearing inside another family's superscript is a
        # separate witness group: \Pi^{C^2\Sigma^M} expands to \Pi^{C^2}
        # and \Sigma^M, not into letters taken from the word "Sigma".
        nested = nested_family.match(sup, pos)
        if nested:
            nested_witness = _canon_witness(nested.group(0))
            out.extend(expand_witness(nested_witness))
            pos = nested.end()
            continue

        manuscript = member.match(sup, pos)
        if manuscript:
            mem = manuscript.group(0)
            out.append(
                f"{base}^{mem}" if len(mem) == 1 else f"{base}^{{{mem}}}"
            )
            pos = manuscript.end()
            continue

        # Ignore grouping whitespace/braces left by noisy transcription, but
        # retain the original witness if nothing recognisable can be split.
        pos += 1

    if not out:                          # e.g. \Phi^{*} — nothing to expand
        return [w]
    return out


def enforce_witness_inventory(witnesses: List[str], force: bool) -> List[str]:
    """Replace atomic witnesses absent from the inventory with ``?``.

    Stars, numbers, daggers, and corrector ``C`` denote a state/hand of a
    witness, so they do not take part in the inventory lookup.  The original
    spelling is kept in the returned list: for example, ``G^{*}`` is checked
    as ``G`` and ``M^{2}`` is checked as ``M``.
    """
    if not force:
        return witnesses

    def inventory_key(witness: str) -> str:
        match = re.fullmatch(r"(.+)\^\{([^{}]+)\}", witness)
        if match:
            base, superscript = match.groups()
            greek_family = re.fullmatch(_GREEK, base) is not None
            # Standalone superscripts are hand/state markers.  Keep C when it
            # is the member of a Greek family (e.g. \Pi^{C^*}), but ignore it
            # when it is the complete superscript of an atomic witness.
            if re.fullmatch(r"(?:\*|\^?\d+|\^?\\dagger|\^?†)", superscript):
                return base

            # \Pi^{C^*}/\Pi^{C^2}/\Pi^{C^\dagger} -> \Pi^C
            superscript = re.sub(
                r"(?:\^\*|\*|\^?\d+|\^?\\dagger|\^?†)$", "", superscript
            )
            if not greek_family and superscript in {"C", "c"}:
                return base
            return base if not superscript else f"{base}^{superscript}"

        # Be tolerant of unbraced forms supplied directly by callers.
        unbraced = re.fullmatch(r"(.+)\^([^{}]+)", witness)
        if unbraced:
            base, superscript = unbraced.groups()
            greek_family = re.fullmatch(_GREEK, base) is not None
            if re.fullmatch(r"(?:\*|\d+|\\dagger|†)", superscript):
                return base
            if not greek_family and superscript in {"C", "c"}:
                return base
        return witness[:-1] if witness.endswith("*") else witness

    inventory = {inventory_key(w) for w in VALID_WITNESS_INVENTORY}
    return [w if inventory_key(w) in inventory else "?" for w in witnesses]


def _tok(kind):
    def action(s, l, t):
        return (kind, t[0])
    return action


# order matters: specific / keyword-like before generic word
edition = pp.Regex(_FRAK).add_parse_action(lambda s, l, t: ("WIT", t[0]))
siglum = pp.Regex(_SIGLUM).add_parse_action(
    lambda s, l, t: ("WIT", _canon_witness(t[0])))

version = (pp.Regex(r"\\LXX") | pp.Keyword("hebr.") | pp.Keyword("graeci")
           | pp.Keyword("antiqui") | pp.Keyword("ant.") | pp.Keyword("vers.")
           ).add_parse_action(_tok("VER"))

abbr = pp.Regex(r"(?<![ABCGMOPTX])[A-Z]\.").add_parse_action(_tok("ABBR"))
# An initial such as ``M.`` is text, but the final letter of a contiguous
# witness run is still a siglum: ``XB.`` must tokenize as X, B, '.'.
# Any capitalised token that is not a siglum: authority names, work titles.
# Deliberately lexicon-free -- these are kept as text, never classified.
capword = pp.Regex(r"[A-Z][A-Za-z\u00e0-\u017f]*\.?").add_parse_action(_tok("WORD"))

kw = (pp.Keyword("cum") | pp.Keyword("cf.") | pp.Keyword("Cf.")
      | pp.Keyword("ex")).add_parse_action(
          lambda s, l, t: ("KW", t[0].lower().rstrip(".")))

om = pp.Keyword("om.").add_parse_action(lambda s, l, t: ("MARK", "om"))
sign = pp.one_of("~ + = ¶").add_parse_action(_tok("MARK"))
dash = pp.Literal("—").add_parse_action(_tok("DASH"))

num = pp.Regex(r"\d+").add_parse_action(_tok("NUM"))

_L = r"a-zçœæ\u00e0-\u017f"    # lowercase Latin incl. Latin-1 supp. + Extended-A
gloss = pp.Regex(r"\([^)]*\s[^)]*\)").add_parse_action(_tok("GLOSS"))
word = pp.Regex(rf"\(?[{_L}][{_L}()]*\.?").add_parse_action(_tok("WORD"))

quote = pp.one_of("« »").add_parse_action(_tok("WORD"))
comma = pp.Literal(",").add_parse_action(_tok("PUNCT"))
other = pp.Regex(r"\S").add_parse_action(_tok("OTHER"))   # robustness fallback

element = (om | kw | version | abbr | sign | dash | num | comma
           | gloss | siglum | edition | quote | word | capword | other)

segment_expr = pp.ZeroOrMore(element)
segment_expr.set_name("segment")


def tokenize_segment(text: str) -> List[Tuple[str, str]]:
    return list(segment_expr.parse_string(text, parse_all=True))


# ---------------------------------------------------------------------------
# 2. Reading model + classifier
# ---------------------------------------------------------------------------

_ANNOT_PHRASES = {
    ("in", "ras."): "in ras.",
    ("bis", "rep."): "bis rep.",
    ("ut", "videtur."): "ut videtur",
    ("ut", "videtur"): "ut videtur",
}


@dataclass
class Reading:
    variant_type: str
    witnesses: List[str] = field(default_factory=list)
    reading: Optional[str] = None
    in_text: Optional[str] = None
    anchor: Optional[str] = None
    annotations: List[str] = field(default_factory=list)
    versions: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    note: Optional[str] = None
    location: str = "main"
    # witnesses declared lacunose (inactive) at this locus by the governing
    # Desunt line -- their silence in this reading is "no data", not dissent.
    missing: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        alt: dict = {"witnesses": self.witnesses}
        if self.reading is not None:
            alt["reading"] = self.reading
        d: dict = {"missing": self.missing, "variant_type": self.variant_type}
        if self.in_text:
            d["in_text"] = self.in_text
        if self.anchor:
            d["anchor"] = self.anchor
        d["alternative_reading"] = alt
        if self.annotations:
            d["annotations"] = self.annotations
        if self.versions:
            d["versions"] = self.versions
        if self.citations:
            d["citations"] = self.citations
        if self.note:
            d["note"] = self.note
        if self.location != "main":
            d["location"] = self.location
        return d


def _join_words(vals: List[str]) -> str:
    out = " ".join(vals)
    out = re.sub(r"\s+([),.])", r"\1", out)
    out = re.sub(r"([(])\s+", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out or None


# sentinel: segment only continues the previous reading's citation run
CONTINUATION = "__CONT__"


def extract_versions(tokens: List[Tuple[str, str]]) -> List[str]:
    """Return the ancient versions named in a span, normalised.

    The version vocabulary is closed -- ``hebrew``, ``LXX``, ``OL``
    (*antiquae versiones* / Vetus Latina), ``greek`` -- so this is the one
    part of the indirect attestation that is worth structuring. The two-token
    sequence ``ant. vers.`` denotes a single version and is merged.
    """
    out: List[str] = []
    i = 0
    while i < len(tokens):
        kind, val = tokens[i]
        if kind == "VER":
            if val == "ant." and i + 1 < len(tokens) \
                    and tokens[i + 1][1] == "vers.":
                name, i = "OL", i + 2
            else:
                name, i = _VERSION_NORM.get(val, val), i + 1
            if name not in out:
                out.append(name)
            continue
        i += 1
    return out


def citation_text(tokens: List[Tuple[str, str]]) -> Optional[str]:
    """Return the non-versional remainder of a span as a verbatim string.

    Patristic citations are *not* segmented into author, work and locus. They
    are of little use for classifying manuscript witnesses, and parsing them
    would require an authority lexicon that cannot be compiled reliably from
    the sample. Keeping the span verbatim loses nothing: the string can be
    re-parsed later without returning to the source.
    """
    keep: List[str] = []
    i = 0
    while i < len(tokens):
        kind, val = tokens[i]
        if kind == "KW":                      # drop the introducing cum / cf.
            i += 1
            continue
        if kind == "VER":                     # already captured as a version
            if val == "ant." and i + 1 < len(tokens) \
                    and tokens[i + 1][1] == "vers.":
                i += 2
            else:
                i += 1
            continue
        keep.append(val)
        i += 1
    # strip connectives left stranded by the removals
    while keep and keep[0] in (",", "et"):
        keep.pop(0)
    while keep and keep[-1] in (",", "et"):
        keep.pop()
    return _join_words(keep)


def classify_segment(tokens: List[Tuple[str, str]], in_text: Optional[str],
                     location: str, force_inventory: bool = False):
    if not tokens:
        return None
    kinds = [t for t, _ in tokens]
    n = len(tokens)

    # index of the last witness, and of the first citation/agreement start
    wit_positions = [i for i, k in enumerate(kinds) if k == "WIT"]
    last_wit = wit_positions[-1] if wit_positions else -1

    # A segment with no witness and no mark that names an authority is a pure
    # citation continuing the previous reading (e.g. "; de gen. c. Man. 14").
    # The whole segment is the citation: leading work-title words included.
    # A ";"-segment with no siglum and no mark that carries a numeric locus is
    # a citation continuing the previous reading ("; de gen. c. Man. 14";
    # "; conf. 13, 18").  No authority lexicon is needed to recognise it: the
    # locus and the absence of witnesses are enough.
    _no_mark = not any(k == "MARK" for k, _ in tokens)
    _has_locus = any(k == "NUM" for k, _ in tokens)
    _text_only = all(k in ("WORD", "NUM", "PUNCT", "GLOSS", "ABBR", "VER")
                     for k, _ in tokens)
    if last_wit < 0 and _no_mark and _has_locus and _text_only \
            and location == "main":
        return (CONTINUATION, extract_versions(tokens), citation_text(tokens))

    src_start = None
    ex_start = None
    for i, (k, v) in enumerate(tokens):
        if k == "KW" and v in ("cum", "cf"):
            src_start = i
            break
        if k == "KW" and v == "ex":
            ex_start = i          # genetic note, not an indirect witness
            break


    cut = src_start if src_start is not None else ex_start
    core = tokens if cut is None else tokens[:cut]
    src_tokens = [] if src_start is None else tokens[src_start:]
    ex_tokens = [] if ex_start is None else tokens[ex_start:]

    # pull annotation phrases out of the core token stream
    annotations: List[str] = []
    cleaned: List[Tuple[str, str]] = []
    i = 0
    core_vals = [v for _, v in core]
    while i < len(core):
        pair = (core_vals[i], core_vals[i + 1]) if i + 1 < len(core) else None
        if pair in _ANNOT_PHRASES:
            annotations.append(_ANNOT_PHRASES[pair])
            i += 2
            continue
        if core[i][0] == "WORD" and core[i][1] == "item":     # "item bis v. seq."
            annotations.append(_join_words([v for _, v in core[i:]]))
            i = len(core)
            continue
        if core[i][0] == "GLOSS":
            annotations.append(core[i][1])
            i += 1
            continue
        cleaned.append(core[i])
        i += 1

    witnesses = [x for k, v in cleaned if k == "WIT" for x in expand_witness(v)]
    witnesses = enforce_witness_inventory(witnesses, force_inventory)
    marks = {v for k, v in cleaned if k == "MARK"}

    # versions (structured, closed vocabulary) and citations (verbatim text)
    indirect: List[str] = []
    citations: List[str] = []
    if src_tokens:
        if location == "margins":
            # inside a scholion the span is discursive prose, not attestation
            annotations.append(_join_words([v for _, v in src_tokens]))
        elif any(k == "WORD" and v in ("v.", "uers.", "versu")
                 for k, v in src_tokens[:3]):
            # "cf. v. 4" -- an internal cross-reference, not an attestation
            annotations.append(_join_words([v for _, v in src_tokens]))
        else:
            indirect = extract_versions(src_tokens)
            cit = citation_text(src_tokens)
            if cit:
                citations.append(cit)

    # "ex versu praec." is a genetic note explaining how the variant arose;
    # any version named inside it is still an indirect witness.
    if ex_tokens:
        _verish = {"vers", "vers.", "ant.", "ant", "antiqui", "hebr.",
                   "graeci"}
        ver_part = [t for t in ex_tokens
                    if t[0] == "VER" or (t[0] == "WORD" and t[1] in _verish)]
        note_part = [t for t in ex_tokens if t not in ver_part]
        while note_part and note_part[-1][1] in ("et", ",", "and"):
            note_part.pop()
        note_txt = _join_words([v for _, v in note_part])
        if note_txt:
            annotations.append(note_txt)
        if ver_part:
            for name in extract_versions([("VER", v) for _, v in ver_part]):
                if name not in indirect:
                    indirect.append(name)

    # --- determine variant type & reading/anchor text ---------------------
    def words_between(lo, hi):
        return _join_words([v for (k, v) in cleaned[lo:hi]
                            if k in ("WORD", "DASH")])

    reading = None
    anchor = None
    vtype = "substitution"

    mark_index = {v: i for i, (k, v) in enumerate(cleaned) if k == "MARK"}

    if "¶" in marks and cleaned and cleaned[0][0] == "MARK" \
            and cleaned[0][1] == "¶":
        cleaned = cleaned[1:]

    if "~" in marks:
        vtype = "transposition"
        idx = mark_index["~"]
        reading = words_between(idx + 1, len(cleaned))
    elif "+" in marks:
        vtype = "addition"
        idx = mark_index["+"]
        anchor = words_between(0, idx)
        # reading = added words after '+', before witnesses
        first_wit = next((j for j in range(idx + 1, len(cleaned))
                          if cleaned[j][0] == "WIT"), len(cleaned))
        reading = words_between(idx + 1, first_wit)
    elif "om" in marks:
        vtype = "omission"
        idx = mark_index["om"]
        reading = words_between(0, idx)          # omitted text if written out
    elif "=" in marks:
        vtype = "positive_apparatus"
        idx = mark_index["="]
        reading = words_between(0, idx)
    else:
        # substitution / in_text-support / margins-note
        first_wit = next((j for j in range(len(cleaned))
                          if cleaned[j][0] == "WIT"), len(cleaned))
        reading = words_between(0, first_wit)
        if not witnesses and reading:
            # no witnesses at all -> editorial note (only meaningful under ¶)
            r = Reading(variant_type="note", note=reading, location=location)
            r.versions = indirect
            r.citations = citations
            r.annotations = annotations
            return r
        if not reading:
            vtype = "in_text_support"
            reading = None      # witnesses attest the printed text: no
            in_text = None        # *alternative* reading is reported here

    # a segment that is purely a citation continuation
    if not witnesses and vtype in ("substitution", "in_text_support") \
            and not reading:
        return (CONTINUATION, indirect, citations)
    if not witnesses and not cleaned and (indirect or citations):
        return (CONTINUATION, indirect, citations)

    r = Reading(variant_type=vtype, witnesses=witnesses, reading=reading,
                in_text=in_text, anchor=anchor, annotations=annotations,
                versions=indirect, citations=citations,
                location=location)
    return r


# ---------------------------------------------------------------------------
# 3. Unit / verse / apparatus assembly
# ---------------------------------------------------------------------------

def parse_unit(unit_raw: str, force_inventory: bool = False) -> List[Reading]:
    """Parse one '|'-delimited unit into a list of Reading objects."""
    unit_raw = unit_raw.strip()
    if unit_raw.endswith(".") and not re.search(r"[a-z]{1,6}\.$", unit_raw[-8:]):
        unit_raw = unit_raw[:-1]
    if not unit_raw:
        return []

    in_text = None
    body = unit_raw
    if "]" in unit_raw:
        in_text_part, body = unit_raw.split("]", 1)
        in_text = _join_words([v for k, v in tokenize_segment(in_text_part.strip())
                             if k in ("WORD", "DASH", "GLOSS")]) or None

    # split into segments on ';' and '¶' (¶ switches location)
    segments: List[Tuple[str, str]] = []
    location = "main"
    buf = ""
    j = 0
    while j < len(body):
        ch = body[j]
        if ch == ";":
            segments.append((location, buf))
            buf = ""
        elif ch == "¶":
            if buf.strip():
                segments.append((location, buf))
            location = "margins"
            buf = ""
        else:
            buf += ch
        j += 1
    if buf.strip():
        segments.append((location, buf))

    readings: List[Reading] = []
    for reg, seg_raw in segments:
        seg_raw = seg_raw.strip()
        if not seg_raw:
            continue
        toks = tokenize_segment(seg_raw)
        result = classify_segment(toks, in_text, reg, force_inventory)
        if result is None:
            continue
        if isinstance(result, tuple) and result[0] == CONTINUATION:
            if readings:
                for name in (result[1] or []):
                    if name not in readings[-1].versions:
                        readings[-1].versions.append(name)
                cit = result[2]
                if isinstance(cit, str):
                    cit = [cit]
                readings[-1].citations.extend(c for c in (cit or []) if c)
            continue
        readings.append(result)
    return readings


# --- Desunt (lacuna) declarations -----------------------------------------

def parse_desunt(content: str, force_inventory: bool = False) -> List[str]:
    """Extract the witnesses declared lacunose in a Desunt span.

    The content has already been canonicalised by ``preprocess`` (Greek macros,
    superscripts, homoglyphs).  Parenthetical scope notes such as
    ``(usque ad 22, crescite et)`` tokenise as GLOSS and are ignored: only the
    witness sigla are taken, expanded to atomic members (``\\Theta^{HG}`` ->
    ``\\Theta^H``, ``\\Theta^G``) for consistency with the reading witnesses.
    """
    out: List[str] = []
    for k, v in tokenize_segment(content):
        if k == "WIT":
            for x in expand_witness(v):
                if x not in out:
                    out.append(x)
    return enforce_witness_inventory(out, force_inventory)


# structural sentinels emitted by preprocess
_STRUCT = re.compile(r"@@V(\d+)@@|@@C(\d+)@@")
_DESUNT = re.compile(r"@@DESUNT@@(.*?)@@ENDDESUNT@@")


def _process_body(body: str, chapter: str, verse: Optional[str],
                  state: Dict[str, List[str]], get_entry,
                  force_inventory: bool = False) -> None:
    """Parse one inter-sentinel body, tagging each reading with the running
    ``missing`` set.

    A ``Desunt`` sentinel may fall inside a body (a page break landing
    mid-verse).  The body is kept contiguous for '|'-unit parsing -- the
    sentinel is only *stripped*, never used to split a unit -- so readings that
    span a page break are not fragmented.  The missing-set update takes effect
    at unit granularity: a Desunt that leads a unit applies to it; otherwise it
    applies from the following unit (so a unit straddling the break keeps the
    pre-break set).
    """
    if not body:
        return
    apps = None
    for uraw in body.split("|"):
        desunts = list(_DESUNT.finditer(uraw))
        pending: Optional[List[str]] = None
        if desunts:
            pending = parse_desunt(desunts[-1].group(1), force_inventory)
            leading = uraw[:desunts[0].start()].strip() == ""
            uraw = _DESUNT.sub(" ", uraw)
            if leading:
                state["missing"] = pending
                pending = None
        for r in parse_unit(uraw, force_inventory):
            if apps is None:
                apps = get_entry(chapter, verse)
            r.missing = list(state["missing"])
            apps.append(r.to_dict())
        if pending is not None:
            state["missing"] = pending


def parse_apparatus(raw: str, chapter: str = "1",
                    force_inventory: bool = False) -> List[dict]:
    """Parse a (possibly multi-chapter) apparatus transcription.

    ``chapter`` is the *starting* chapter (Genesis 1 is unmarked in the
    source); ``@@Cn@@`` markers advance it.  The lacuna ``missing`` set is
    carried across verses and chapters until a new ``Desunt`` re-declares it.
    """
    text = preprocess(raw)
    chapter = str(chapter)
    state: Dict[str, List[str]] = {"missing": []}

    order: List[Tuple[str, Optional[str]]] = []
    table: Dict[Tuple[str, Optional[str]], List[dict]] = {}

    def get_entry(ch, vs):
        key = (ch, vs)
        if key not in table:
            table[key] = []
            order.append(key)
        return table[key]

    verse: Optional[str] = None
    last = 0
    for m in _STRUCT.finditer(text):
        _process_body(text[last:m.start()], chapter, verse, state, get_entry,
                      force_inventory)
        if m.group(1) is not None:          # @@Vn@@
            verse = m.group(1)
        else:                               # @@Cn@@ — opens a new chapter
            chapter = m.group(2)
            verse = None
        last = m.end()
    _process_body(text[last:], chapter, verse, state, get_entry,
                  force_inventory)

    entries: List[dict] = []
    for ch, vs in order:
        apps = table[(ch, vs)]
        if apps:
            entries.append({"chapter": ch, "verse": vs, "apparatus": apps})
    return entries


# ---------------------------------------------------------------------------

