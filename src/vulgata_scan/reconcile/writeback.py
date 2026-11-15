"""The reconciliation stage merges the classifier's decoded sigla back into the VLM output
and emits the corrected transcription. Three output files per run:

  <stem>_corrected_latex.json     canonical corrected transcription, LaTeX notation
  <stem>_corrected_unicode.md     human-readable Unicode view
  <report-stem>_writeback.json    per-region diff info + review list (make_report.py input)

GLM gives no character boxes, so alignment works on sigla occurrences in reading order.
Only matched spans are replaced; unmatched material is left untouched and sent to the review list.

Step map (module-level helpers, then the per-region driver writeback_region):
  1 normalization + span extraction  split_sup_groups, _serialization_cleanup, glm_sigla_spans
  2 pre-alignment token gates        _implausible_siglum, _geometry_veto, _embedded_w_veto
  3 coordinate estimation            _ink_words, _word_boxes, _span_bracket
  4 alignment                        _key_cost, _global_align, _geom_rescue, difflib route
  5 post-pairing adjudication        _mods_conflict, _frak_ok, superset guard, double-write removal
  6 recovery of unmarked material    fraktur gap interpolation, '|' insertion, */† absorption
  rendering                          recon_to_latex/recon_to_unicode, fold_to_unicode,
                                     _polish_tex/_polish_uni, _vd_fix_tex/_vd_fix_uni
"""
from __future__ import annotations

import os
import difflib
import html
import json
import re
import unicodedata
from pathlib import Path

from ..classifier.taxonomy import _FRAKTUR_LOWER, _SUPERSCRIPT_LETTER_TO_CHAR, load_taxonomy

_GREEK_TO_LATEX = {"Θ": r"\Theta", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
                   "Λ": r"\Lambda", "Σ": r"\Sigma", "Π": r"\Pi"}
_LATEX_TO_GREEK = {v.lstrip("\\").lower(): k for k, v in _GREEK_TO_LATEX.items()}  # theta -> Θ
_FRAK_CHAR_TO_LETTER = {v: k for k, v in _FRAKTUR_LOWER.items()}
_DECL_TO_LATEX = {"†": r"\dagger", "¶": r"\P", "~": r"\sim", "+": "+", "]": "]", "*": "*",
                  "|": "|"}
# "=" (the positive-apparatus sign) deliberately has NO slot here or in _STANDALONE,
# because it has no CNN class and the VLM reads it reliably,
# so a reserved span could never pair and only generated review noise (35 of 779 items).

_TAX = load_taxonomy()

# Sigla attested in the printed apparatus (Σ^I, Σ^B, Θ^F) but absent from the
# reference-crop inventory. They are legal for the legality check yet not trainable
# classes, so _illegal_mods must never flag a span merely outside the training alphabet.
_EXTRA_LEGAL = {"Σ": {"I", "B"},
                "Θ": {"F"}}

# bare witness capitals attested in the ground truth,
# used by the over-grouped-span split to decide whether a modifier letter the
# classifier did NOT see could instead be a swallowed NEIGHBOURING witness
_WITNESS_SET = set("GXBATMOPC")


def _illegal_mods(base_key: str, mods: str) -> bool:
    """True when the modifiers contain a letter the grammar forbids for this base.
    GLM habitually lowercases legal modifiers, so only a case-folded letter outside
    the base's alphabet counts. Digits, '*' and '†' are always legal; unknown bases never flag."""
    spec = _TAX.bases.get(base_key)
    if spec is None:
        return False
    legal = set(spec.superscripts) | _EXTRA_LEGAL.get(base_key, set())
    for ch in mods:
        for lit in unicodedata.normalize("NFKC", ch):
            if lit.isalpha() and lit.upper() not in legal:
                return True
    return False


# ---------------------------------------------------------------------------
# classifier token -> output notations + alignment key
# ---------------------------------------------------------------------------

def token_key(tok: dict) -> str:
    """Return the alignment key, one character summarizing what the token IS."""
    p = tok["prediction"]
    fam = p["family"]
    if fam == "manuscript_siglum":
        return p["base"]
    if fam == "declaratio_signorum":
        return p["symbol"]
    if fam == "fraktur_cluster":
        # single letter -> the letter itself, so \mathfrak{s} pairs with the right
        # token instead of any 𝔉 whatsoever
        s = tok["reconstructed_string"]
        return s if len(s) == 1 else "𝔉"
    if fam == "latin_letter":
        return "Ł"
    return "?"


def recon_to_latex(tok: dict, in_math: bool = False) -> str:
    def wrap(s: str) -> str:
        return s if in_math or "\\" not in s and "^" not in s else "$%s$" % s

    p = tok["prediction"]
    fam = p["family"]
    recon = tok["reconstructed_string"]
    if fam == "manuscript_siglum":
        base = p["base"]
        base_tex = _GREEK_TO_LATEX.get(base, r"\mathrm{%s}" % base)
        mods = ""
        for ch in recon[1:]:
            if ch == "†":
                mods += r"\dagger "
            else:
                # NFKC folds a modifier letter or digit to its literal form;
                # letters normalize to uppercase (the internal label case)
                lit = unicodedata.normalize("NFKC", ch)
                mods += lit.upper() if lit.isalpha() else lit
        return wrap("%s^{\\mathrm{%s}}" % (base_tex, mods) if mods else base_tex)
    if fam == "fraktur_cluster":
        letters = "".join(_FRAK_CHAR_TO_LETTER.get(c, c) for c in recon)
        return wrap("\\mathfrak{%s}" % letters)
    if fam == "declaratio_signorum":
        return wrap(_DECL_TO_LATEX.get(recon, recon))
    # latin_letter tokens never pair (token_key "Ł" has no GLM-side span), so
    # they can only reach the review list; no latex serialization path exists.
    return recon


def recon_to_unicode(tok: dict) -> str:
    return tok["reconstructed_string"]


# ---------------------------------------------------------------------------
# GLM content normalization: split merged superscript groups
# ---------------------------------------------------------------------------

_WRAP_MACRO = re.compile(r"\\(?:math(?:rm|bf|it)|text(?:bf|it|rm)?)")


def split_sup_groups(content: str) -> str:
    """Rewrite every multi-symbol superscript group as one ^{x} per symbol.
    GLM merges whole stacks into one group (\\Phi^{RAZG}, ^{z \\dagger v}), nested to any depth.
    A Greek capital inside a group is the next siglum glued in; a Greek base is never a modifier,
    so it breaks out as a plain base (G^{c\\Omega} -> G^{c}\\Omega).
    All later steps operate on this split form."""
    def units(inner: str) -> list[str]:
        out, i = [], 0
        while i < len(inner):
            ch = inner[i]
            if ch.isspace() or ch in "^{}":
                i += 1
            elif ch == "\\":
                m = re.match(r"\\[A-Za-z]+", inner[i:])
                if m is None:
                    i += 1      # a lone backslash before a non-letter (^{\}, ^{\ })
                    continue    # is GLM garbage rather than a macro, so drop it
                if not _WRAP_MACRO.fullmatch(m.group(0)):
                    out.append(m.group(0))  # wrapper macros drop; their braces
                i += m.end()                # are skipped by the branch above
            else:
                out.append(ch)
                i += 1
        return out

    def is_greek(u: str) -> bool:
        return u in _GREEK_TO_LATEX or u.lstrip("\\").lower() in _GREEK_MACROS

    out, i, n = [], 0, len(content)
    while i < n:
        if content[i] == "^" and content[i + 1:i + 2] == "{":
            depth, j = 1, i + 2
            while j < n and depth:
                depth += {"{": 1, "}": -1}.get(content[j], 0)
                j += 1
            if depth:  # unbalanced group from GLM truncation, leave untouched
                out.append(content[i:])
                break
            us = units(content[i + 2:j - 1])
            if len(us) <= 1 and not any(map(is_greek, us)):
                out.append(content[i:j])
            else:
                out.append("".join(u if is_greek(u) else "^{%s}" % u for u in us))
            i = j
        else:
            out.append(content[i])
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# GLM content -> candidate sigla spans
# ---------------------------------------------------------------------------

# a "unit" inside a math segment is an optional \macro or letter base plus ^{...} chains
_MATH_UNIT = re.compile(
    r"(\\[A-Za-z]+(?:\{[^{}]*\})?|[A-Za-zΘΦΨΩΛΣΠ])"           # base
    r"((?:\s*[\^_]\s*(?:\{[^{}]*(?:\{[^{}]*\})?[^{}]*\}|\\?[A-Za-z0-9*†]))*)"  # sup/sub chain
    # † sits in the bare-char alternative because X^† used to break the chain
)
_SUP_TAG = re.compile(r"([A-Za-zΘΦΨΩΛΣΠ])((?:<su[pb]>.*?</su[pb]>)+)", re.S)
# bare ASCII caret in plain text (Σ^T outside any $...$ or <sup>). Without this,
# the unicode-siglum scan caught only the base and left a stray '^T' behind.
_BARE_CARET = re.compile(r"([A-ZΘΦΨΩΛΣΠ])((?:\^(?:\{[^{}]*\}|[A-Za-z0-9*†]))+)")
# ™ = GLM's superscript-M garble. ᶜ-ᶻ (U+1D9C-1DBB) and ˢ (U+02E2) hold modifier
# letters outside the ᴬ-ᵸ block. A bare Latin capital is ordinary text,
# so the Latin alternative requires at least one modifier (+ not *).
_UNI_SIGLUM = re.compile(r"([ΘΦΨΩΛΣΠ])([ᴬ-ᵸᶜ-ᶻⱽꟳˢ¹²³*†™]*)"
                         r"|([A-Z])([ᴬ-ᵸᶜ-ᶻⱽꟳ¹²³*†™]+)")
_UNI_FRAK = re.compile(r"[\U0001D51E-\U0001D537]+")
# one span per '*', so a '**' run can match two classifier '*' tokens;
# '=' deliberately absent (see the _DECL_TO_LATEX note)
_STANDALONE = re.compile(r"[†¶~+\]*|]")

_GREEK_MACROS = {"theta", "phi", "psi", "omega", "lambda", "sigma", "pi"}


def _math_unit_key(base: str, sups: str) -> str | None:
    """Key for a unit inside $...$; None = not a siglum candidate."""
    name = base.lstrip("\\")
    # the backslash is already stripped, so match bare macro NAMES
    inner = re.sub(r"^(?:math(?:rm|bf|it|frak)|text(?:bf|it|rm)?)|[{}\s]", "", name) or name
    if name.lower() in _GREEK_MACROS:
        return _LATEX_TO_GREEK[name.lower()]
    if name == "sum":
        return "Σ"  # GLM's habitual misread of the Σ siglum as the sum operator
    if name.lower() == "alpha":
        return "𝔞"  # ... and of the fraktur 𝔞 witness letter (recurrent)
    if name == "eta":
        return "¶"  # GLM's habitual misread of the pilcrow; inert unless the classifier saw a ¶
    if name.startswith("mathfrak"):
        letters = inner.replace("mathfrak", "")
        return _FRAKTUR_LOWER.get(letters, "𝔉")  # single letter -> the letter itself
    if name == "dagger":
        return "†"
    if name == "S" and base.startswith("\\"):
        return "¶"  # \S = the same pilcrow misread; the guard keeps bare capital S a witness letter
    if name in ("P",) and not sups:
        return None  # bare \P pilcrow is ambiguous; leave alone
    if base.startswith(("\\math", "\\text")) and len(inner) == 1 and inner.isupper():
        base_ch = inner
    elif len(base) == 1 and (base.isupper() or base in _GREEK_TO_LATEX):
        base_ch = base
    else:
        return None
    if base_ch in _GREEK_TO_LATEX:
        return base_ch
    return base_ch if sups.strip() else None  # plain capital: siglum only WITH sups


def glm_sigla_spans(content: str) -> list[dict]:
    """Return [{start, end, key}] candidate sigla spans in `content`, non-overlapping, in
    document order."""
    spans: list[dict] = []

    def add(s: int, e: int, key: str) -> None:
        spans.append({"start": s, "end": e, "key": key, "in_math": False})

    # math segments
    for m in re.finditer(r"\$(.*?)\$", content, re.S):
        body, off = m.group(1), m.start(1)
        for u in _MATH_UNIT.finditer(body):
            key = _math_unit_key(u.group(1), u.group(2))
            if key:
                spans.append({"start": off + u.start(), "end": off + u.end(),
                              "key": key, "in_math": True})
    taken = [(s["start"], s["end"]) for s in spans]

    def free(s: int, e: int) -> bool:
        return all(e <= a or s >= b for a, b in taken)

    # base + <sup>...</sup>
    for m in _SUP_TAG.finditer(content):
        if free(m.start(), m.end()):
            add(m.start(), m.end(), m.group(1))
            taken.append((m.start(), m.end()))
    # base + bare-caret sup chain in plain text (Σ^T)
    for m in _BARE_CARET.finditer(content):
        if free(m.start(), m.end()):
            add(m.start(), m.end(), m.group(1))
            taken.append((m.start(), m.end()))
    # unicode greek + modifiers (or Latin capital WITH modifiers)
    for m in _UNI_SIGLUM.finditer(content):
        if free(m.start(), m.end()):
            add(m.start(), m.end(), m.group(1) or m.group(3))
            taken.append((m.start(), m.end()))
    # unicode fraktur runs
    for m in _UNI_FRAK.finditer(content):
        if free(m.start(), m.end()):
            add(m.start(), m.end(), m.group(0) if len(m.group(0)) == 1 else "𝔉")
            taken.append((m.start(), m.end()))
    # plain-text α is GLM's habitual misread of fraktur 𝔞; inert unless the classifier saw an 𝔞
    for m in re.finditer("α", content):
        if free(m.start(), m.end()):
            add(m.start(), m.end(), "𝔞")
            taken.append((m.start(), m.end()))
    # standalone declaratio symbols in plain text
    for m in _STANDALONE.finditer(content):
        if free(m.start(), m.end()):
            add(m.start(), m.end(), m.group(0))  # always a single char
            taken.append((m.start(), m.end()))

    # in_math is decided by POSITION, because a standalone '*' picked up inside a
    # $...$ segment must not get re-wrapped in $ at replacement time
    math_ranges = [(m.start(), m.end()) for m in re.finditer(r"\$.*?\$", content, re.S)]
    for s in spans:
        s["in_math"] = any(a <= s["start"] < b for a, b in math_ranges)
    return sorted(spans, key=lambda s: s["start"])


# ---------------------------------------------------------------------------
# LaTeX/HTML leftovers -> Unicode (for the human-readable variant)
# ---------------------------------------------------------------------------

def fold_to_unicode(text: str) -> str:
    def sup_content(s: str) -> str:
        out = ""
        for ch in s:
            if ch.upper() in _SUPERSCRIPT_LETTER_TO_CHAR:
                out += _SUPERSCRIPT_LETTER_TO_CHAR[ch.upper()]
            elif ch in "123":
                out += {"1": "¹", "2": "²", "3": "³"}[ch]
            else:
                out += ch
        return out

    t = text
    t = re.sub(r"<su[pb]>(.*?)</su[pb]>", lambda m: sup_content(m.group(1)), t, flags=re.S)
    for name, ch in _LATEX_TO_GREEK.items():
        t = re.sub(r"\\%s\b" % name.capitalize(), ch, t)
        t = re.sub(r"\\%s\b" % name, ch, t)
    prev = None
    while prev != t:  # an innermost-first fixpoint that peels wrappers nested to
        prev = t      # any depth and tolerates "\mathrm {...}" with a space
        t = re.sub(r"\\mathfrak\s*\{([^{}]*)\}",
                   lambda m: "".join(_FRAKTUR_LOWER.get(c, c) for c in m.group(1)), t)
        t = re.sub(r"\\(?:math(?:rm|bf|it)|text(?:bf|it|rm)?)\s*\{([^{}]*)\}", r"\1", t)
    t = t.replace(r"\dagger", "†").replace(r"\mid", "|").replace(r"\sim", "~")
    t = re.sub(r"\\P(?![A-Za-z])", "¶", t)  # \P pilcrow; lookahead spares \Phi/\Pi/\Psi
    t = t.replace(r"\circ", "ᴼ")  # GLM's habitual misread of superscript O
    t = re.sub(r"\\sum\b", "Σ", t)  # ... and of the Σ siglum as the sum operator
    t = re.sub(r"\\alpha\b", "α", t)  # fraktur 𝔞 misread; keyed span handles matched ones
    t = re.sub(r"\^\{([^{}]*)\}", lambda m: sup_content(m.group(1)), t)
    t = re.sub(r"\^([A-Za-z0-9*])", lambda m: sup_content(m.group(1)), t)
    t = re.sub(r"\^(?=[¹²³ᴬ-ᵸⱽꟳ*†])", "", t)   # caret residue before an already-folded sup
    t = re.sub(r"\$\s*|\s*\$", " ", t)          # drop math delimiters
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t


# ---------------------------------------------------------------------------
# per-region write-back
# ---------------------------------------------------------------------------

def _geometry_veto(cls_tokens: list[dict], bg_tokens: list[dict]) -> set[int]:
    """Position gate. A manuscript_siglum whose box top sits >= 8 px below the line's cap
    reference is lowercase text misread as a siglum, since real sigla align with the cap band.
    Calibrated on a hand-annotated sample where 88 of 96 down-shifted candidates were false positives.
    The reference is the median top of the other sigla on the line, falling back to same-line
    tall tokens; no reference means no veto. There is deliberately no confidence term,
    because half the annotated false positives sit at confidence >= 0.9,
    and vetoed-but-real cases land in review.
    Returns id()s of vetoed tokens."""
    by_line: dict[int | None, list[dict]] = {}
    for t in list(cls_tokens) + list(bg_tokens):
        by_line.setdefault(t.get("line"), []).append(t)
    sig_hs = sorted(t["bbox_in_region"][3] - t["bbox_in_region"][1]
                    for t in cls_tokens
                    if t["prediction"]["family"] == "manuscript_siglum")
    # 33 px = measured median height of a capital siglum box (cap height plus
    # superscript band; deliberately neither taxonomy value)
    tall = 0.75 * (sig_hs[len(sig_hs) // 2] if sig_hs else 33)
    veto: set[int] = set()
    for t in cls_tokens:
        if t["prediction"]["family"] != "manuscript_siglum":
            continue
        mates = [m for m in by_line.get(t.get("line"), []) if m is not t]
        ref = [m for m in mates
               if m["prediction"]["family"] == "manuscript_siglum"]
        if not ref:
            ref = [m for m in mates
                   if m["bbox_in_region"][3] - m["bbox_in_region"][1] >= tall]
        if not ref:
            continue
        tops = sorted(m["bbox_in_region"][1] for m in ref)
        if t["bbox_in_region"][1] - tops[len(tops) // 2] >= 8:
            veto.add(id(t))
    return veto


def _implausible_siglum(tok: dict) -> bool:
    """Grammar gate. A modifier stack is a set of witness designations, so the same letter
    never repeats. A decode with a repeat (Φᶻᶻ) is a false positive on plain text and must
    not enter alignment. Digits, '*' and '†' may repeat; no length cap (stacks are unbounded)."""
    if tok["prediction"]["family"] != "manuscript_siglum":
        return False
    letters = [unicodedata.normalize("NFKC", c).upper()
               for c in tok["reconstructed_string"][1:]]
    letters = [c for c in letters if c.isalpha()]
    return len(letters) != len(set(letters))


# Only siglum-shaped cross-key pairs qualify, where the span key is a capital/Greek base
# AND the token a manuscript_siglum. Unrestricted, this rule paired any lone leftovers and corrupted
# six development regions; declaratio symbols and fraktur match by exact key or not at all.
# A cross-key pairing overwrites GLM's base, so both confidence floors are required.
def _sc_compatible(span_key: str, tok: dict) -> bool:
    if tok["prediction"].get("family_confidence", 1.0) < 0.9:
        return False
    # GLM recurrently misreads the pilcrow as a bare Ψ
    if tok["prediction"]["family"] == "declaratio_signorum" \
            and tok["reconstructed_string"] == "¶" and span_key == "Ψ":
        return True
    # base confidence is required too, because the family head can be certain while the base head guesses
    return (tok["prediction"]["family"] == "manuscript_siglum"
            and tok["prediction"].get("base_confidence", 1.0) >= 0.9
            and (span_key in _GREEK_TO_LATEX
                 or (span_key.isascii() and span_key.isupper())))


# ---------------------------------------------------------------------------
# word-level geometry: virtual bounding boxes for the OCR text
# ---------------------------------------------------------------------------
# The OCR emits no character boxes and the segmenter boxes connected components,
# so the unit the two sides share is the WORD, reconstructed on the ink side first,
# where consecutive same-line components merge into one ink word while their gap stays small.
# The gap distribution is sharply bimodal (intra-word 0-8 px vs 17-26 px word spaces
# at 330 dpi), so the cut at 0.35 of median component height (~10 px) is safe.
# Ink words are then aligned to OCR words by estimated character count (width DP),
# and each matched OCR word inherits its ink word's coordinates.
#
# The ink side uses BACKGROUND tokens only, because this layer exists to LOCATE a span,
# and adding the span's own ink merges it into a neighbouring ink word and coarsens
# the very bracket being interpolated (measured slightly worse overall).
# char_w is therefore background ink over ALL transcribed characters,
# an underestimate by construction, and the DP costs are calibrated against it as such.
# Do not "fix" either without re-running the comparison.

def _ink_words(toks: list[dict]) -> list[tuple[int, int, int]]:
    """Merge connected components into ink words. Returns [(line, x0, x1)] in reading order."""
    ts = sorted(toks, key=lambda t: (t["line"], t["bbox_in_region"][0]))
    if not ts:
        return []
    hs = sorted(t["bbox_in_region"][3] - t["bbox_in_region"][1] for t in ts)
    sep = max(4.0, 0.35 * hs[len(hs) // 2])
    out: list[list[int]] = []
    for t in ts:
        b = t["bbox_in_region"]
        if out and out[-1][0] == t["line"] and b[0] - out[-1][2] <= sep:
            out[-1][2] = max(out[-1][2], b[2])
        else:
            out.append([t["line"], b[0], b[2]])
    return [tuple(w) for w in out]


def _word_boxes(content: str, toks: list[dict]) -> tuple[list[tuple], float]:
    """Return ([(word_start, word_end, box|None)], char_w), where box = (line, x0, x1)."""
    words = [(m.start(), m.end(), max(1, len(fold_to_unicode(m.group(0)).strip())))
             for m in re.finditer(r"\S+", content)]
    ink = _ink_words(toks)
    flat = [(s, e, None) for s, e, _ in words]
    if not words or not ink:
        return flat, 0.0
    # mean character width calibrated on the region itself; a height-based estimate
    # is thrown off by the segmenter's one-pixel punctuation components
    char_w = sum(x1 - x0 for _, x0, x1 in ink) / sum(w[2] for w in words)
    if char_w <= 0:
        return flat, 0.0
    est = [max(1, round((x1 - x0) / char_w)) for _, x0, x1 in ink]
    n, m = len(est), len(words)
    INF = 1e9

    def sub(a: int, b: int) -> float:
        return min(6, abs(est[a] - words[b][2]))

    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for a in range(n + 1):
        for b in range(m + 1):
            c = dp[a][b]
            if c >= INF:
                continue
            if a < n and b < m:
                dp[a + 1][b + 1] = min(dp[a + 1][b + 1], c + sub(a, b))
            if a < n:
                dp[a + 1][b] = min(dp[a + 1][b], c + 2 + 0.3 * est[a])
            if b < m:
                dp[a][b + 1] = min(dp[a][b + 1], c + 2 + 0.3 * words[b][2])
    out: list = [None] * m
    a, b = n, m
    while a > 0 or b > 0:
        c = dp[a][b]
        if a and b and abs(dp[a - 1][b - 1] + sub(a - 1, b - 1) - c) < 1e-9:
            out[b - 1] = ink[a - 1]
            a, b = a - 1, b - 1
        elif a and abs(dp[a - 1][b] + 2 + 0.3 * est[a - 1] - c) < 1e-9:
            a -= 1
        else:
            b -= 1
    return [(words[i][0], words[i][1], out[i]) for i in range(m)], char_w


def _span_bracket(content: str, wboxes: list[tuple], pos: int):
    """Where the OCR text says character `pos` sits, as a horizontal interval (lo, hi);
    each side is a (line, x, x') pair whose inner edge bounds it, or None (no constraint).

    An anchored word locates the interval by linear interpolation on the character offset,
    widened by a two-character tolerance, since word granularity alone cannot separate the parts
    of a fused run like `\\Omega agrelsc`. Otherwise the interval is the open space between
    the nearest anchored words on either side."""
    left = right = None
    for s, e, box in wboxes:
        if box is None:
            continue
        if s <= pos < e:
            ln, x0, x1 = box
            n = max(1, len(fold_to_unicode(content[s:e]).strip()))
            k = len(fold_to_unicode(content[s:pos]).strip())
            step = (x1 - x0) / n
            lo = x0 + step * max(0, k - 2)
            hi = x0 + step * min(n, k + 2)
            return (ln, lo, lo), (ln, hi, hi)
        if e <= pos:
            left = box
        elif s > pos and right is None:
            right = box
    return left, right


def _geom_agrees(bracket, tok: dict) -> bool:
    """True when both anchors sit on the token's line and the token lies between them,
    which means the span and the token are the same patch of ink. Both anchors are required,
    because this predicate licenses overruling a grammar veto."""
    left, right = bracket
    return (left is not None and right is not None
            and left[0] == right[0] == tok["line"]
            and left[2] <= tok["bbox_in_region"][2]
            and tok["bbox_in_region"][0] <= right[1])


def _key_cost(span_key: str, tok: dict) -> float | None:
    """Pairing price by shape alone; None = the two are not pairable at all."""
    if span_key == token_key(tok):
        return 0.0
    return 0.45 if _sc_compatible(span_key, tok) else None


def _geom_rescue(pairs: list[tuple[int, int]], spans: list[dict], toks: list[dict],
                 sboxes: list) -> list[tuple[int, int]]:
    """Pair up leftovers that occupy the same patch of ink.

    When OCR drops or invents material, the two reading orders slip apart and a span
    and a token that ARE the same printed siglum both end up discarded. If the OCR's
    own words place the span on the token's line and within its horizontal extent,
    the two are the same ink: add the pair. Existing pairings are untouched and shape
    compatibility (_key_cost) still holds. Rescued pairs are exempt from monotonicity
    by necessity, since they exist only because the reading orders slipped apart."""
    if not sboxes:
        return pairs
    free_i = [i for i in range(len(spans)) if i not in {a for a, _ in pairs}]
    free_j = [j for j in range(len(toks)) if j not in {b for _, b in pairs}]
    out = list(pairs)
    for i in free_i:
        cands = sorted((c, j) for c, j in
                       ((_key_cost(spans[i]["key"], toks[j]), j) for j in free_j
                        if _geom_agrees(sboxes[i], toks[j])) if c is not None)
        if not cands:
            continue
        _, j = cands[0]
        out.append((i, j))
        free_j.remove(j)
    return sorted(out)


def _global_align(spans: list[dict], toks: list[dict]) -> list[tuple[int, int]]:
    """Cost-based global alignment (Needleman-Wunsch) over the two reading-order sequences.
    Every monotone pairing is priced and the cheapest one wins:

      exact key match          0.0
      cross-key siglum pair    0.45  (under the _sc_compatible conditions)
      span left unmatched      0.35  (OCR text kept, the safe default)
      token left unmatched     0.25 + 0.5 * family_confidence
                                     (dropping a confident detection hurts more)

    Geometry deliberately does NOT enter these costs, because the cost model is calibrated
    as a whole and a positional penalty re-ranks entire blocks (measured; three regions
    were rewritten, all for the worse). Geometry runs afterwards as a rescue pass (_geom_rescue),
    where it can only add pairings the sequence view left unmatched."""
    n, m = len(spans), len(toks)
    GAP_SPAN = 0.35
    INF = float("inf")

    def gap_tok(tok: dict) -> float:
        return 0.25 + 0.5 * tok["prediction"].get("family_confidence", 1.0)

    def pair_cost(i: int, j: int) -> float | None:
        return _key_cost(spans[i]["key"], toks[j])

    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c = dp[i][j]
            if c == INF:
                continue
            if i < n and j < m:  # checked first so that on equal cost a match wins
                pc = pair_cost(i, j)
                if pc is not None and c + pc < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = c + pc
                    back[i + 1][j + 1] = "pair"
            if i < n and c + GAP_SPAN < dp[i + 1][j]:
                dp[i + 1][j] = c + GAP_SPAN
                back[i + 1][j] = "skip_span"
            if j < m and c + gap_tok(toks[j]) < dp[i][j + 1]:
                dp[i][j + 1] = c + gap_tok(toks[j])
                back[i][j + 1] = "skip_tok"
    pairs = []
    i, j = n, m
    while i or j:
        mv = back[i][j]
        if mv == "pair":
            i, j = i - 1, j - 1
            pairs.append((i, j))
        elif mv == "skip_span":
            i -= 1
        else:
            j -= 1
    return sorted(pairs)


def _polish_tex(t: str) -> str:
    """Final-form post-processing. Superscript letters become uppercase (the printed
    grammar has no lowercase modifiers) and a digit superscript nests on its preceding
    letter (^{F}^{2} -> ^{F^2}, the ground-truth convention). Serialization only;
    \\macro names are protected, so OCR garbage macros pass through unchanged."""
    out: list[str] = []
    i, n = 0, len(t)
    while i < n:
        ch = t[i]
        if ch == "^":
            if i + 1 < n and t[i + 1] == "{":
                j, depth = i + 2, 1
                while j < n and depth:
                    depth += {"{": 1, "}": -1}.get(t[j], 0)
                    j += 1
                if depth:              # unbalanced brace, leave the tail alone
                    out.append(t[i:])
                    break
                inner = re.sub(r"\\[A-Za-z]+|[a-z]",
                               lambda m: m.group(0) if m.group(0).startswith("\\")
                               else m.group(0).upper(), t[i + 2:j - 1])
                # a flat digit nests on its letter, UNBRACED (^{BD2} -> ^{BD^2}),
                # because the braced form derails the evaluation parser
                inner = re.sub(r"([A-Z])([123])", r"\1^\2", inner)
                out.append("^{" + inner + "}")
                i = j
                continue
            if i + 1 < n and t[i + 1].isalpha():
                out.append("^" + t[i + 1].upper())
                i += 2
                continue
        out.append(ch)
        i += 1
    t = "".join(out)
    t = re.sub(r"<sup>([^<]*)</sup>",
               lambda m: "<sup>" + re.sub(r"[a-z]", lambda w: w.group(0).upper(),
                                          m.group(1)) + "</sup>", t)
    # the same nesting for adjacent groups, ^{...X}^{2} -> ^{...X^2}
    return re.sub(r"\^\{([^{}]*[A-Z])\}\^\{([123])\}", r"^{\1^\2}", t)


def _polish_uni(t: str) -> str:
    """Unicode-view counterpart of _polish_tex. A lowercase modifier letter becomes the
    canonical modifier character of its uppercase letter (ᵃ -> ᴬ; ᶻ IS canonical and stays).
    Only BMP characters with a single-letter compatibility decomposition qualify,
    so fraktur, ordinary text and html markup all pass through."""
    out: list[str] = []
    for ch in t:
        if ord(ch) <= 0xFFFF:
            d = unicodedata.normalize("NFKC", ch)
            if d != ch and len(d) == 1 and d.isalpha() and d.islower():
                up = _SUPERSCRIPT_LETTER_TO_CHAR.get(d.upper())
                if up:
                    out.append(up)
                    continue
        out.append(ch)
    return "".join(out)


# every macro name observed in the GLM output for this corpus, plus the LaTeX
# accents the ground truth itself uses; anything else with >=2 letters is
# OCR garbage (\agrelsc) and gets its backslash stripped
_KNOWN_MACROS = frozenset("""
mathrm mathbf mathit mathfrak mathcal mathsf mathring
text textbf textit textrm textsuperscript
Theta Phi Psi Omega Lambda Sigma Pi Xi
theta phi psi omega lambda sigma pi alpha eta nu tau
dagger sim circ sum mid bullet overline bar
""".split())


# The fraktur letterform once labelled 'b' or 'd' is fraktur v (𝔳), crop-verified.
# The released checkpoint already ships the class as 'v', so the d -> v half of this
# table is dead code kept deliberately, because result JSONs from older checkpoints
# still need it, and the b -> v half stays live because GLM often misreads the
# letterform as 'b'. Guarded to the fraktur-run alphabet so a frakturized plain word
# is not garbled. The known ceiling is that any lowercase word built only of run
# letters would be folded; no such span occurs in pipeline output.
_FRAK_RUN_LETTERS = set("agrelbdscvw ")
_VD_CANON = str.maketrans({"ç": "c", "ῥ": "r", "b": "v", "d": "v"})


def _vd_fix_tex(text: str) -> str:
    def sub(m):
        letters = m.group(1)
        if set(letters) <= _FRAK_RUN_LETTERS:
            return "\\mathfrak{%s}" % letters.replace("d", "v").replace("b", "v")
        return m.group(0)
    return re.sub(r"\\mathfrak\{([a-z ]+)\}", sub, text)


def _vd_fix_uni(text: str) -> str:
    return text.replace("𝔡", "𝔳").replace("𝔟", "𝔳")


def _serialization_cleanup(content: str) -> tuple[str, list[dict]]:
    """Strip the LaTeX/HTML/Markdown fragments GLM leaks, before span extraction,
    so they neither survive into the corrected output nor produce bogus span candidates.
    Paired <sup>x</sup> stays untouched because it carries span structure."""
    notes: list[dict] = []

    def note(what: str, frag: str) -> None:
        notes.append({"why": "serialization cleanup: " + what, "text": frag})

    # markdown emphasis around a lowercase phrase (*om.*); erasure-star runs never
    # match because the inner text must start with a lowercase letter
    def _emph(m: re.Match) -> str:
        note("markdown emphasis stripped", m.group(0))
        return m.group(1)
    content = re.sub(r"\*([a-zà-ÿ][a-z0-9à-ÿ .,;:()\-]{0,38}?)\*", _emph, content)
    # bold/italic tags carry no meaning in this corpus, paired or not
    content = re.sub(r"</?[bi]>", lambda m: (note("html tag removed", m.group(0)), "")[1],
                     content)
    # unpaired <sup>/<sub>/</sup>/</sub>
    stack, drop = [], []
    for t in re.finditer(r"</?su[pb]>", content):
        if not t.group(0).startswith("</"):
            stack.append(t)
        elif stack:
            stack.pop()
        else:
            drop.append(t)
    for t in sorted(drop + stack, key=lambda t: -t.start()):
        note("orphan html tag removed", t.group(0))
        content = content[:t.start()] + content[t.end():]
    # a stray '<' that opens no tag (Ψ<BDF²)
    content = re.sub(r"<(?!/?(?:su[pb]|br|b|i)\b)",
                     lambda m: (note("stray '<' removed", "<"), "")[1], content)
    # garbage macros (\agrelsc): stripping the backslash restores the plain text
    # that fraktur gap interpolation needs as an anchor
    def _unmacro(m: re.Match) -> str:
        if m.group(1) in _KNOWN_MACROS:
            return m.group(0)
        note("garbage macro unwrapped", m.group(0))
        return m.group(1)
    content = re.sub(r"\\([A-Za-z]{2,})", _unmacro, content)
    # correction digits are 1-3 only, so a stray ⁰/⁴-⁹ is OCR noise glued to a siglum
    content = re.sub(r"[⁰⁴-⁹]",
                     lambda m: (note("orphan superscript digit removed", m.group(0)), "")[1],
                     content)
    # digits only ever appear singly and attached to a letter, so a digit-only
    # multi-digit group (^{\mathrm{11}}) is garble
    content = re.sub(r"\^\{(?:\\mathrm\{)?\d{2,}\}?\}",
                     lambda m: (note("digit-only superscript group removed", m.group(0)), "")[1],
                     content)
    # ... and so is its split form, >=2 consecutive digit-only groups (^{1}^{1})
    content = re.sub(r"(?:\^\{(?:\\mathrm\{)?\d\}?\}){2,}",
                     lambda m: (note("digit-run superscript groups removed", m.group(0)), "")[1],
                     content)
    # GLM writes the same modifier twice across a math boundary (\Omega^{\mathrm{M}}$$^{M});
    # same-letter repetition is grammatically illegal, so an identical adjacent copy
    # is always the double-write
    def _dupsup(m):
        canon = lambda s: re.sub(r"mathrm|[\\{}\s]", "", s).upper()
        if canon(m.group(2)) == canon(m.group(4)):
            note("cross-math duplicated superscript dropped", m.group(0))
            return m.group(1) + m.group(3)
        return m.group(0)
    content = re.sub(r"(\^\{(?:\\mathrm\{)?([A-Za-z0-9^{}*†]{1,8}?)\}?\})"
                     r"(\s*\${0,2}\s*)"
                     r"\^\{(?:\\mathrm\{)?([A-Za-z0-9^{}*†]{1,8}?)\}?\}",
                     _dupsup, content)
    # Do NOT collapse spaced letter runs (\mathrm{a g r e l s c}) here, because
    # per-letter interpolation already pairs them, and collapsing measurably
    # regressed witness accuracy.
    return content, notes


def _embedded_w_veto(cls_tokens: list[dict], bg_tokens: list[dict]) -> set[int]:
    """Embedded-𝔴 veto. A single-letter 𝔴 glued (<= 8 px) to a non-fraktur same-line
    neighbour is a prose ligature-pair misread (mi/ct/iu). A real standalone 𝔴 has
    word-sized gaps, and a 𝔴 inside a letter run has fraktur neighbours, so a glued
    fraktur neighbour marks it a cluster member and overrides a glued non-fraktur
    neighbour on the other side (the Ω^{M} wsc case). This applies to 𝔴 only,
    since 𝔞 legitimately hugs the preceding siglum."""
    toks = sorted(list(cls_tokens) + list(bg_tokens),
                  key=lambda t: (t["line"], t["bbox_in_region"][0]))
    fam = lambda t: t["prediction"]["family"]
    out: set[int] = set()
    for i, t in enumerate(toks):
        if fam(t) != "fraktur_cluster" or t.get("reconstructed_string") != "𝔴":
            continue
        near = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(toks) and toks[j]["line"] == t["line"]:
                g = (t["bbox_in_region"][0] - toks[j]["bbox_in_region"][2]) if j < i \
                    else (toks[j]["bbox_in_region"][0] - t["bbox_in_region"][2])
                if g <= 8:
                    near.append(toks[j])
        if any(fam(n) == "fraktur_cluster" for n in near):
            continue
        if any(fam(n) != "fraktur_cluster" for n in near):
            out.add(id(t))
    return out


def writeback_region(content: str, cls_tokens: list[dict],
                     bg_tokens: list[dict] = (), align: str = "difflib",
                     geometry: bool = True) -> dict:
    """Reconcile one region's VLM transcription with its classifier tokens
    (the six steps of the module docstring).

    Args:
        content: the region's VLM transcription (after split_sup_groups).
        cls_tokens: non-background classifier tokens, in line-major/x order.
        bg_tokens: background tokens of the same region (ink geometry source).
        align: "global" (cost-based alignment, the production route) or
            "difflib" (exact-key blocks, the frozen regression baseline).
        geometry: whether the global route may use word-level bbox geometry.

    Returns:
        A dict with the corrected renderings, HTML diff views, per-region counts
        and the review list.
    """
    # ---- STEP 1: normalization (span extraction follows the gates) ----
    content, cleanup_notes = _serialization_cleanup(content)
    # ---- STEP 2: pre-alignment token gates + raised-† pre-merge ----
    gated = [t for t in cls_tokens if _implausible_siglum(t)]
    cls_tokens = [t for t in cls_tokens if not _implausible_siglum(t)]
    geo = _geometry_veto(cls_tokens, bg_tokens)
    geo_gated = [t for t in cls_tokens if id(t) in geo]
    cls_tokens = [t for t in cls_tokens if id(t) not in geo]
    embw = _embedded_w_veto(cls_tokens, bg_tokens)
    embw_gated = [t for t in cls_tokens if id(t) in embw]
    cls_tokens = [t for t in cls_tokens if id(t) not in embw]
    # The raised-† pre-merge exists because the segmenter almost never attaches the
    # dagger to its base, so the siglum loses its † and the standalone † can never
    # pair. The geometry is cleanly bimodal (attached <= 6 px right of a siglum,
    # freestanding 40 px+), so the tight ones are merged.
    merged_notes: list[dict] = []
    order = sorted(range(len(cls_tokens)),
                   key=lambda k: (cls_tokens[k]["line"], cls_tokens[k]["bbox_in_region"][0]))
    drop: set[int] = set()
    for pi_, k in enumerate(order):
        t = cls_tokens[k]
        if t["prediction"]["family"] != "declaratio_signorum" \
                or t.get("reconstructed_string") != "†":
            continue
        for pj in range(pi_ - 1, -1, -1):
            L = cls_tokens[order[pj]]
            if L["line"] != t["line"]:
                break
            gap = t["bbox_in_region"][0] - L["bbox_in_region"][2]
            if L["prediction"]["family"] == "manuscript_siglum" and 0 <= gap <= 10 \
                    and not L["reconstructed_string"].endswith("†"):
                L["reconstructed_string"] += "†"
                L.setdefault("modifier_sequence", []).append(
                    {"glyph": "†",
                     "confidence": t["prediction"].get("family_confidence", 1.0),
                     "bbox": list(t["bbox_in_region"])})
                drop.add(k)
                merged_notes.append({"why": "standalone raised † merged into left siglum",
                                     "token": L["reconstructed_string"], "line": t["line"]})
            break
    if drop:
        cls_tokens = [t for k, t in enumerate(cls_tokens) if k not in drop]
    # ---- STEP 1 (cont.): candidate sigla span extraction from the GLM text ----
    spans = glm_sigla_spans(content)

    # ---- STEP 3: word-level coordinate estimation for the spans ----
    # (global route only; the difflib route stays byte-identical)
    sboxes = None
    if align == "global" and geometry:
        wboxes, _ = _word_boxes(content, bg_tokens)
        sboxes = [_span_bracket(content, wboxes, s["start"]) for s in spans]

    # ---- STEP 4: alignment (NW global with geometric rescue, or frozen difflib) ----
    if align == "global":
        pairs = _geom_rescue(_global_align(spans, cls_tokens), spans, cls_tokens, sboxes)
    else:
        cls_keys = [token_key(t) for t in cls_tokens]
        glm_keys = [s["key"] for s in spans]
        sm = difflib.SequenceMatcher(a=glm_keys, b=cls_keys, autojunk=False)
        pairs = [(b.a + k, b.b + k) for b in sm.get_matching_blocks() for k in range(b.size)]

        # Second-chance pairing, needed because a misread BASE (Λᴸ read as A^L)
        # breaks the exact-key match. Inside each gap a lone leftover span facing a
        # lone leftover token is paired, 1:1 within a gap only; the zero-size
        # sentinel block covers the trailing gap.
        prev_a = prev_b = 0
        for b in sm.get_matching_blocks():
            if b.a - prev_a == 1 and b.b - prev_b == 1 \
                    and _sc_compatible(spans[prev_a]["key"], cls_tokens[prev_b]):
                pairs.append((prev_a, prev_b))
            prev_a, prev_b = b.a + b.size, b.b + b.size
        pairs.sort()

    # degenerate-region channel (global route only). In a region whose GLM text is
    # syntactically wrecked, alignment leaves a large block of confident siglum
    # tokens unmatched even though same-base spans sit unmatched too.
    # The trigger floor of 12 works because the one wrecked region in the corpus
    # has 22 such tokens and the runner-up 8, so only the wrecked one ever fires.
    degen: set[tuple[int, int]] = set()
    if align == "global":
        m_i = {i for i, _ in pairs}
        m_j = {j for _, j in pairs}
        un_j = [j for j, t in enumerate(cls_tokens)
                if j not in m_j
                and t["prediction"]["family"] == "manuscript_siglum"
                and t.get("reconstructed_string")
                and t["prediction"].get("family_confidence", 1.0) >= 0.99
                and t["prediction"].get("base_confidence", 1.0) >= 0.99]
        if len(un_j) >= 12:
            # throw the cost-based pairing away and re-align on exact base keys alone,
            # since the cost model is calibrated for sane regions and is pure noise here
            sm2 = difflib.SequenceMatcher(a=[s["key"] for s in spans],
                                          b=[token_key(t) for t in cls_tokens],
                                          autojunk=False)
            pairs = sorted((blk.a + k, blk.b + k)
                           for blk in sm2.get_matching_blocks()
                           for k in range(blk.size))
            # No blanket superset-guard bypass is needed, because the garbled long
            # spans carry ILLEGAL modifiers and the illegal-modifier branch already
            # replaces them. `degen` stays as the review marker set.
            degen = set(pairs)

    # ---- STEP 5: post-pairing adjudication gates ----
    # The modifier-contradiction filter exists because witness lists repeat the same
    # bases, so one misread base can hand the aligner a longer shifted block than
    # the true alignment. The bases agree on such a false pair but the MODIFIERS clash.
    # Drop the pair when (a) both sides have modifiers and the sets are fully
    # disjoint, or (b) both have a */digit signature and they differ (Λᴸ* vs Λᴸ²);
    # both sides then stay unmatched and GLM text is kept.
    def _mod_set(chars) -> set[str]:
        out = set()
        for ch in chars:
            for lit in unicodedata.normalize("NFKC", ch):
                # GLM confuses the small round modifiers (0/O/c/ᴼ) among themselves,
                # so they are folded together and the veto never fires inside that cluster
                lit = {"0": "O", "C": "O", "c": "O"}.get(lit, lit)
                if lit.isalnum() or lit in "*†":
                    out.add(lit.upper())
        return out

    # The geometric exemption exists because the veto cannot distinguish "wrong
    # occurrence" from "right occurrence, OCR misread the modifier" and resolves
    # both by keeping OCR. When the span's virtual bbox and the token's real bbox
    # occupy the same patch of ink, misalignment is ruled out by construction, so
    # the pair is kept and the classifier corrects the modifiers, provided it is
    # sure of itself. Every override is recorded for review.
    geom_overrides: list[dict] = []

    def _mods_conflict(i: int, j: int) -> bool:
        if cls_tokens[j]["prediction"]["family"] != "manuscript_siglum":
            return False
        raw = re.sub(r"[\s^{}$_]", "",
                     fold_to_unicode(content[spans[i]["start"]:spans[i]["end"]]))
        g = _mod_set(raw[1:])
        c = _mod_set(cls_tokens[j]["reconstructed_string"][1:])
        if not g or not c:
            return False
        if g & c:
            gs = {x for x in g if not x.isalpha()}
            cs = {x for x in c if not x.isalpha()}
            if not (gs and cs and gs != cs):
                return False
        tok = cls_tokens[j]
        p = tok["prediction"]
        # the modifiers are what the disagreement is about, so the modifier decode
        # itself has to be certain; family and base confidence say nothing about it
        mods = [s["confidence"] for s in tok.get("modifier_sequence", [])
                if s["glyph"] != "reject"]
        # Known-misread exemptions, all one-way (GLM side only). GLM habitually
        # writes superscript Z as the digit 2, and superscript T/M as lowercase ᵇ.
        # When mapping the GLM character dissolves the conflict AND the classifier
        # decoded its own character at >= 0.99, the conflict is the known misread,
        # not a misalignment. The CNN confuses 2/z too, so no reverse exemption.
        for g_ch, c_ch in (("2", "Z"), ("B", "T"), ("B", "M")):
            if g_ch not in g or c_ch not in c:
                continue
            cconf = [s["confidence"] for s in tok.get("modifier_sequence", [])
                     if s["glyph"].upper() == c_ch]
            g2 = (g - {g_ch}) | {c_ch}
            gs2 = {x for x in g2 if not x.isalpha()}
            cs2 = {x for x in c if not x.isalpha()}
            if cconf and min(cconf) >= 0.99 and g2 & c \
                    and not (gs2 and cs2 and gs2 != cs2):
                geom_overrides.append(
                    {"why": "modifier conflict is the known %s<->%s OCR misread; "
                            "classifier kept" % (g_ch, c_ch),
                     "span": content[spans[i]["start"]:spans[i]["end"]].strip(),
                     "token": tok["reconstructed_string"], "line": tok["line"]})
                return False
        # The illegal-modifier exemption applies when GLM's modifiers are
        # grammatically illegal for this base, since then the span cannot be correct
        # and vetoing the pair only preserves a certain error; a legal, confident
        # decode wins
        if _illegal_mods(spans[i]["key"], raw[1:]) \
                and not _illegal_mods(spans[i]["key"],
                                      tok["reconstructed_string"][1:]) \
                and p.get("family_confidence", 1.0) >= 0.9 \
                and p.get("base_confidence", 1.0) >= 0.9 \
                and mods and min(mods) >= 0.9:
            geom_overrides.append(
                {"why": "OCR modifiers illegal for this base; classifier kept",
                 "span": content[spans[i]["start"]:spans[i]["end"]].strip(),
                 "token": tok["reconstructed_string"], "line": tok["line"]})
            return False
        if sboxes and _geom_agrees(sboxes[i], tok) \
                and p.get("family_confidence", 1.0) >= 0.9 \
                and p.get("base_confidence", 1.0) >= 0.9 \
                and mods and min(mods) >= 0.9:
            geom_overrides.append(
                {"why": "modifier conflict overruled by bbox overlap; classifier kept",
                 "span": content[spans[i]["start"]:spans[i]["end"]].strip(),
                 "token": cls_tokens[j]["reconstructed_string"],
                 "line": cls_tokens[j]["line"]})
            return False
        return True

    pairs = [(i, j) for i, j in pairs if not _mods_conflict(i, j)]

    # The fraktur content check exists because the fraktur key is a bare "𝔉" with no
    # letter content, so a false-positive token could overwrite a correct \mathfrak span.
    # Keep the pair only if GLM's letters and the decode overlap
    # (subsequence either way, or >= 0.5 similarity).
    def _frak_ok(i: int, j: int) -> bool:
        if spans[i]["key"] != "𝔉":
            return True
        folded = fold_to_unicode(content[spans[i]["start"]:spans[i]["end"]])
        g = "".join(_FRAK_CHAR_TO_LETTER.get(c, c) for c in folded if not c.isspace())
        c = "".join(_FRAK_CHAR_TO_LETTER.get(ch, ch)
                    for ch in cls_tokens[j]["reconstructed_string"])
        it1, it2 = iter(g), iter(c)
        return (all(ch in it1 for ch in c) or all(ch in it2 for ch in g)
                or difflib.SequenceMatcher(None, g, c).ratio() >= 0.5)

    pairs = [(i, j) for i, j in pairs if _frak_ok(i, j)]

    # Tail extension, because GLM sometimes spills part of a superscript stack as
    # following plain capitals (\phi^{RA*}ZGVP). If the text right after a matched span equals
    # the TAIL of the decoded modifier sequence, absorb it into the span.
    for i, j in pairs:
        s = spans[i]
        tok = cls_tokens[j]
        fam = tok["prediction"]["family"]
        recon_lit = "".join(unicodedata.normalize("NFKC", c) for c in tok["reconstructed_string"])
        if fam == "manuscript_siglum":
            mods = recon_lit[1:].upper()
            # sees through a \mathrm{...} wrapper and a trailing sup chain.
            # SPACE discriminates the two cases, since a glued single char is GLM
            # flattening the modifier (Πc) while a spaced single capital is a
            # separate witness (Σᴼᴹ M).
            # Two passes, where uppercase-only stops before a following lowercase
            # run (a separate fraktur word), then mixed covers a lowercased modifier.
            for run, stop in ((r"[A-Z][A-Z0-9*^{}]{0,9}", r"(?![A-Z])"),
                              (r"[A-Za-z][A-Za-z0-9*^{}]{0,9}", r"(?![A-Za-z])")):
                m = re.match(
                    r"( ?)(?:\\(?:math(?:rm|bf|it)|text(?:bf|it|rm)?)\s*\{)?"
                    rf"({run})\}}?"
                    rf"((?:\s*\^\{{?[A-Za-z0-9*]{{1,3}}\}}?){{0,3}}){stop}",
                    content[s["end"]:])
                tail = re.sub(r"[ ^{}]", "", m.group(2) + m.group(3)).upper() if m else ""
                if tail and mods.endswith(tail) and (len(tail) >= 2 or m.group(1) == ""):
                    s["end"] += m.end()
                    break
            # a broken sup fragment (Ψ<sup>d}$.) is absorbed when its letters are
            # the tail of the decoded modifiers
            m = re.match(r"<su[pb]>([A-Za-z0-9*]{1,4})\}?\$?", content[s["end"]:])
            if m and mods.endswith(m.group(1).upper()):
                s["end"] += m.end()
            # a sup group split across the math boundary (Φ$^{z}$*, so no _MATH_UNIT
            # span forms) has its $^{...}$ residue absorbed, plus a glued trailing
            # */†, when the folded content is already inside the decoded modifiers
            m = re.match(r"\s{0,2}\$\s*\^\{?([A-Za-z0-9*†]{1,6})\}?\s*\$([*†]?)",
                         content[s["end"]:])
            if m and (m.group(1) + m.group(2)).upper() in mods:
                s["end"] += m.end()
            # For the same split where the math segment continues after the group
            # (Λ$^{H}.), absorb the whole ^-group CHAIN, because split_sup_groups
            # rewrote ^{D*} as ^{D}^{*} and eating only the first link would tear
            # the chain apart.
            # The ^ in the lookahead keeps backtracking from swallowing half a chain.
            m = re.match(r"\s{0,2}\$((?:\s*\^\{?[A-Za-z0-9*†]{1,3}\}?)+)(?![A-Za-z0-9*†}^])",
                         content[s["end"]:])
            if m:
                body = re.sub(r"[\s^{}]", "", m.group(1)).upper()
                if body and body in mods:
                    s["end"] += m.end()
            # a sup fragment with UNBALANCED braces (the ^{ group never closes before
            # the math segment ends) is OCR garble, absorbed when every fragment
            # letter already sits in the decoded modifiers
            m = re.match(r"\^\{(?:[^{}$]|\{[^{}$]*\})*(?=\$|$)", content[s["end"]:])
            if m and m.group(0).count("{") > m.group(0).count("}"):
                body = re.sub(r"\\[A-Za-z]+|[\^{}\s\d]", "", m.group(0))
                if set(body.upper()) <= set(mods):
                    s["end"] += m.end()
            # orphan unicode-superscript garble glued to the span (Πᴰⁿ) is absorbed
            # when its folded letters are grammatically illegal for this base,
            # since that makes it OCR debris rather than a missed modifier
            m = re.match(r"[\u2070-\u209f]{1,2}", content[s["end"]:])
            if m:
                lits = [l for ch in m.group(0)
                        for l in unicodedata.normalize("NFKC", ch) if l.isalpha()]
                if lits and all(_illegal_mods(s["key"], l) for l in lits):
                    s["end"] += m.end()
            # The same applies to the LEFT, where pairing often matches only the trailing '*' of a
            # siglum GLM wrote as plain text (ΨB*); the preceding capitals are absorbed
            # when they are the decoded siglum's head. Lowercase greek is included
            # because GLM writes the base lowercase (ψB*) and the ψ would survive otherwise.
            m = re.search(r"[A-ZΘΦΨΩΛΣΠθφψωλσπ]{1,8}$", content[:s["start"]])
            if m and recon_lit.upper().startswith(m.group(0).upper()) \
                    and len(m.group(0)) < len(recon_lit):
                s["start"] -= len(m.group(0))
        elif fam == "fraktur_cluster":
            letters = "".join(_FRAK_CHAR_TO_LETTER.get(c, c) for c in tok["reconstructed_string"])
            m = re.match(r"[a-z]{1,9}", content[s["end"]:])
            if m and letters.endswith(m.group(0)):
                s["end"] += m.end()

    # a tail/left extension may have swallowed a neighboring span whole;
    # a covered span must not render a second time
    _rng = {i: (spans[i]["start"], spans[i]["end"]) for i, _ in pairs}
    _covered = {i for i in _rng for k in _rng if k != i and _rng[k] != _rng[i]
                and _rng[k][0] <= _rng[i][0] and _rng[i][1] <= _rng[k][1]}
    pairs = [(i, j) for i, j in pairs if i not in _covered]

    for i, j in pairs:
        if (i, j) in degen:
            geom_overrides.append(
                {"why": "degenerate region: confident siglum force-paired to "
                        "leftover same-base span (CNN trusted)",
                 "span": content[spans[i]["start"]:spans[i]["end"]].strip(),
                 "token": cls_tokens[j]["reconstructed_string"],
                 "line": cls_tokens[j]["line"]})

    # ---- STEP 6: recovery of material the VLM never marked ----
    # Gap interpolation, needed because GLM writes fraktur letters as plain
    # lowercase text, invisible to glm_sigla_spans, so those tokens never match. For each run of
    # unmatched fraktur tokens, search the window between the neighboring matched
    # spans for a lowercase word that contains the decoded letters as an ordered
    # subsequence covering at least half of it; convert its letters to fraktur
    # in place. Anything less certain stays in the review list.
    def _is_subseq(needle: str, hay: str) -> bool:
        it = iter(hay)
        return all(c in it for c in needle)

    # The lookbehind allows an UPPERCASE prefix because GLM glues the witness
    # capital onto the fraktur run ('Pgrelbsc'), and a plain [A-Za-z] lookbehind
    # would yield no candidate word at all.
    _WORD = re.compile(r"(?<![a-z\\])(?:[a-z](?: [a-z])+|[a-z]+)(?![A-Za-z])")
    # homoglyph view for candidate search and in-place mapping, strictly 1:1 so
    # offsets stay valid. Capital I folds to 'l' (GLM's take on a lone 𝔩),
    # ç is the c-with-cedilla misread, and ῥ the Greek take on 𝔯.
    # b/d -> v via _VD_CANON canonicalizes both sides, making matching
    # model-agnostic; _vd_fix_* stays as the safety net for decode-sourced renders.
    content_n = re.sub(r"(?<=[a-z])I|(?<![A-Za-z])I(?![A-Za-z])", "l", content) \
        .translate(_VD_CANON)
    math_ranges = [(m.start(), m.end()) for m in re.finditer(r"\$.*?\$", content, re.S)]
    matched_j = {j for _, j in pairs}
    groups: list[list[int]] = []
    for j, t in enumerate(cls_tokens):
        if j in matched_j or t["prediction"]["family"] != "fraktur_cluster":
            continue
        if groups and groups[-1][-1] == j - 1 \
                and t["line"] == cls_tokens[groups[-1][-1]]["line"] \
                and t["bbox_in_region"][0] - cls_tokens[groups[-1][-1]]["bbox_in_region"][2] <= 60:
            # same line only (a stray next-line token poisons the group's letters);
            # 60 px cap separates spaced fraktur runs (~4-25 px apart) from
            # unrelated same-line detections
            groups[-1].append(j)
        else:
            groups.append([j])

    # The decode-trust split exists because a token whose per-glyph decode collapsed
    # (every component under 0.1) must not share a group with solidly decoded
    # letters, or the group's letters absorb its garbage and the whole gap match
    # fails. The split happens at trust boundaries; an unsure-only group still
    # matches on its own.
    def _unsure(j: int) -> bool:
        seq = [s["confidence"] for s in cls_tokens[j].get("modifier_sequence", [])
               if s["glyph"] != "reject"]
        return bool(seq) and max(seq) < 0.1

    split: list[list[int]] = []
    for grp in groups:
        run = [grp[0]]
        for j in grp[1:]:
            if _unsure(j) != _unsure(run[-1]):
                split.append(run)
                run = [j]
            else:
                run.append(j)
        split.append(run)
    groups = split

    # Word-width anchors align the gap's BACKGROUND token widths against the gap's
    # whitespace-words by estimated character count (pure geometry, robust to GLM
    # misreads), narrowing the search window to between the words anchored around
    # the group. Only inside a narrowed window is the RELAXED matcher allowed,
    # where candidate words may be concatenated and matched fuzzily
    # and the replacement is then the classifier's own decode.
    _tkey = lambda t: (t["line"], t["bbox_in_region"][0])
    bg_sorted = sorted(bg_tokens, key=_tkey)
    heights = sorted(t["bbox_in_region"][3] - t["bbox_in_region"][1] for t in bg_sorted)
    char_w = 0.55 * heights[len(heights) // 2] if heights else 0

    def _refine_window(grp, lo: int, hi: int) -> tuple[int, int]:
        if not char_w:
            return lo, hi
        j_left = max([j for _, j in pairs if j < grp[0]], default=None)
        j_right = min([j for _, j in pairs if j > grp[-1]], default=None)
        k_left = _tkey(cls_tokens[j_left]) if j_left is not None else (-1, -1)
        k_right = _tkey(cls_tokens[j_right]) if j_right is not None else (1 << 30, 0)
        kL, kR = _tkey(cls_tokens[grp[0]]), _tkey(cls_tokens[grp[-1]])
        # anchor only on substantial words (>= ~3 chars of ink), because tiny bg
        # tokens are often letters inside the group's own run and poison the width alignment
        def _wide(b):
            return b["bbox_in_region"][2] - b["bbox_in_region"][0] >= 2.5 * char_w
        before = [b for b in bg_sorted if k_left < _tkey(b) < kL and _wide(b)]
        after = [b for b in bg_sorted if kR < _tkey(b) < k_right and _wide(b)]
        if not before and not after:
            return lo, hi
        # the group participates in the DP as one pseudo-token spanning its ink,
        # so an anchor can never claim the group's own word
        gx0 = cls_tokens[grp[0]]["bbox_in_region"][0]
        gx1 = cls_tokens[grp[-1]]["bbox_in_region"][2]
        est = ([max(1, round((b["bbox_in_region"][2] - b["bbox_in_region"][0]) / char_w))
                for b in before]
               + [max(1, round((gx1 - gx0) / char_w))]
               + [max(1, round((b["bbox_in_region"][2] - b["bbox_in_region"][0]) / char_w))
                  for b in after])
        words = [(m.start() + lo, m.end() + lo,
                  max(1, len(fold_to_unicode(m.group(0)).strip())))
                 for m in re.finditer(r"\S+", content[lo:hi])]
        n, mw = len(est), len(words)
        INF = 1e9
        dp = [[INF] * (mw + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0
        for a in range(n + 1):
            for b in range(mw + 1):
                c = dp[a][b]
                if c >= INF:
                    continue
                if a < n and b < mw:
                    dp[a + 1][b + 1] = min(dp[a + 1][b + 1],
                                           c + min(6, abs(est[a] - words[b][2])))
                if a < n:
                    dp[a + 1][b] = min(dp[a + 1][b], c + 2 + 0.3 * est[a])
                if b < mw:
                    dp[a][b + 1] = min(dp[a][b + 1], c + 2 + 0.3 * words[b][2])
        # backtrack to recover which word each bg token matched
        assign: dict[int, int] = {}
        a, b = n, mw
        while a > 0 or b > 0:
            c = dp[a][b]
            if a > 0 and b > 0 and abs(dp[a - 1][b - 1] + min(6, abs(est[a - 1] - words[b - 1][2])) - c) < 1e-9:
                assign[a - 1] = b - 1
                a, b = a - 1, b - 1
            elif a > 0 and abs(dp[a - 1][b] + 2 + 0.3 * est[a - 1] - c) < 1e-9:
                a -= 1
            else:
                b -= 1
        win_lo, win_hi = lo, hi
        for ai in range(len(before) - 1, -1, -1):
            if ai in assign:
                win_lo = words[assign[ai]][1]
                break
        for ai in range(len(before) + 1, n):  # +1 skips the group pseudo-token
            if ai in assign:
                win_hi = words[assign[ai]][0]
                break
        return (win_lo, win_hi) if win_lo < win_hi else (lo, hi)

    def _loose_ok(letters: str, w: str) -> bool:
        """The decode only has to LOCATE the word (GLM's letters get frakturized),
        so one misdecoded letter is tolerated. Accepted are a full subsequence,
        or an LCS of >= 75% / all-but-one of the letters for groups of 3+."""
        if 2 * len(letters) < len(w):
            return False
        if _is_subseq(letters, w):
            return True
        lcs = sum(b.size for b in difflib.SequenceMatcher(None, letters, w).get_matching_blocks())
        # min() because a plain 0.75 floor demanded zero errors for 3-letter groups
        if len(letters) >= 3 and lcs >= min(0.75 * len(letters), len(letters) - 1):
            return True
        # 2-letter groups get one mismatch when the words are the same length
        # and open with the same letter (the LCS route needs >= 3 letters)
        return len(letters) == 2 and len(w) == 2 and letters[0] == w[0]

    def _find_candidate(letters: str, lo2: int, hi2: int, relaxed: bool):
        """Return (start, end, replace_with_decode: bool) or None."""
        cands = list(_WORD.finditer(content_n, lo2, hi2))
        for m in cands:
            w = m.group(0).replace(" ", "")
            if _loose_ok(letters, w):
                return m.start(), m.end(), False
        if relaxed and len(letters) >= 3:
            for i0 in range(len(cands)):
                for i1 in range(i0, min(i0 + 4, len(cands))):
                    w = "".join(c.group(0) for c in cands[i0:i1 + 1]).replace(" ", "")
                    if difflib.SequenceMatcher(None, letters, w).ratio() >= 0.75:
                        return cands[i0].start(), cands[i1].end(), True
        return None

    interp_matched: set[int] = set()
    override_notes: list[dict] = []  # decode-trust overrides, surfaced in review
    used_end = 0
    for grp in groups:
        letters = "".join(_FRAK_CHAR_TO_LETTER.get(c, c)
                          for j in grp for c in cls_tokens[j]["reconstructed_string"]) \
            .translate(_VD_CANON)
        lo = max([used_end] + [spans[i]["end"] for i, j in pairs if j < grp[0]])
        hi = min([spans[i]["start"] for i, j in pairs if j > grp[-1]] + [len(content)])
        if lo >= hi:
            continue
        win_lo, win_hi = _refine_window(grp, lo, hi)
        narrowed = (win_lo, win_hi) != (lo, hi)
        # a narrowed window is trusted, since a candidate outside it is another
        # group's territory
        hit = _find_candidate(letters, win_lo, win_hi, relaxed=narrowed)
        if not hit and narrowed:
            # the anchored window missed, so retry the STRICT rule on the full gap;
            # strict single-word matching cannot claim another group's territory
            hit = _find_candidate(letters, lo, hi, relaxed=False)
        if not hit:
            # unanchored-run insertion, for when GLM dropped the letter run outright
            # and there is no anchor text to map onto. Insert the classifier's
            # decode after the paired non-fraktur token to the group's left.
            # The guards are >= 3 letters, every glyph >= 0.99, and a paired left
            # neighbour on the same line within 120 px, because prose
            # hallucinations have no such neighbour.
            confs = [s["confidence"] for j in grp
                     for s in cls_tokens[j].get("modifier_sequence", [])
                     if s["glyph"] != "reject"]
            j2i = {j: i for i, j in pairs}
            # the left anchor is strict because relaxing it measurably let wrong chains in
            anchor = None
            t0 = cls_tokens[grp[0]]
            for j in range(grp[0] - 1, -1, -1):
                tj = cls_tokens[j]
                if tj["line"] != t0["line"]:
                    break
                if tj["prediction"]["family"] != "fraktur_cluster":
                    if j in j2i and t0["bbox_in_region"][0] - tj["bbox_in_region"][2] <= 120:
                        anchor = spans[j2i[j]]["end"]
                    break
            if len(letters) >= 3 and confs and min(confs) >= 0.99 and anchor is not None:
                pos = anchor
                in_math = any(a <= pos < b for a, b in math_ranges)
                frak = "".join(_FRAKTUR_LOWER.get(c, c) for c in letters)
                tex = r"\mathfrak{%s}" % letters
                si = len(spans)
                spans.append({"start": pos, "end": pos, "key": "𝔉",
                              "in_math": in_math, "render_uni": " " + frak,
                              "render_tex": " " + (tex if in_math else "$%s$" % tex)})
                pairs.append((si, grp[0]))
                interp_matched.update(grp)
                override_notes.append({"why": "unanchored fraktur run inserted "
                                              "after left-neighbour siglum",
                                       "letters": letters})
                continue
            if os.environ.get("WB_DEBUG_INTERP"):
                print(f"[interp-miss] letters={letters!r} narrowed={narrowed} "
                      f"win={content[win_lo:win_hi][:90]!r} gap={content[lo:hi][:150]!r}")
            continue
        s0, e0, use_decode = hit
        # the narrowed window may cut a letter run mid-way, so re-extend the match
        # up to the full gap and keep it only if it still passes
        m2 = _WORD.match(content_n, s0, hi)
        if m2 and m2.end() > e0:
            w2 = m2.group(0).replace(" ", "")
            if _loose_ok(letters, w2) if not use_decode \
                    else difflib.SequenceMatcher(None, letters, w2).ratio() >= 0.75:
                e0 = m2.end()
        ws, we = s0, e0  # word extent, before any wrapper absorption
        # a relaxed hit trusts the classifier decode, but when the re-extended
        # match is a single plain letter run whose GLM letters pass the STRICT test
        # after all, flip back to in-place mapping and keep GLM as the letter source
        if use_decode and re.fullmatch(r"[a-z]+(?: [a-z])*", content_n[ws:we]) \
                and _loose_ok(letters, content_n[ws:we].replace(" ", "")):
            use_decode = False
        # absorb a \text{...}/\mathrm{...} wrapper around the matched slice
        head = re.search(r"\\(?:text(?:bf|it|rm)?|math(?:rm|bf|it))\s*\{\s*$", content[:s0])
        tail_br = re.match(r"\s*\}", content[e0:])
        if head and tail_br:
            s0, e0 = head.start(), e0 + tail_br.end()
        in_math = any(a <= s0 < b for a, b in math_ranges)
        if use_decode:   # GLM garbled the letter run, so trust the classifier decode
            frak = "".join(cls_tokens[j]["reconstructed_string"] for j in grp)
            tex = r"\mathfrak{%s}" % letters
        else:            # GLM letters are a superset, so map them in place
            word = content_n[ws:we]
            # Decode-trust override. Where the two sides disagree at equal letter
            # counts, a >= 0.99-confidence glyph decode overrides the GLM letter,
            # and positions below that keep GLM.
            dec = [(s["glyph"].translate(_VD_CANON), s["confidence"]) for j in grp
                   for s in cls_tokens[j].get("modifier_sequence", [])
                   if s["glyph"] != "reject"]
            glm_pos = [i2 for i2, ch in enumerate(word) if ch.islower()]
            glm_letters = "".join(word[i2] for i2 in glm_pos)
            _ins = next((k2 for k2 in range(len(glm_letters))
                         if glm_letters[k2] != letters[k2]), len(glm_letters)) \
                if len(letters) == len(glm_letters) + 1 else 0
            if (len(dec) == len(glm_pos) + 1 and len(glm_pos) == len(word)
                    and _is_subseq(glm_letters, letters)
                    and 0 < _ins < len(letters) - 1
                    # INTERIOR insertions only, because a leading/trailing extra
                    # letter is the signature of a stray group member or duplicated
                    # ink, not of GLM dropping a letter mid-run
                    and dec and min(cf2 for _, cf2 in dec) >= 0.99):
                # GLM dropped exactly ONE letter the classifier decoded at full
                # confidence; the run must be plain lowercase so no capital is
                # swallowed by the swap
                override_notes.append({"why": "decode inserts letter GLM dropped "
                                              "in fraktur run",
                                       "glm": glm_letters, "classifier": letters})
                word = letters
            elif len(dec) == len(glm_pos) and "".join(g for g, _ in dec) == letters:
                w = list(word)
                for i2, (g, conf) in zip(glm_pos, dec):
                    if conf >= 0.99 and g != w[i2]:
                        override_notes.append({"why": "decode override in fraktur run",
                                               "glm": w[i2], "classifier": g,
                                               "confidence": conf})
                        w[i2] = g
                word = "".join(w)
            frak = re.sub(r"[a-z]", lambda mm: _FRAKTUR_LOWER.get(mm.group(0), mm.group(0)), word)
            tex = re.sub(r"[a-z]+", lambda mm: r"\mathfrak{%s}" % mm.group(0), word)
        si = len(spans)
        spans.append({"start": s0, "end": e0, "key": "𝔉",
                      "in_math": in_math, "render_uni": frak,
                      "render_tex": tex if in_math else "$%s$" % tex})
        pairs.append((si, grp[0]))
        interp_matched.update(grp)
        used_end = e0

    # unanchored insertion for standalone declaratio marks, which GLM routinely
    # omits from its text, so there is no anchor to pair with. Insert a >= 0.99-
    # confidence mark after the paired token to its left on the same line;
    # | IS the unit separator, and its grammatical slot is exactly "between two
    # witness groups". Zero-width span, review-visible via override_notes.
    j2i_d = {j: i for i, j in pairs}
    for j, tok in enumerate(cls_tokens):
        if j in j2i_d or j in interp_matched:
            continue
        if tok["prediction"]["family"] != "declaratio_signorum":
            continue
        rec = tok.get("reconstructed_string")
        # '|' ONLY, because inserting † or ~ measurably duplicated raised daggers
        # already absorbed into sigla tokens
        if rec != "|" \
                or tok["prediction"].get("family_confidence", 0.0) < 0.99:
            continue
        # shape gate, since a real | is a full-height hairline while letter-stem
        # fragments that fool the class are shorter and squatter
        bx0, by0, bx1, by1 = tok["bbox_in_region"]
        if (by1 - by0) < 5 * (bx1 - bx0) or (by1 - by0) < 24:
            continue
        anchor2 = None
        for j2 in range(j - 1, -1, -1):
            t2 = cls_tokens[j2]
            if t2["line"] != tok["line"]:
                break
            if j2 in j2i_d:
                if tok["bbox_in_region"][0] - t2["bbox_in_region"][2] <= 150:
                    anchor2 = j2i_d[j2]
                break
        if anchor2 is None:
            continue
        pos = spans[anchor2]["end"]
        in_math = any(a <= pos < b for a, b in math_ranges)
        tex = _DECL_TO_LATEX.get(rec, rec)
        si = len(spans)
        spans.append({"start": pos, "end": pos, "key": rec, "in_math": in_math,
                      "render_uni": " " + rec,
                      "render_tex": " " + (tex if in_math or "\\" not in tex
                                           else "$%s$" % tex)})
        pairs.append((si, j))
        interp_matched.add(j)
        override_notes.append({"why": "unanchored %s inserted after left-neighbour "
                                      "paired token" % rec, "line": tok["line"]})
    n_interpolated = len(interp_matched)

    # Trailing-star absorption, because GLM prints the '*' of a siglum outside the
    # math segment ($\Psi^{D}$*;) while the classifier decodes it inside the token.
    # An unmatched '*' span immediately after a replaced siglum whose decode
    # already ends in '*' renders as nothing.
    matched_glm_now = {i for i, _ in pairs}
    for i, j in list(pairs):
        tok = cls_tokens[j]
        if tok["prediction"]["family"] != "manuscript_siglum" \
                or not tok["reconstructed_string"].endswith(("*", "†")):
            continue
        tail = tok["reconstructed_string"][-1]
        for k, s in enumerate(spans):
            if k in matched_glm_now or s["key"] != tail:
                continue
            gap_w = s["start"] - spans[i]["end"]
            if 0 <= gap_w <= 2 and re.fullmatch(r"[\s$]*", content[spans[i]["end"]:s["start"]]):
                s["render_uni"] = ""
                s["render_tex"] = ""
                pairs.append((k, j))
                matched_glm_now.add(k)
                break

    # When span == decode after folding notation to Unicode, the span was RIGHT,
    # so render it as "confirmed" rather than struck-through as a correction.
    def _norm(s: str) -> str:
        # ^{}$ are notation residue; only CONTENT decides same/subset, never markup
        return re.sub(r"[\s^{}$]+", "", unicodedata.normalize("NFC", s))

    same, kept_glm = {}, []
    for i, j in pairs:
        raw = content[spans[i]["start"]:spans[i]["end"]]
        g = _norm(fold_to_unicode(raw))
        c = _norm(spans[i]["render_uni"]) if "render_uni" in spans[i] \
            else _norm(recon_to_unicode(cls_tokens[j]))
        if c == g:
            same[i] = True
            continue
        it = iter(g)
        if c and len(c) < len(g) and all(ch in it for ch in c) \
                and "render_uni" not in spans[i]:
            gf, cf = g.casefold(), c.casefold()
            if len(cf) > 1 and gf == cf + cf[1:]:
                # For an in-span double-write (\Phi^{Z*}^{z*}) the content agrees
                # once the duplicate collapses, so confirm the classifier's single copy
                same[i] = True
                kept_glm.append({"why": "OCR span duplicates its own modifiers; "
                                        "duplicate collapsed",
                                 "span": raw.strip(),
                                 "token": cls_tokens[j]["reconstructed_string"]})
                continue
            p = cls_tokens[j]["prediction"]
            # modifier-char multiset difference GLM-minus-classifier, order
            # preserved, regrouped into witness units (letter plus its trailing
            # digits/stars, so "O2" re-emits as O², not a bare O)
            def _fseq(s2: str) -> list[str]:
                out = []
                for ch in s2:
                    for lit in unicodedata.normalize("NFKC", ch):
                        if lit.isalnum() or lit in "*†":
                            out.append(lit.upper() if lit.isalpha() else lit)
                return out
            gseq, cseq = _fseq(g[1:]), _fseq(c[1:])
            rem = list(cseq)
            leftover = []
            for chx in gseq:
                if chx in rem:
                    rem.remove(chx)
                else:
                    leftover.append(chx)
            units_left: list[str] = []
            for chx in leftover:
                if chx.isalpha():
                    units_left.append(chx)
                elif units_left:
                    units_left[-1] += chx
            extras = (units_left if all(u[0] in _WITNESS_SET for u in units_left)
                      else [])
            _SUPD = {"1": "¹", "2": "²", "3": "³"}

            def _extra_tails(in_math: bool) -> tuple[str, str]:
                uni = " " + " ".join(
                    u[0] + "".join(_SUPD.get(d, d) for d in u[1:]) for u in extras)
                parts = []
                for u in extras:
                    if len(u) == 1:
                        parts.append(u)
                    else:
                        t = "%s^{%s}" % (u[0], u[1:])
                        parts.append(t if in_math else "$%s$" % t)
                return uni, " " + " ".join(parts)
            if _illegal_mods(spans[i]["key"], g[1:]) \
                    and not _illegal_mods(spans[i]["key"], c[1:]) \
                    and p.get("family_confidence", 1.0) >= 0.9 \
                    and p.get("base_confidence", 1.0) >= 0.9:
                # Illegal-modifier branch. A grammatically illegal modifier in the
                # longer GLM reading is the signature of GLM swallowing a
                # NEIGHBOURING witness into the superscript group, not of the
                # classifier under-reading, so a legal, confident decode wins.
                # Swallowed letters that are attested bare-witness capitals are
                # re-emitted as plain text after the replacement.
                if extras:
                    t_uni, t_tex = _extra_tails(spans[i]["in_math"])
                    spans[i]["render_uni"] = recon_to_unicode(cls_tokens[j]) + t_uni
                    spans[i]["render_tex"] = recon_to_latex(
                        cls_tokens[j], in_math=spans[i]["in_math"]) + t_tex
                kept_glm.append({"why": "OCR span grammatically illegal for its "
                                        "base; classifier replacement allowed",
                                 "span": raw.strip(),
                                 "token": cls_tokens[j]["reconstructed_string"]})
                same[i] = False
                continue
            # Over-grouped-span split. GLM crams a neighbouring BASELINE witness
            # into the superscript group (\Sigma^{M B} for "Σᴹ B", where B is legal
            # for Σ so the illegal-mods branch never sees it). Geometry
            # discriminates the two, because a swallowed witness is full-height ink
            # immediately right of the token while an under-read modifier is small
            # raised ink.
            if extras and leftover and leftover[0].isalpha() \
                    and any(chx.isalpha() for chx in cseq) \
                    and p.get("family_confidence", 1.0) >= 0.9 \
                    and p.get("base_confidence", 1.0) >= 0.9:
                tok = cls_tokens[j]
                tx1 = tok["bbox_in_region"][2]
                th = tok["bbox_in_region"][3] - tok["bbox_in_region"][1]
                # the full-height box right of the token must be OTHERWISE
                # unaccounted for, because when the very next GLM text is itself a
                # capital or macro, the box is that witness's ink and the
                # classifier merely under-read the group
                after = content[spans[i]["end"]:].lstrip("$} \t\n")
                unclaimed = not (after[:1].isupper() or after[:1] == "\\"
                                 or after[:1] in "ΘΦΨΩΛΣΠ")
                if unclaimed and any(
                        b["line"] == tok["line"]
                        and -4 <= b["bbox_in_region"][0] - tx1 <= 24
                        and b["bbox_in_region"][3] - b["bbox_in_region"][1] >= 0.7 * th
                        for b in bg_tokens):
                    t_uni, t_tex = _extra_tails(spans[i]["in_math"])
                    spans[i]["render_uni"] = recon_to_unicode(tok) + t_uni
                    spans[i]["render_tex"] = recon_to_latex(
                        tok, in_math=spans[i]["in_math"]) + t_tex
                    same[i] = False
                    kept_glm.append({"why": "OCR span over-grouped a neighbouring "
                                            "full-height witness into the superscript; "
                                            "split per classifier",
                                     "span": raw.strip(),
                                     "token": tok["reconstructed_string"]})
                    continue
            # Superset guard. The decode is a strict SUBSEQUENCE of what GLM read,
            # meaning the classifier merely missed small ink and saw nothing GLM
            # didn't, so GLM wins; count as confirmed, flag for review.
            ru = fold_to_unicode(raw)
            if not re.search(r"[A-Za-z]{2}", ru):
                # a pure siglum span, where spaces and stray ^ are fold artifacts
                ru = re.sub(r"[\s^]", "", ru)
            spans[i]["render_uni"] = ru
            spans[i]["render_tex"] = raw
            same[i] = True
            kept_glm.append({"why": "classifier decoded subset of OCR span; OCR kept",
                             "span": raw.strip(),
                             "token": cls_tokens[j]["reconstructed_string"]})
            continue
        same[i] = False

    matched_glm = {i for i, _ in pairs}
    matched_cls = {j for _, j in pairs} | interp_matched
    review = list(kept_glm) + override_notes + geom_overrides + cleanup_notes + merged_notes
    for t in gated:
        review.append({"why": "implausible siglum decode (repeated modifier letter); "
                              "excluded from alignment",
                       "token": t["reconstructed_string"],
                       "line": t["line"], "family": t["prediction"]["family"]})
    for t in geo_gated:
        review.append({"why": "geometry veto: siglum box sits in the lowercase band "
                              "(top >= 8px below line cap reference); "
                              "excluded from alignment",
                       "token": t["reconstructed_string"],
                       "line": t["line"], "family": t["prediction"]["family"]})
    for t in embw_gated:
        review.append({"why": "embedded-w veto: single 𝔴 glued (<=8px) to a "
                              "non-fraktur neighbour is a prose ligature-pair "
                              "misread; excluded from alignment",
                       "token": t["reconstructed_string"],
                       "line": t["line"], "family": t["prediction"]["family"]})
    for j, t in enumerate(cls_tokens):
        if j not in matched_cls:
            review.append({"why": "classifier token not found in OCR text",
                           "token": t["reconstructed_string"],
                           "line": t["line"], "family": t["prediction"]["family"]})
    replaced_ranges = [(spans[i]["start"], spans[i]["end"]) for i, _ in pairs]
    for i, s in enumerate(spans):
        if i not in matched_glm:
            if any(a <= s["start"] and s["end"] <= b for a, b in replaced_ranges):
                continue  # swallowed by an extended replacement span, so not left behind
            raw = content[s["start"]:s["end"]].strip()
            if set(raw) <= set("*]+~|"):
                continue  # a legitimate plain-text symbol expected to stay, so not review-worthy
            review.append({"why": "OCR sigla-like span not matched by classifier",
                           "span": raw})

    # adjacent double-write, where GLM writes the siglum twice, a span then the
    # same content again as plain text right after (\Psi^{\mathrm{FM2}}$FM²).
    # When an UNPAIRED plain-text run adjacent to a written-back span folds to
    # the token's content, drop the duplicate from the corrected output.
    # The floor of >= 2 folded chars ensures a single capital (a real independent
    # witness) is never eaten.
    suppressed: list[tuple[int, int]] = []
    # only PAIRED spans veto the scan, because the duplicate text itself usually
    # parses as an unmatched span candidate and must not protect itself from removal
    paired_ranges = [(spans[i]["start"], spans[i]["end"]) for i, _ in pairs]
    for i, j in pairs:
        end = spans[i]["end"]
        m = re.match(r"(\s{0,2}\$?\s{0,2})((?:\^\{[^{}]*\}|[A-Za-z0-9*†¹²³ᴬ-ᶻꟳ™])+)",
                     content[end:end + 24])
        if not m:
            continue
        a, b = end + m.start(2), end + m.end(2)
        if any(x < b and a < y for x, y in paired_ranges if (x, y) != (spans[i]["start"], end)):
            continue  # the run belongs to another REPLACED span, which is alignment's business
        tokf = unicodedata.normalize(
            "NFKC", _norm(recon_to_unicode(cls_tokens[j]))).casefold()
        # Right-trim retry, because the greedy run may glue a FOLLOWING witness
        # onto the duplicate. Shrink from the right until the folded run is a
        # suffix of the token, braces kept balanced.
        run = m.group(2)
        while run:
            dupf = unicodedata.normalize("NFKC", _norm(fold_to_unicode(run))).casefold()
            if len(dupf) >= 2 and run.count("{") == run.count("}") \
                    and (dupf == tokf or tokf.endswith(dupf)):
                suppressed.append((a, a + len(run)))
                review.append({"why": "plain-text duplicate of written-back token removed",
                               "text": run,
                               "token": cls_tokens[j]["reconstructed_string"]})
                break
            run = run[:-1]

    # replaced spans enter the fold as \x00<idx>\x00 placeholders so the WHOLE
    # region folds as one string, since folding piece by piece left wrapper
    # residue whenever a wrapper straddled a replacement boundary
    span_order = sorted((i for i, _ in pairs), key=lambda i: spans[i]["start"])
    _SENT = re.compile("\x00(\\d+)\x00")

    def _plain(a: int, b: int, cut: bool) -> str:
        if not cut or not suppressed:
            return content[a:b]
        out, pos = [], a
        for x, y in sorted(suppressed):
            if y <= pos or x >= b:
                continue
            out.append(content[pos:max(pos, x)])
            pos = min(b, y)
        out.append(content[pos:b])
        return "".join(out)

    # ---- OUTPUT RENDERING: both encodings off one sentinel string ----
    def _with_sentinels(cut: bool = False) -> str:
        parts, pos = [], 0
        for i in span_order:
            parts += [_plain(pos, spans[i]["start"], cut), "\x00%d\x00" % i]
            pos = spans[i]["end"]
        parts.append(_plain(pos, len(content), cut))
        return "".join(parts)

    def build(rendered: dict[int, str], fold: bool) -> tuple[str, str]:
        """Return (plain corrected string, html with <mark> highlights)."""
        text = _with_sentinels(cut=True)  # double-write duplicates leave the corrected views only
        out, out_html = [], []
        for k, seg in enumerate(_SENT.split(fold_to_unicode(text) if fold else text)):
            if k % 2:
                rep = rendered[int(seg)]
                out.append(rep)
                out_html.append('<mark class="%s">%s</mark>'
                                % ("same" if same[int(seg)] else "ins", html.escape(rep)))
            else:
                out.append(seg)
                out_html.append(html.escape(seg))
        return "".join(out), "".join(out_html)

    ren_uni = {i: spans[i]["render_uni"] if "render_uni" in spans[i]
                  else recon_to_unicode(cls_tokens[j])
               for i, j in pairs}
    ren_tex = {i: spans[i]["render_tex"] if "render_tex" in spans[i]
                  else recon_to_latex(cls_tokens[j], in_math=spans[i]["in_math"])
               for i, j in pairs}
    corr_uni, corr_uni_html = build(ren_uni, fold=True)
    corr_tex, _ = build(ren_tex, fold=False)
    corr_uni, corr_uni_html = _polish_uni(corr_uni), _polish_uni(corr_uni_html)
    corr_tex = _polish_tex(corr_tex)
    # the single choke point for the d->v adjudication flip, covering span renders,
    # gap-interpolation renders and untouched GLM text alike
    corr_uni, corr_uni_html = _vd_fix_uni(corr_uni), _vd_fix_uni(corr_uni_html)
    corr_tex = _vd_fix_tex(corr_tex)

    # the original with replaced spans highlighted, rendered twice: raw GLM notation,
    # and the same text folded to Unicode (the view a reviewer proofreads against)
    orig_html, pos = [], 0
    for i in span_order:
        s = spans[i]
        cls = "same" if same[i] else "del"
        orig_html.append(html.escape(content[pos:s["start"]]))
        orig_html.append('<mark class="%s">%s</mark>'
                         % (cls, html.escape(content[s["start"]:s["end"]])))
        pos = s["end"]
    orig_html.append(html.escape(content[pos:]))
    orig_uni_html = []
    for k, seg in enumerate(_SENT.split(fold_to_unicode(_with_sentinels()))):
        if k % 2:
            s = spans[int(seg)]
            orig_uni_html.append('<mark class="%s">%s</mark>'
                                 % ("same" if same[int(seg)] else "del",
                                    html.escape(fold_to_unicode(content[s["start"]:s["end"]]))))
        else:
            orig_uni_html.append(html.escape(seg))

    n_confirmed = sum(same.values())
    return {
        "n_spans_glm": len(spans), "n_tokens_classifier": len(cls_tokens),
        "n_replaced": len(pairs), "n_confirmed": n_confirmed,
        "n_corrected": len(pairs) - n_confirmed,
        "n_interpolated": n_interpolated, "review": review,
        "corrected_unicode": corr_uni, "corrected_latex": corr_tex,
        "orig_html": "".join(orig_html), "orig_uni_html": "".join(orig_uni_html),
        "corr_html": corr_uni_html,
    }


def reconcile_chunk(model_pages: list, report: list, out_dir: Path,
                    stem: str, align: str = "global", geometry: bool = True,
                    report_stem: str | None = None) -> dict:
    """Stage-3 driver: reconcile a whole chunk and write the output files.

    This is the library equivalent of the original ``writeback.py`` command;
    the reconciliation logic itself is unchanged (:func:`writeback_region`).

    Args:
        model_pages: the pages-of-regions structure of the stage-1 artifact
            (the ``<stem>_model.json`` content).
        report: the region/token report produced by the classifier stage
            (:func:`vulgata_scan.classifier.stage.run`).
        out_dir: output dir for the three result files
            (``<stem>_corrected_latex.json``, ``<stem>_corrected_unicode.md``,
            ``<report-stem>_writeback.json``).
        stem: file stem used for the output names.
        align: "global" (cost-based alignment, the production route) or
            "difflib" (exact-key blocks, the frozen regression baseline).
        geometry: whether the global route may use word-level bbox geometry.
        report_stem: stem for the writeback.json name (defaults to ``stem``).

    Returns:
        A dict with the written file paths and the correction counters.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(json.dumps(report))  # never mutate the caller's report

    by_region = {(r["page"], r["region_index"]): r for r in report}
    pages = json.loads(json.dumps(model_pages))  # deep copy; never mutate the caller's
    pages_tex = json.loads(json.dumps(pages))   # deep copies
    pages_uni = json.loads(json.dumps(pages))
    wb_info, review_all = {}, []
    n_corrected = n_confirmed = n_interp = 0

    for pi, page in enumerate(pages):
        for ri, region in enumerate(page):
            rep = by_region.get((pi + 1, region["index"]))
            content = split_sup_groups(region.get("content", ""))
            if rep is None:
                continue
            cls_tokens = [t for t in rep["tokens"] if t["prediction"]["family"] != "background"]
            cls_tokens.sort(key=lambda t: (t["line"], t["bbox_in_region"][0]))
            bg_tokens = [t for t in rep["tokens"] if t["prediction"]["family"] == "background"]
            info = writeback_region(content, cls_tokens, bg_tokens, align=align,
                                    geometry=geometry)
            wb_info[f"p{pi + 1}_r{region['index']}"] = info
            n_corrected += info["n_corrected"]
            n_confirmed += info["n_confirmed"]
            n_interp += info["n_interpolated"]
            for rv in info["review"]:
                review_all.append({"page": pi + 1, "region": region["index"], **rv})
            pages_tex[pi][ri]["content_original"] = content
            pages_tex[pi][ri]["content"] = info["corrected_latex"]
            pages_uni[pi][ri]["content_original"] = content
            pages_uni[pi][ri]["content"] = info["corrected_unicode"]

    # keep just the regions the pipeline processed (Desunt witness line +
    # two-column apparatus); everything else is dropped from the deliverables
    pages_tex = [[r for r in page if (pi + 1, r["index"]) in by_region]
                 for pi, page in enumerate(pages_tex)]
    pages_uni = [[r for r in page if (pi + 1, r["index"]) in by_region]
                 for pi, page in enumerate(pages_uni)]

    # The '## page N region M (label)' heading format below is a hard contract;
    # downstream evaluation tooling parses it, so do not edit.
    jp = out_dir / f"{stem}_corrected_latex.json"
    jp.write_text(json.dumps(pages_tex, ensure_ascii=False, indent=2), encoding="utf-8")
    md = out_dir / f"{stem}_corrected_unicode.md"
    md.write_text("\n\n".join(
        f"## page {pi + 1} region {r['index']} ({r.get('label')})\n\n{r.get('content', '')}"
        for pi, page in enumerate(pages_uni) for r in page), encoding="utf-8")
    print("->", jp.name, "+", md.name)

    # goes to --out-dir, not next to the report, so a variant run cannot silently
    # regenerate the canonical writeback.json in place
    wb_path = out_dir / ((report_stem or stem) + "_writeback.json")
    wb_path.write_text(json.dumps({"regions": wb_info, "review": review_all},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {wb_path.name}   ({n_corrected} corrected, {n_confirmed} confirmed, "
          f"{n_interp} tokens rescued by gap interpolation, {len(review_all)} review items)")
    return {"corrected_latex_json": jp, "corrected_unicode_md": md,
            "writeback_json": wb_path,
            "n_corrected": n_corrected, "n_confirmed": n_confirmed,
            "n_interpolated": n_interp, "n_review": len(review_all)}
