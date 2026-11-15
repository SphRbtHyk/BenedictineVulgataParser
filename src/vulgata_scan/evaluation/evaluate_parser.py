#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
apparatus_eval.py — evaluation of the apparatus-OCR pipeline against a
hand-annotated ground truth.

Both sides go through the same parser, so parser behaviour is common-mode and
cancels out; what the numbers below capture is the difference in what the OCR
put on the page.

This version prints, under section headings that follow the paper one to one,
every quantity the evaluation section names.  Map from the paper to the output:

  §Alignment
    similarity sigma = .55*sigma_txt + .30*sigma_wit + .15*sigma_type   entry_sim()
    no-match threshold 0.20                                             --sim-floor
    verses present in GT and absent from the output -> "missing"        [0] structure

  §Ground-truth description ("X verses, X units, X segments")
    the placeholders are filled by                                      [C] census

  §Metric 1 — detecting apparatus structure
    entry_precision, entry_recall, entry_F1                             [1]
    the same three broken down by variant type                          [1] table
    detection AND classification failures of one type, printed first    [S]

  §Metric 2 — witness extraction
    sigla_group_exact_match                                             [2]
    sigla_exact_match, sigla_precision, sigla_recall, sigla_F1          [2]
    nature of the differences (error taxonomy, glyph confusions)        [2]
    `witnesses` and `missing` scored apart and together                 [2]

  §Metric 3 — quality of reading transcription
    CER on reading / in_text / anchor                                   [3]
    exact_reading                                                       [3]

  §Metric 4 — computing witness agreements
    agreement matrix ("grid tables"), knn_overlap at k=3                [4]

  §Global_CER (raw OCR vs annotated ground truth)                       [G]
    from --gt-text/--pred-text if given, otherwise recomputed from the
    parsed text fields and labelled as such.

  Figure "example of evaluation metric computation"
    --example CH:V prints the worked computation for one verse, plus a
    LaTeX tabular for the figure.

Everything else (bootstrap CIs, run-length analysis, alignment diagnostics)
is diagnostic and is not required by the paper; it is kept but clearly
separated under [D].

Usage
-----
    python3 apparatus_eval.py --gt gt.json --pred pred.json \
        --gt-text gt.txt --pred-text ocr.txt \
        --per-verse --dump-errors errors.tsv --latex --bootstrap 1000

Design notes:

* Alignment is done in two stages and never by list index, because a single OCR
  error (a lost `|`) shifts every following entry.  Verses are matched on
  (chapter, verse); entries inside a verse are matched by Needleman-Wunsch on the
  similarity above.  NW is order-preserving, which the paper's "units cannot be
  considered ordered" does not strictly license: --assign greedy gives the
  order-free maximum-similarity matching, and the two are reported to differ (or
  not) by --assign-compare.
* Unordered fields (witnesses, versions, citations) get precision/recall, not
  accuracy: dropping a siglum and inventing one are different failures of a VLM
  and the paper needs to tell them apart.
* Ordered text gets CER plus exact-match rate; CER alone hides that most fields
  are perfect and a few are catastrophic.  CER is normalised by the UNION length
  max(|gold|, |pred|) so that an invented field cannot add numerator against a
  zero denominator, and so that CER and the exact-match rate are computed over
  the same population and stay on the same scale.
* Canonical sigla: `\frak{a}` == `\mathfrak{a}` == `a`, `\Theta^{M^1}` ==
  `\Theta^{M1}` == `Θ^M1`.
* Every siglum mismatch is classified (base glyph / superscript letter / hand
  mark / dropped / spurious) so the paper can say *where* in the glyph the OCR
  fails rather than only how often.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# shared LaTeX normalisation (from the CER script; keep both files together)
# --------------------------------------------------------------------------
try:
    from apparatus_cer import normalize as _tex_normalize, edit_distance
except ImportError:  # stand-alone fallback, deliberately minimal
    GREEK = {"Lambda": "Λ", "Sigma": "Σ", "Theta": "Θ", "Phi": "Φ", "Psi": "Ψ",
             "Omega": "Ω", "Pi": "Π", "Gamma": "Γ", "Delta": "Δ", "Xi": "Ξ"}

    def _tex_normalize(s: str, **_kw) -> str:
        s = re.sub(r"\\(?:math)?(?:rm|it|bf|frak|sf|tt|cal|bb)\s*\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\text(?:it|bf|rm|sc)?\s*\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\([A-Za-z]+)", lambda m: GREEK.get(m.group(1), " "), s)
        s = s.replace("$", " ").replace("{", " ").replace("}", " ")
        s = unicodedata.normalize("NFKC", s)
        s = "".join(c for c in unicodedata.normalize("NFD", s)
                    if not unicodedata.combining(c))
        s = re.sub(r"\s*([\^_])\s*", r"\1", s)
        return re.sub(r"\s+", " ", s).strip()

    def edit_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


# --------------------------------------------------------------------------
# long-string edit distance (Global_CER): quadratic DP is unusable on 26k
# characters, so use Ukkonen's banded DP with a doubling threshold. Exact,
# and O(n*d) when the two strings are close, which they are here.
# --------------------------------------------------------------------------

def _banded_edit(a: str, b: str, t: int) -> Optional[int]:
    n, m = len(a), len(b)
    if abs(n - m) > t:
        return None
    INF = t + 1
    prev = [INF] * (m + 1)
    for j in range(0, min(m, t) + 1):
        prev[j] = j
    for i in range(1, n + 1):
        cur = [INF] * (m + 1)
        lo = max(1, i - t)
        hi = min(m, i + t)
        if lo == 1:
            cur[0] = i if i <= t else INF
        ai = a[i - 1]
        row_min = INF
        for j in range(lo, hi + 1):
            c = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
            cur[j] = c
            if c < row_min:
                row_min = c
        if row_min > t:
            return None
        prev = cur
    return prev[m] if prev[m] <= t else None


try:                                   # optional, only makes Global_CER faster
    from rapidfuzz.distance import Levenshtein as _rf_lev

    def _fast_edit(a: str, b: str) -> Optional[int]:
        return _rf_lev.distance(a, b)
except ImportError:                    # pragma: no cover
    def _fast_edit(a: str, b: str) -> Optional[int]:
        return None


def edit_distance_long(a: str, b: str) -> int:
    """Exact Levenshtein, usable on whole pages. Uses rapidfuzz when it is
    installed, otherwise Ukkonen's banded DP with a widening band."""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    d = _fast_edit(a, b)
    if d is not None:
        return d
    if len(a) * len(b) <= 250_000:
        return edit_distance(a, b)
    # start from a plausible band rather than from 1: a widening search pays
    # for every failed pass, and apparatus CER is a few percent at worst
    t = max(1, abs(len(a) - len(b)), len(a) // 32)
    limit = max(len(a), len(b))
    while t <= limit:
        d = _banded_edit(a, b, t)
        if d is not None:
            return d
        t *= 4
    return edit_distance(a, b)


# --------------------------------------------------------------------------
# canonical forms
# --------------------------------------------------------------------------

HAND_MARKS = set("*†‡0123456789¹²³")


def canon_text(s: Any) -> str:
    """Reading / note / annotation text, LaTeX spelling neutralised."""
    if s is None:
        return ""
    if isinstance(s, (list, tuple)):
        s = " ".join(str(x) for x in s)
    t = _tex_normalize(str(s))
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t


def canon_siglum(s: Any) -> str:
    """`\\Theta^{M^1}` == `\\Theta^{M1}` == `Θ^M1`; `\\frak{a}` == `\\mathfrak{a}` == `a`."""
    t = _tex_normalize(str(s))
    t = t.replace(" ", "")
    if "^" in t:
        base, _, rest = t.partition("^")
        rest = rest.replace("^", "")          # nested superscripts flattened
        t = base + "^" + rest
    return t


def split_siglum(sig: str) -> Tuple[str, str, str]:
    """canonical siglum -> (base glyph, superscript letters, hand marks).

    Ψ^{B*} -> (Ψ, B, *) ; Φ^{F^2} -> (Φ, F, 2) ; M^2 -> (M, '', 2) ; a -> (a, '', '')
    """
    base, _, sup = sig.partition("^")
    letters = "".join(c for c in sup if c not in HAND_MARKS)
    marks = "".join(c for c in sup if c in HAND_MARKS)
    return base, letters, marks


# --------------------------------------------------------------------------
# tolerant JSON loading (subsets pasted by hand are often truncated)
# --------------------------------------------------------------------------

def load_json(path: str) -> List[dict]:
    raw = open(path, encoding="utf-8").read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # salvage: cut back to the last complete top-level object
        depth, last_ok, in_str, esc = 0, None, False, False
        for i, ch in enumerate(raw):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 1:
                    last_ok = i
        if last_ok is None:
            raise
        salvaged = raw[: last_ok + 1] + "]"
        data = json.loads(salvaged)
        print("!! %s is truncated (%s); salvaged %d complete verse objects"
              % (path, exc.msg, len(data)), file=sys.stderr)
        return data


# --------------------------------------------------------------------------
# accessors — one place to change if the schema moves
# --------------------------------------------------------------------------

def verse_key(v: dict) -> str:
    ch = canon_text(v.get("chapter")) or "?"
    vs = v.get("verse")
    return "%s:%s" % (ch, canon_text(vs) if vs not in (None, "") else "tit")


def entry_segments(e: dict) -> List[dict]:
    """The segments of a unit.

    The paper's hierarchy is verse > unit > segment. The current schema stores a
    single `alternative_reading` per unit, so a unit holds exactly one segment;
    a list under `alternative_reading` or an `alternative_readings` key is also
    accepted so that the census and the agreement matrix stay correct if the
    schema grows a real segment list.
    """
    alt = e.get("alternative_reading")
    if alt is None:
        alt = e.get("alternative_readings")
    if alt is None:
        return []
    if isinstance(alt, dict):
        return [alt]
    if isinstance(alt, (list, tuple)):
        return [x for x in alt if isinstance(x, dict)]
    return []


def segment_witnesses(seg: dict) -> List[str]:
    return [canon_siglum(w) for w in (seg.get("witnesses") or [])]


def entry_witness_groups(e: dict) -> List[List[str]]:
    """One witness group per segment — the unit of `sigla_group_exact_match`
    and the unit over which two witnesses are said to 'agree'."""
    return [segment_witnesses(s) for s in entry_segments(e)]


def entry_witnesses(e: dict) -> List[str]:
    out: List[str] = []
    for g in entry_witness_groups(e):
        out.extend(g)
    return out


def entry_missing(e: dict) -> List[str]:
    return [canon_siglum(w) for w in (e.get("missing") or [])]


def entry_reading(e: dict) -> str:
    segs = entry_segments(e)
    return canon_text(" ".join(str(s.get("reading") or "") for s in segs))


TEXT_FIELDS = ("in_text", "anchor", "note")
# Fields entering the text score. `note` (editorial scholia: Hug., Guill., Iac.)
# is excluded by default -- it is discursive prose, not apparatus text, and its
# length would dominate the CER. Put it back with --score-notes.
# Mutated in main(), so it must stay a list.
TEXT_ORDER = ["reading", "in_text", "anchor", "annotations"]
# The three fields the paper names for metric 3.
PAPER_TEXT_FIELDS = ("reading", "in_text", "anchor")


def entry_texts(e: dict) -> Dict[str, str]:
    out = {f: canon_text(e.get(f)) for f in TEXT_FIELDS}
    out["reading"] = entry_reading(e)
    out["annotations"] = canon_text(e.get("annotations"))
    return out


def entry_text_blob(e: dict) -> str:
    t = entry_texts(e)
    return " ".join(t[k] for k in ("in_text", "anchor", "reading", "note", "annotations") if t[k])


def entry_versions(e: dict) -> List[str]:
    return [canon_text(x).lower() for x in (e.get("versions") or [])]


def entry_citations(e: dict) -> List[str]:
    return [canon_text(x) for x in (e.get("citations") or [])]


def entry_type(e: dict) -> str:
    return canon_text(e.get("variant_type")) or "-"


def all_entries(data: Iterable[dict]) -> Iterable[Tuple[str, dict]]:
    for v in data:
        key = verse_key(v)
        for e in (v.get("apparatus") or []):
            yield key, e


# ==========================================================================
# [C] census — fills the "X verses, X units, X segments" placeholders
# ==========================================================================

def census(data: List[dict], raw_text: Optional[str] = None) -> Dict[str, Any]:
    chapters, verses = set(), set()
    n_units = n_segments = 0
    types: Counter = Counter()
    wit: set = set()
    wit_missing: set = set()
    chars_fields = 0
    n_versions = n_citations = 0
    for v in data:
        key = verse_key(v)
        verses.add(key)
        chapters.add(key.split(":")[0])
        for e in (v.get("apparatus") or []):
            n_units += 1
            types[entry_type(e)] += 1
            groups = entry_witness_groups(e)
            n_segments += len(groups)
            for g in groups:
                wit.update(g)
            wit_missing.update(entry_missing(e))
            t = entry_texts(e)
            chars_fields += sum(len(t[f]) for f in TEXT_ORDER)
            n_versions += len(entry_versions(e))
            n_citations += len(entry_citations(e))
    out = {
        "chapters": len(chapters),
        "verses": len(verses),
        "units": n_units,
        "segments": n_segments,
        "witness_slots": sum(len(g) for _, e in all_entries(data)
                             for g in entry_witness_groups(e)),
        "distinct_witnesses": len(wit),
        "distinct_witnesses_incl_missing": len(wit | wit_missing),
        "chars_text_fields": chars_fields,
        "versions": n_versions,
        "citations": n_citations,
        "variant_types": dict(types),
    }
    if raw_text is not None:
        out["chars_raw"] = len(raw_text)
        out["chars_raw_normalised"] = len(canon_text(raw_text))
    return out


def print_census(name: str, c: Dict[str, Any]) -> None:
    print("  %s:" % name)
    print("     %-34s %8d" % ("chapters", c["chapters"]))
    print("     %-34s %8d" % ("verses", c["verses"]))
    print("     %-34s %8d" % ("units (apparatus entries)", c["units"]))
    print("     %-34s %8d" % ("segments (readings)", c["segments"]))
    print("     %-34s %8d" % ("witness slots (token sigla)", c["witness_slots"]))
    print("     %-34s %8d" % ("distinct witnesses", c["distinct_witnesses"]))
    print("     %-34s %8d" % ("distinct witnesses incl. Desunt",
                              c["distinct_witnesses_incl_missing"]))
    if "chars_raw" in c:
        print("     %-34s %8d" % ("characters, raw", c["chars_raw"]))
        print("     %-34s %8d" % ("characters, normalised", c["chars_raw_normalised"]))
    print("     %-34s %8d" % ("characters in scored text fields", c["chars_text_fields"]))
    print("     %-34s %8d / %d" % ("versions / citations", c["versions"], c["citations"]))
    print("     variant types: %s"
          % ", ".join("%s=%d" % kv for kv in
                      sorted(c["variant_types"].items(), key=lambda kv: -kv[1])))


# ==========================================================================
# alignment
# ==========================================================================

def str_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - edit_distance(a, b) / max(len(a), len(b))


def multiset_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    ca, cb = Counter(a), Counter(b)
    if not ca and not cb:
        return 1.0
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 1.0


W_TXT, W_WIT, W_TYPE = 0.33, 0, 0.33


def entry_sim(g: dict, p: dict) -> float:
    """sigma(gamma, pi) of the paper. Used only to decide *which entries are the
    same entry*, never to award partial credit."""
    s_text = str_sim(entry_text_blob(g), entry_text_blob(p))
    s_wit = multiset_jaccard(entry_witnesses(g), entry_witnesses(p))
    s_type = 1.0 if entry_type(g) == entry_type(p) else 0.0
    return W_TXT * s_text + W_WIT * s_wit + W_TYPE * s_type


def sim_matrix(gs: List[dict], ps: List[dict], floor: float) -> List[List[float]]:
    return [[(lambda s: s if s >= floor else -1.0)(entry_sim(g, p)) for p in ps]
            for g in gs]


def align_entries_nw(gs: List[dict], ps: List[dict], gap: float = 0.45,
                     floor: float = 0.20) -> List[Tuple[Optional[int], Optional[int]]]:
    """Needleman-Wunsch over entry lists; order-preserving, no index assumption.

    gap   : cost of leaving an entry unmatched
    floor : below this similarity two entries are never called the same entry
    """
    n, m = len(gs), len(ps)
    sim = sim_matrix(gs, ps, floor)
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        best[i][0] = best[i - 1][0] - gap
    for j in range(1, m + 1):
        best[0][j] = best[0][j - 1] - gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = best[i - 1][j - 1] + (sim[i - 1][j - 1] if sim[i - 1][j - 1] >= 0 else -9e9)
            best[i][j] = max(diag, best[i - 1][j] - gap, best[i][j - 1] - gap)
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and sim[i - 1][j - 1] >= 0 and \
                abs(best[i][j] - (best[i - 1][j - 1] + sim[i - 1][j - 1])) < 1e-9:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(best[i][j] - (best[i - 1][j] - gap)) < 1e-9:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return list(reversed(pairs))


def align_entries_greedy(gs: List[dict], ps: List[dict], gap: float = 0.45,
                         floor: float = 0.20) -> List[Tuple[Optional[int], Optional[int]]]:
    """Order-free maximum-similarity matching.

    The paper says units inside a verse cannot be considered ordered; NW is
    order-preserving and therefore cannot match a transposed pair. This greedy
    mutual-best assignment is the order-free counterpart, and the two are
    compared by --assign-compare so the paper can state that the choice is
    immaterial (or by how much it is not).
    """
    cands = []
    for i, g in enumerate(gs):
        for j, p in enumerate(ps):
            s = entry_sim(g, p)
            if s >= floor:
                cands.append((-s, i, j))
    cands.sort()
    used_g: set = set()
    used_p: set = set()
    matched: Dict[int, int] = {}
    for _s, i, j in cands:
        if i in used_g or j in used_p:
            continue
        matched[i] = j
        used_g.add(i)
        used_p.add(j)
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    for i in range(len(gs)):
        pairs.append((i, matched.get(i)))
    for j in range(len(ps)):
        if j not in used_p:
            pairs.append((None, j))
    return pairs


def align_entries(gs, ps, gap=0.45, floor=0.20, mode="nw"):
    if mode == "greedy":
        return align_entries_greedy(gs, ps, gap=gap, floor=floor)
    return align_entries_nw(gs, ps, gap=gap, floor=floor)


# ==========================================================================
# per-verse accumulator
# ==========================================================================

VARIANT_TYPES_PAPER = ("substitution", "omission", "transposition", "addition")


class Rec:
    """Counters for one verse. Aggregation = summing these, which is what makes
    the bootstrap over verses possible."""

    SET_CATS = ("witnesses", "missing", "versions", "citations")
    CLS_CATS = ("variant_type", "in_text_presence", "register")

    def __init__(self, key: str = ""):
        self.key = key
        self.tp = Counter()
        self.fp = Counter()
        self.fn = Counter()
        self.dist = Counter()      # CER numerator, per text field
        self.length = Counter()    # CER denominator (union length)
        self.exact_ok = Counter()
        self.exact_n = Counter()
        self.cls_ok = Counter()
        self.cls_n = Counter()
        self.struct = Counter()    # matched / dropped / spurious entries, verses
        self.ent_tp = Counter()    # entry detection, keyed by gold variant type
        self.ent_fp = Counter()
        self.ent_fn = Counter()
        self.taxo = Counter()      # siglum error taxonomy
        self.confusion = Counter() # (gt_type, pred_type)
        self.glyph = Counter()     # (gt_base, pred_base) confusions
        self.agr_gt = Counter()    # (w1, w2) -> co-occurrences, ground truth
        self.agr_pr = Counter()    # (w1, w2) -> co-occurrences, pipeline
        self.att_gt = Counter()    # w -> segments attesting w, ground truth
        self.att_pr = Counter()    # w -> segments attesting w, pipeline
        # one row per gold entry: (gold list length, sigla missed, sigla
        # invented, whole-list-exact flag, entry-aligned flag). Kept raw so the
        # length analysis can be re-binned and bootstrapped without recomputing.
        self.points: List[Tuple[int, int, int, int, int]] = []
        # (verse, kind, gold_type, pred_type, gold_text, pred_text)
        # kind: "dropped" | "spurious" | "type"
        self.fails: List[Tuple[str, str, str, str, str, str]] = []
        # every text field that did not come back exactly, for error analysis:
        # (verse, field, kind, gold, pred, edit_distance, union_length)
        # kind: "differs" (aligned pair) | "dropped" | "spurious"
        self.texterr: List[Tuple[str, str, str, str, str, int, int]] = []

    _COUNTERS = ("tp", "fp", "fn", "dist", "length", "exact_ok", "exact_n",
                 "cls_ok", "cls_n", "struct", "ent_tp", "ent_fp", "ent_fn",
                 "taxo", "confusion", "glyph", "agr_gt", "agr_pr",
                 "att_gt", "att_pr")

    def merge(self, o: "Rec") -> None:
        for name in self._COUNTERS:
            getattr(self, name).update(getattr(o, name))
        self.points.extend(o.points)
        self.fails.extend(o.fails)
        self.texterr.extend(o.texterr)


def score_sets(rec: Rec, cat: str, gold: Sequence[str], pred: Sequence[str],
               errors: list, key: str, fuzzy: float = 1.0) -> None:
    """Multiset P/R. With fuzzy<1, near-identical strings (citations) still count
    as matched but their residual distance is recorded as a CER."""
    cg, cp = Counter(gold), Counter(pred)
    inter = cg & cp
    rec.tp[cat] += sum(inter.values())
    left_g = list((cg - inter).elements())
    left_p = list((cp - inter).elements())
    if fuzzy < 1.0:
        for g in list(left_g):
            best, bs = None, 0.0
            for p in left_p:
                s = str_sim(g, p)
                if s > bs:
                    best, bs = p, s
            if best is not None and bs >= fuzzy:
                rec.tp[cat] += 1
                rec.dist[cat] += edit_distance(g, best)
                rec.length[cat] += max(len(g), len(best))
                left_g.remove(g)
                left_p.remove(best)
                errors.append((key, cat, "near-match", g, best))
    rec.fn[cat] += len(left_g)
    rec.fp[cat] += len(left_p)
    for g in left_g:
        errors.append((key, cat, "dropped", g, ""))
    for p in left_p:
        errors.append((key, cat, "spurious", "", p))


def score_sigla(rec: Rec, cat: str, gold: Sequence[str], pred: Sequence[str],
                errors: list, key: str, group_stat: bool = True) -> None:
    """Witness sigla: multiset P/R + a taxonomy of what went wrong in the glyph."""
    cg, cp = Counter(gold), Counter(pred)
    inter = cg & cp
    if group_stat:
        # all-or-nothing verdict on the segment's whole witness list: every
        # siglum present, none invented, every hand mark right, or it is a miss
        rec.exact_n["witness_set"] += 1
        rec.exact_n["witness_set_aligned"] += 1
        if cg == cp:
            rec.exact_ok["witness_set"] += 1
            rec.exact_ok["witness_set_aligned"] += 1
        rec.points.append((sum(cg.values()),
                           sum(cg.values()) - sum(inter.values()),
                           sum(cp.values()) - sum(inter.values()),
                           1 if cg == cp else 0, 1))
    rec.tp[cat] += sum(inter.values())
    rec.taxo["exact"] += sum(inter.values())
    left_g = list((cg - inter).elements())
    left_p = list((cp - inter).elements())
    # pair up leftovers by closest glyph so that "same witness, wrong hand" is
    # not counted as one deletion + one insertion of unrelated symbols
    for g in list(left_g):
        gb, gl, gm = split_siglum(g)
        cand, score = None, -1
        for cur in left_p:
            pb, pl, pm = split_siglum(cur)
            # two sigla are "the same witness misread" only if the base glyph
            # agrees (then the superscript or the hand mark is wrong) or if the
            # superscript agrees and is non-empty (then the base glyph is a
            # confusion such as Lambda -> A). Anything else is a genuine
            # deletion plus a genuine hallucination, not one substitution.
            if pb == gb:
                sc = 3 + (1 if pl == gl else 0) + (1 if pm == gm else 0)
            elif gl and gl == pl and gm == pm:
                sc = 2
            else:
                continue
            if sc > score:
                cand, score = cur, sc
        if cand is None:
            continue
        pb, pl, pm = split_siglum(cand)
        if gb != pb:
            kind = "base-glyph"
            rec.glyph[(gb, pb)] += 1
        elif gl != pl:
            kind = "superscript-letter"
        else:
            kind = "hand-mark"
        rec.taxo[kind] += 1
        # a wrong siglum is strictly one miss and one false alarm; the taxonomy
        # above says what kind of wrong it was, it does not grant partial credit
        rec.fn[cat] += 1
        rec.fp[cat] += 1
        errors.append((key, cat, kind, g, cand))
        left_g.remove(g)
        left_p.remove(cand)
    rec.fn[cat] += len(left_g)
    rec.fp[cat] += len(left_p)
    rec.taxo["dropped"] += len(left_g)
    rec.taxo["spurious"] += len(left_p)
    for g in left_g:
        errors.append((key, cat, "dropped", g, ""))
    for p in left_p:
        errors.append((key, cat, "spurious", "", p))


def score_text(rec: Rec, field: str, gold: str, pred: str,
               errors: list, key: str) -> None:
    if not gold and not pred:
        return
    d = edit_distance(gold, pred)
    rec.dist[field] += d
    # union length, not reference length: an invented field has len(gold)==0
    # and would otherwise add numerator against a zero denominator. Bounded
    # by 1 per item, so CER and exact-match stay on the same scale.
    n = max(len(gold), len(pred))
    rec.length[field] += n
    rec.exact_n[field] += 1
    # score_text is reached only from the matched branch, so the `aligned:`
    # counters are by construction the aligned-entries-only population
    rec.dist["aligned:" + field] += d
    rec.length["aligned:" + field] += n
    rec.exact_n["aligned:" + field] += 1
    if d == 0:
        rec.exact_ok[field] += 1
        rec.exact_ok["aligned:" + field] += 1
    else:
        rec.texterr.append((key, field, "differs", gold, pred,
                            d, max(len(gold), len(pred))))
        errors.append((key, "text:" + field, "differs", gold, pred))


def score_class(rec: Rec, cat: str, gold: str, pred: str,
                errors: list, key: str) -> None:
    rec.cls_n[cat] += 1
    if gold == pred:
        rec.cls_ok[cat] += 1
    else:
        errors.append((key, cat, "differs", gold, pred))
    if cat == "variant_type":
        rec.confusion[(gold or "-", pred or "-")] += 1


def add_attestations(counter: Counter, groups: Iterable[Sequence[str]]) -> None:
    """How many segments each witness is cited in — the denominator of every
    normalised co-occurrence coefficient."""
    for g in groups:
        for w in set(g):
            counter[w] += 1


def add_agreements(counter: Counter, groups: Iterable[Sequence[str]]) -> None:
    """Quentin's grid table: +1 for every unordered pair of witnesses that
    appear together in the same segment of the apparatus."""
    for g in groups:
        ws = sorted(set(g))
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                counter[(ws[i], ws[j])] += 1


# ==========================================================================
# main comparison
# ==========================================================================

def compare(gt: List[dict], pred: List[dict], count_missing: bool = True,
            gap: float = 0.45, floor: float = 0.20, mode: str = "nw"
            ) -> Tuple[List[Rec], list, list]:
    errors: list = []
    issues: list = []
    gt_by = defaultdict(list)
    pr_by = defaultdict(list)
    for v in gt:
        gt_by[verse_key(v)].append(v)
    for v in pred:
        pr_by[verse_key(v)].append(v)

    recs: List[Rec] = []
    for key in sorted(set(gt_by) | set(pr_by), key=_verse_sort):
        rec = Rec(key)
        g_entries = [e for v in gt_by.get(key, []) for e in (v.get("apparatus") or [])]
        p_entries = [e for v in pr_by.get(key, []) for e in (v.get("apparatus") or [])]

        # --- [4] agreement matrices: computed from the data as it stands, not
        # from the alignment, because that is exactly the downstream artefact
        for e in g_entries:
            add_agreements(rec.agr_gt, entry_witness_groups(e))
            add_attestations(rec.att_gt, entry_witness_groups(e))
        for e in p_entries:
            add_agreements(rec.agr_pr, entry_witness_groups(e))
            add_attestations(rec.att_pr, entry_witness_groups(e))

        rec.struct["verses_gt"] += bool(gt_by.get(key))
        rec.struct["verses_pred"] += bool(pr_by.get(key))
        if not gt_by.get(key):
            rec.struct["verses_spurious"] += 1
            errors.append((key, "structure", "spurious verse", "", str(len(p_entries))))
        elif not pr_by.get(key):
            rec.struct["verses_dropped"] += 1
            errors.append((key, "structure", "dropped verse", str(len(g_entries)), ""))

        # Desunt: identical for every entry of a page, so deduplicate per verse
        if count_missing:
            gm = sorted(set(w for e in g_entries for w in entry_missing(e)))
            pm = sorted(set(w for e in p_entries for w in entry_missing(e)))
            score_sigla(rec, "missing", gm, pm, errors, key, group_stat=False)

        pairs = align_entries(g_entries, p_entries, gap=gap, floor=floor, mode=mode)
        issues.extend(diagnose(g_entries, p_entries, pairs, key, floor))
        for gi, pi in pairs:
            if gi is None:
                # hallucinated entry: whole-field error. It enters both sides
                # of the ratio and the exact-match denominator, exactly like a
                # dropped entry, so neither metric is computed over a
                # population the other does not see.
                rec.struct["entries_spurious"] += 1
                rec.ent_fp[entry_type(p_entries[pi])] += 1
                rec.ent_fp["__all__"] += 1
                rec.fails.append((key, "spurious", "-",
                                  entry_type(p_entries[pi]), "",
                                  entry_text_blob(p_entries[pi])[:70]))
                errors.append((key, "structure", "spurious entry", "",
                               entry_text_blob(p_entries[pi])[:60]))
                ptx = entry_texts(p_entries[pi])
                for f in TEXT_ORDER:
                    if ptx[f]:
                        rec.dist[f] += len(ptx[f])
                        rec.length[f] += len(ptx[f])
                        rec.exact_n[f] += 1
                        rec.texterr.append((key, f, "spurious", "", ptx[f],
                                            len(ptx[f]), len(ptx[f])))
                for w in entry_witnesses(p_entries[pi]):
                    rec.fp["witnesses"] += 1
                    rec.fp["witnesses_unaligned"] += 1
                    rec.taxo["spurious-unaligned-entry"] += 1
                    errors.append((key, "witnesses", "spurious-unaligned-entry",
                                   "", w))
                continue
            if pi is None:
                # lost entry: its whole text is a deletion, so the category does
                # not look better when the OCR skips a unit than when it garbles it
                rec.struct["entries_dropped"] += 1
                rec.ent_fn[entry_type(g_entries[gi])] += 1
                rec.ent_fn["__all__"] += 1
                rec.fails.append((key, "dropped",
                                  entry_type(g_entries[gi]), "-",
                                  entry_text_blob(g_entries[gi])[:70], ""))
                errors.append((key, "structure", "dropped entry",
                               entry_text_blob(g_entries[gi])[:60], ""))
                for _grp in entry_witness_groups(g_entries[gi]) or [[]]:
                    rec.exact_n["witness_set"] += 1
                    rec.points.append((len(_grp), len(_grp), 0, 0, 0))
                gtx = entry_texts(g_entries[gi])
                for f in TEXT_ORDER:
                    if gtx[f]:
                        rec.dist[f] += len(gtx[f])
                        rec.length[f] += len(gtx[f])
                        rec.exact_n[f] += 1
                        rec.texterr.append((key, f, "dropped", gtx[f], "",
                                            len(gtx[f]), len(gtx[f])))
                for w in entry_witnesses(g_entries[gi]):
                    rec.fn["witnesses"] += 1
                    rec.fn["witnesses_unaligned"] += 1
                    rec.taxo["dropped-unaligned-entry"] += 1
                    errors.append((key, "witnesses", "dropped-unaligned-entry",
                                   w, ""))
                continue
            g, p = g_entries[gi], p_entries[pi]
            rec.struct["entries_matched"] += 1
            rec.ent_tp[entry_type(g)] += 1
            rec.ent_tp["__all__"] += 1
            gtx, ptx = entry_texts(g), entry_texts(p)
            for f in TEXT_ORDER:
                score_text(rec, f, gtx[f], ptx[f], errors, key)
            # segment-level witness groups, so that sigla_group_exact_match is
            # measured on segments as the paper says
            ggroups, pgroups = entry_witness_groups(g), entry_witness_groups(p)
            for si in range(max(len(ggroups), len(pgroups))):
                gg = ggroups[si] if si < len(ggroups) else []
                pg = pgroups[si] if si < len(pgroups) else []
                score_sigla(rec, "witnesses", gg, pg, errors, key)
            score_class(rec, "variant_type", entry_type(g), entry_type(p), errors, key)
            if entry_type(g) != entry_type(p):
                rec.fails.append((key, "type", entry_type(g), entry_type(p),
                                  entry_text_blob(g)[:70],
                                  entry_text_blob(p)[:70]))
            score_class(rec, "in_text_presence", str(bool(gtx["in_text"])),
                        str(bool(ptx["in_text"])), errors, key)
            score_class(rec, "register", canon_text(g.get("register")),
                        canon_text(p.get("register")), errors, key)
            score_sets(rec, "versions", entry_versions(g), entry_versions(p),
                       errors, key)
            score_sets(rec, "citations", entry_citations(g), entry_citations(p),
                       errors, key, fuzzy=0.85)
        recs.append(rec)
    return recs, errors, issues


NEAR = 0.15   # below this, two entries have nothing to do with each other


def diagnose(g_entries: List[dict], p_entries: List[dict],
             pairs: List[Tuple[Optional[int], Optional[int]]],
             key: str, floor: float) -> List[dict]:
    """Explain every unaligned entry instead of just counting it.

      split      the best partner is already taken -> one gold unit was read as
                 two (a `;` or a line break taken for a `|`)
      merge      symmetrical case: two gold units collapsed into one
      near-miss  the best partner is itself unaligned and the pair sits just
                 below --sim-floor -> the unit exists on both sides but the OCR
                 garbled it enough that the aligner refused to call it the same
      crossed    the best partner is free and above the floor, yet the aligner
                 did not take it -> an order violation NW cannot represent
      orphan     nothing resembling it on the other side -> genuine loss or
                 genuine hallucination of content
    """
    matched_g = {gi for gi, pi in pairs if gi is not None and pi is not None}
    matched_p = {pi for gi, pi in pairs if gi is not None and pi is not None}
    issues: List[dict] = []
    for gi, pi in pairs:
        if (gi is None) == (pi is None):
            continue
        if pi is None:                      # gold entry with no partner
            src, others, taken = g_entries[gi], p_entries, matched_p
        else:                               # pipeline entry with no partner
            src, others, taken = p_entries[pi], g_entries, matched_g
        best, bs = None, 0.0
        for j, o in enumerate(others):
            sim = entry_sim(src, o) if pi is None else entry_sim(o, src)
            if sim > bs:
                best, bs = j, sim
        contained = 0.0
        if best is not None:
            src_toks = set(entry_text_blob(src).split())
            oth_toks = set(entry_text_blob(others[best]).split())
            if src_toks:
                contained = len(src_toks & oth_toks) / len(src_toks)
        if best is None or bs < NEAR:
            kind = "orphan"
        elif best in taken and contained >= 0.6:
            # its closest counterpart is already spoken for: gold-only means the
            # unit was folded into a neighbour, pipeline-only means a gold unit
            # was cut in two
            kind = "merge" if pi is None else "split"
        elif best in taken:
            # resembles a kept unit but its wording is not inside it: nothing
            # was folded anywhere, the unit simply is not on the other side
            kind = "orphan"
        elif bs < floor:
            # free partner, but too garbled for the aligner to accept it
            kind = "near-miss"
        else:
            # free partner above the floor that the aligner still refused:
            # only an order-preserving aligner does that
            kind = "crossed"
        issues.append({
            "verse": key,
            "side": "gt-only" if pi is None else "pred-only",
            "class": kind,
            "sim": bs,
            "variant_type": entry_type(src),
            "n_wit": len(entry_witnesses(src)),
            "text": entry_text_blob(src)[:90],
            "best_text": (entry_text_blob(others[best])[:90] if best is not None else ""),
        })

    return issues


def _verse_sort(k: str):
    ch, _, vs = k.partition(":")
    def num(x):
        m = re.match(r"\d+", x)
        return (int(m.group()) if m else 0, x)
    return (num(ch), (0, "") if vs == "tit" else (1, ""), num(vs))


# ==========================================================================
# [4] witness agreements — Quentin's grid tables and knn_overlap
# ==========================================================================

def pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def degrees(agr: Counter, universe: Sequence[str]) -> Dict[str, int]:
    d = {w: 0 for w in universe}
    for (a, b), n in agr.items():
        if a in d:
            d[a] += n
        if b in d:
            d[b] += n
    return d


def knn_sets(agr: Counter, w: str, universe: Sequence[str], k: int
             ) -> Tuple[set, set, Optional[int], int]:
    """The top-k neighbourhood of w, with its ambiguity made explicit.

    Returns (core, tie, boundary_score, slots):
      core     witnesses strictly better than the k-th best score — in every
               valid top-k selection
      tie      witnesses sitting exactly on the k-th score — any `slots` of
               them complete a valid selection, and the choice is arbitrary
      slots    k - |core|
    With no ties, tie is empty and the neighbourhood is unique.
    """
    scored = [(agr.get(pair_key(w, o), 0), o) for o in universe if o != w]
    if not scored:
        return set(), set(), None, 0
    scored.sort(key=lambda t: -t[0])
    if len(scored) <= k:
        return {o for _s, o in scored}, set(), scored[-1][0], 0
    thr = scored[k - 1][0]
    core = {o for s, o in scored if s > thr}
    tie = {o for s, o in scored if s == thr}
    return core, tie, thr, k - len(core)


def overlap_variants(ac: set, at: set, sa: int,
                     bc: set, bt: set, sb: int, k: int
                     ) -> Tuple[float, float, float]:
    """|kNN_GT ∩ kNN_OCR| under the three readings of an ambiguous tie.

    pessimistic  the worst |S ∩ T| over all valid selections S, T
    expected     E|S ∩ T| when each side breaks its tie uniformly at random
                 (exact, not sampled: E = sum_j P(j in S) P(j in T))
    optimistic   the best |S ∩ T| over all valid selections

    Without ties the three coincide and equal the plain overlap, so switching
    the convention cannot change a number that was never ambiguous.
    """
    base = len(ac & bc)

    # --- optimistic: spend the free slots on witnesses the other side already
    #     has for certain, then on witnesses tied on both sides
    x1, x2 = at & bc, ac & bt
    x3 = at & bt
    t1 = min(len(x1), sa)
    t2 = min(len(x2), sb)
    t3 = min(len(x3), sa - t1, sb - t2)
    hi = min(k, base + t1 + t2 + t3)

    # --- pessimistic: spend them on witnesses the other side cannot have,
    #     then on the doubly-tied ones (which collide only when forced)
    q = at & bt
    ra = min(sa, len(at - bc - bt))
    qa = min(sa - ra, len(q))
    pa = sa - ra - qa
    rb = min(sb, len(bt - ac - at))
    qb = min(sb - rb, len(q))
    pb = sb - rb - qb
    lo = min(k, base + pa + pb + max(0, qa + qb - len(q)))

    # --- expected
    p_at = (sa / len(at)) if at else 0.0
    p_bt = (sb / len(bt)) if bt else 0.0
    exp = 0.0
    for j in (ac | at) & (bc | bt):
        exp += (1.0 if j in ac else p_at) * (1.0 if j in bc else p_bt)
    return lo, min(exp, float(k)), hi


TIE_MODES = ("pessimistic", "expected", "optimistic")


def row_agreement_correlation(agr_gt: Counter, agr_pr: Counter,
                              universe: Sequence[str], min_degree: int = 1
                              ) -> float:
    """Mean per-witness Spearman between the GT and OCR rows of the grid table.

    Tie-free by construction (ranks average over ties) and it uses the whole
    row rather than an arbitrary cut at k, so it is the figure to quote when
    the k-NN neighbourhood is not identified by the data.
    """
    deg = degrees(agr_gt, universe)
    vals = []
    for w in universe:
        if deg[w] < min_degree:
            continue
        xs = [agr_gt.get(pair_key(w, o), 0) for o in universe if o != w]
        ys = [agr_pr.get(pair_key(w, o), 0) for o in universe if o != w]
        r = _spearman(xs, ys)
        if r == r:
            vals.append(r)
    return sum(vals) / len(vals) if vals else float("nan")


def knn_overlap(agr_gt: Counter, agr_pr: Counter, universe: Sequence[str],
                k: int = 3, min_degree: int = 1, min_positive: int = 0,
                ties: str = "expected") -> Dict[str, Any]:
    """Neighbourhood preservation ratio, averaged over witnesses.

    Witnesses with fewer than `min_degree` agreements in the ground truth are
    excluded: their neighbourhood is arbitrary and would only add noise.
    `min_positive` additionally requires that many *strictly positive* GT
    neighbours, i.e. that the top-k neighbourhood is defined by the data rather
    than by a tie at zero — set it to k for the strict reading.
    """
    deg = degrees(agr_gt, universe)
    pos = {w: sum(1 for o in universe
                  if o != w and agr_gt.get(pair_key(w, o), 0) > 0)
           for w in universe}
    used = [w for w in universe
            if deg[w] >= min_degree and pos[w] >= min_positive]
    per: Dict[str, Dict[str, float]] = {}
    n_tied = n_zero_boundary = 0
    tie_widths: List[int] = []
    for w in used:
        ac, at, thr_g, sa = knn_sets(agr_gt, w, universe, k)
        bc, bt, _thr_p, sb = knn_sets(agr_pr, w, universe, k)
        if at or bt:
            n_tied += 1
            tie_widths.append(max(len(at), len(bt)))
        if not thr_g:
            n_zero_boundary += 1
        lo, exp, hi = overlap_variants(ac, at, sa, bc, bt, sb, k)
        # tie-inclusive neighbourhoods: no arbitrary choice is made at all,
        # at the price of comparing sets that may be larger than k
        ea, eb = ac | at, bc | bt
        jac = len(ea & eb) / len(ea | eb) if (ea | eb) else float("nan")
        per[w] = {"pessimistic": lo / k, "expected": exp / k, "optimistic": hi / k,
                  "jaccard": jac, "gt_boundary_score": thr_g,
                  "gt_tie_width": len(at)}

    def avg(mode: str) -> float:
        return (sum(v[mode] for v in per.values()) / len(per)
                if per else float("nan"))

    vals = {mode: avg(mode) for mode in TIE_MODES}
    return {"k": k, "N": len(per), "excluded": len(universe) - len(used),
            "knn_jaccard": avg("jaccard"),
            "knn_overlap": vals[ties], "ties_mode": ties,
            "knn_overlap_pessimistic": vals["pessimistic"],
            "knn_overlap_expected": vals["expected"],
            "knn_overlap_optimistic": vals["optimistic"],
            "witnesses_with_ties": n_tied,
            "witnesses_zero_boundary": n_zero_boundary,
            "median_tie_width": (sorted(tie_widths)[len(tie_widths) // 2]
                                 if tie_widths else 0),
            "per_witness": per}


def matrix_stats(agr_gt: Counter, agr_pr: Counter,
                 universe: Sequence[str]) -> Dict[str, Any]:
    keys = [pair_key(universe[i], universe[j])
            for i in range(len(universe)) for j in range(i + 1, len(universe))]
    xs = [agr_gt.get(kk, 0) for kk in keys]
    ys = [agr_pr.get(kk, 0) for kk in keys]
    tot_gt, tot_pr = sum(xs), sum(ys)
    l1 = sum(abs(a - b) for a, b in zip(xs, ys))

    # Most cells of a large grid table are structural zeros: witnesses that
    # never co-occur on either side. They agree trivially and inflate any
    # correlation, so report the same figures over the attested support too.
    #   union : pairs attested in at least one of the two matrices
    #   gt    : pairs attested in the ground truth (the recall-side reading)
    # Frobenius over the upper triangle. Raw ||.||_F is scale-dependent and the
    # OCR matrix is systematically smaller (every dropped siglum destroys
    # agreements), so it is decomposed: the optimal rescaling c minimises
    # ||M_gt - c*M_pred||_F, and what survives it is structural rather than a
    # uniform shrinkage.
    fro_gt = sum(a * a for a in xs) ** 0.5
    fro_d = sum((a - b) ** 2 for a, b in zip(xs, ys)) ** 0.5
    den = sum(b * b for b in ys)
    c_opt = (sum(a * b for a, b in zip(xs, ys)) / den) if den else float("nan")
    fro_scaled = (sum((a - c_opt * b) ** 2 for a, b in zip(xs, ys)) ** 0.5
                  if den else float("nan"))

    u = [(a, b) for a, b in zip(xs, ys) if a or b]
    g = [(a, b) for a, b in zip(xs, ys) if a]
    return {
        "pairs": len(keys),
        "agreements_gt": tot_gt,
        "agreements_pred": tot_pr,
        "relative_L1": l1 / tot_gt if tot_gt else float("nan"),
        "frobenius": fro_d,
        "frobenius_relative": fro_d / fro_gt if fro_gt else float("nan"),
        "frobenius_optimal_scale": c_opt,
        "frobenius_relative_rescaled": (fro_scaled / fro_gt if fro_gt
                                        else float("nan")),
        "spearman": _spearman(xs, ys),
        "pearson": _pearson(xs, ys),
        "nonzero_gt": sum(1 for x in xs if x),
        "nonzero_pred": sum(1 for y in ys if y),
        "pairs_nonzero_union": len(u),
        "spearman_nonzero_union": _spearman([a for a, _ in u], [b for _, b in u]),
        "pearson_nonzero_union": _pearson([a for a, _ in u], [b for _, b in u]),
        "pairs_nonzero_gt": len(g),
        "spearman_nonzero_gt": _spearman([a for a, _ in g], [b for _, b in g]),
        "pearson_nonzero_gt": _pearson([a for a, _ in g], [b for _, b in g]),
        "structural_zeros": len(keys) - len(u),
    }


def dump_matrix(path: str, agr: Counter, universe: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t" + "\t".join(universe) + "\n")
        for a in universe:
            row = [str(agr.get(pair_key(a, b), 0)) if a != b else "-"
                   for b in universe]
            fh.write(a + "\t" + "\t".join(row) + "\n")


# ==========================================================================
# scatter plot: gold vs predicted shared-reading count, one point per pair
# ==========================================================================

def scatter_data(agr_gt: Counter, agr_pr: Counter, universe: Sequence[str],
                 drop_zeros: bool = True, highlight: Optional[str] = None
                 ) -> List[Tuple[str, str, int, int, bool]]:
    r"""One row per witness pair: (w1, w2, gold, pred, highlighted).

    All witnesses are treated as independent. Expanding a group siglum
    (\Phi^{RAZG} -> \Phi^R \Phi^A \Phi^Z \Phi^G) recovers what the apparatus
    states -- that those manuscripts attest the reading -- so the resulting
    agreements are data, not an artefact of the expansion, and sharing a base
    glyph is not on its own evidence of a family.

    `highlight` is an optional regex: pairs whose BOTH sigla match it are drawn
    in a second colour. Nothing is inferred; the subset is whatever the caller
    asks for.
    """
    rx = re.compile(highlight) if highlight else None
    rows = []
    for i in range(len(universe)):
        for j in range(i + 1, len(universe)):
            a, b = universe[i], universe[j]
            g = agr_gt.get(pair_key(a, b), 0)
            p_ = agr_pr.get(pair_key(a, b), 0)
            if drop_zeros and not g and not p_:
                continue
            hit = bool(rx and rx.search(a) and rx.search(b))
            rows.append((a, b, g, p_, hit))
    return rows


def write_scatter(path: str, rows: Sequence[Tuple[str, str, int, int, bool]],
                  stats: Dict[str, Any], logscale: bool = False) -> List[str]:
    """Write the plot data as TSV and, if matplotlib is present, the figure.

    The TSV is written unconditionally: pgfplots or R may be preferable for a
    camera-ready figure, and the numbers should not be locked inside a PNG.
    """
    made = []
    tsv = path + ".tsv"
    with open(tsv, "w", encoding="utf-8") as fh:
        fh.write("witness_1\twitness_2\tgold\tpred\thighlighted\n")
        for a, b, g, p_, fam in rows:
            fh.write("%s\t%s\t%d\t%d\t%d\n" % (a, b, g, p_, int(fam)))
    made.append(tsv)
    if not rows:
        return made

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: TSV written, figure skipped)")
        return made

    hi_rows = [r for r in rows if r[4]]
    oth_rows = [r for r in rows if not r[4]]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    hi = max([r[2] for r in rows] + [r[3] for r in rows] + [1])
    ax.plot([0, hi], [0, hi], lw=0.8, ls="--", c="0.4", zorder=1, label="$y = x$")
    c = stats.get("frobenius_optimal_scale")
    if c and c == c and abs(c - 1) > 0.02:
        ax.plot([0, hi], [0, hi / c], lw=0.8, ls=":", c="0.55", zorder=1,
                label="best uniform scale ($c=%.3f$)" % c)
    groups = [(oth_rows, "#2b6cb0", "witness pairs" if not hi_rows else "other pairs")]
    if hi_rows:
        groups.append((hi_rows, "#c05621", "highlighted pairs"))
    for subset, colour, lab in groups:
        if not subset:
            continue
        mult = Counter((r[2], r[3]) for r in subset)
        xs = [k[0] for k in mult]
        ys = [k[1] for k in mult]
        sz = [8 + 5 * (v ** 0.5) for v in mult.values()]
        ax.scatter(xs, ys, s=sz, alpha=0.45, edgecolors="none", c=colour,
                   zorder=3, label="%s" % (lab))
    if logscale:
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Pipeline output")
    sub = []
    if stats.get("spearman_nonzero_union") == stats.get("spearman_nonzero_union"):
        sub.append(r"$\rho=%.3f$" % stats["spearman_nonzero_union"])
        sub.append("$r=%.3f$" % stats["pearson_nonzero_union"])
    ax.set_title("Agreement counts \n" + "   ".join(sub), fontsize=10)
    ax.legend(fontsize=7, loc="upper left", frameon=False, markerscale=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig("%s.%s" % (path, ext), dpi=200)
        made.append("%s.%s" % (path, ext))
    plt.close(fig)
    return made


# ==========================================================================
# [5] clustering agreement — CAH on the two grid tables
#
# The correlation metrics compare cells; this compares the *groupings* a
# stemmatologist would actually read off the grid table, which is the task the
# dataset exists for. Everything here is deterministic: same input, same
# dendrogram, same partition, no seed.
#
# No scipy: the linkage, the cuts and the indices are ~150 lines and keeping
# them in the file makes the paper's numbers reproducible from this script
# alone.
# ==========================================================================

DIST_KINDS = ("jaccard", "ochiai", "raw")


def distance_matrix(agr: Counter, att: Counter, universe: Sequence[str],
                    kind: str = "jaccard") -> List[List[float]]:
    """Co-occurrence counts -> a distance in [0, 1].

    Raw counts must not be clustered directly: the OCR matrix is systematically
    smaller than the ground truth one (every dropped siglum removes agreements),
    so an unnormalised distance would separate the two partitions for a reason
    that has nothing to do with which witnesses go together.

      jaccard  1 - a_ij / (n_i + n_j - a_ij)   segments where both are cited,
                                               over segments where either is
      ochiai   1 - a_ij / sqrt(n_i * n_j)      cosine on the incidence vectors,
                                               gentler on rare witnesses
      raw      1 - a_ij / max(a)               unnormalised, for comparison only
    """
    n = len(universe)
    mx = max(agr.values()) if agr else 1
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a = agr.get(pair_key(universe[i], universe[j]), 0)
            ni, nj = att.get(universe[i], 0), att.get(universe[j], 0)
            if kind == "jaccard":
                den = ni + nj - a
                sim = a / den if den else 0.0
            elif kind == "ochiai":
                den = (ni * nj) ** 0.5
                sim = a / den if den else 0.0
            else:
                sim = a / mx if mx else 0.0
            d = 1.0 - sim
            D[i][j] = D[j][i] = d
    return D


LINKAGES = ("average", "complete", "single", "ward")


def hac(D: List[List[float]], linkage: str = "average"
        ) -> Tuple[List[Tuple[int, int, float, int]], int]:
    """Agglomerative clustering by Lance-Williams update.

    Returns (merges, n_ambiguous). Each merge is (cluster_a, cluster_b,
    distance, size), with cluster ids 0..n-1 for singletons and n+step for
    internal nodes, as in a scipy linkage matrix.

    Ties are the same nuisance as in the k-NN section: with small integer counts
    several pairs routinely sit at the minimum distance. The tie is broken by
    the lowest index pair — reproducible, but arbitrary — and the number of
    ambiguous merges is returned so it can be reported rather than hidden.
    """
    n = len(D)
    d = [row[:] for row in D]
    active = list(range(n))
    size = {i: 1 for i in range(n)}
    node = {i: i for i in range(n)}          # position -> cluster id
    merges: List[Tuple[int, int, float, int]] = []
    ambiguous = 0
    nxt = n
    while len(active) > 1:
        best = None
        best_val = float("inf")
        hits = 0
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                v = d[active[ii]][active[jj]]
                if v < best_val - 1e-12:
                    best_val, best, hits = v, (ii, jj), 1
                elif abs(v - best_val) <= 1e-12:
                    hits += 1
        if hits > 1:
            ambiguous += 1
        ii, jj = best
        pi, pj = active[ii], active[jj]
        ni, nj = size[node[pi]], size[node[pj]]
        merges.append((node[pi], node[pj], best_val, ni + nj))
        for pk in active:
            if pk in (pi, pj):
                continue
            dki, dkj = d[pk][pi], d[pk][pj]
            if linkage == "average":
                v = (ni * dki + nj * dkj) / (ni + nj)
            elif linkage == "complete":
                v = max(dki, dkj)
            elif linkage == "ward":
                # Ward.D2 in distance space. Ward assumes Euclidean geometry,
                # which 1-Jaccard does not satisfy; it is used here the way
                # R's ward.D2 is used on any dissimilarity, because it is the
                # linkage that does not chain. Cross-check with --linkage-
                # robustness before leaning on it.
                nk = size[node[pk]]
                v = (((nk + ni) * dki ** 2 + (nk + nj) * dkj ** 2
                      - nk * best_val ** 2) / (nk + ni + nj)) ** 0.5
            else:
                v = min(dki, dkj)
            d[pk][pi] = d[pi][pk] = v
        node[pi] = nxt
        size[nxt] = ni + nj
        nxt += 1
        active.remove(pj)
    return merges, ambiguous


def cut_tree(merges: Sequence[Tuple[int, int, float, int]], n: int,
             k: int) -> List[int]:
    """Flat partition with k clusters: apply the first n-k merges."""
    parent = list(range(n + len(merges)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for step, (a, b, _dist, _sz) in enumerate(merges[:max(0, n - k)]):
        parent[find(a)] = n + step
        parent[find(b)] = n + step
    roots: Dict[int, int] = {}
    out = []
    for i in range(n):
        r = find(i)
        out.append(roots.setdefault(r, len(roots)))
    return out


def cophenetic(merges: Sequence[Tuple[int, int, float, int]], n: int
               ) -> List[List[float]]:
    """Distance at which each pair of witnesses first ends up in one cluster."""
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    C = [[0.0] * n for _ in range(n)]
    for step, (a, b, dist, _sz) in enumerate(merges):
        ma, mb = members.pop(a), members.pop(b)
        for x in ma:
            for y in mb:
                C[x][y] = C[y][x] = dist
        members[n + step] = ma + mb
    return C


# ---- partition comparison -------------------------------------------------

def _contingency(x: Sequence[int], y: Sequence[int]) -> Counter:
    return Counter(zip(x, y))


def _c2(v: int) -> float:
    return v * (v - 1) / 2.0


def adjusted_rand(x: Sequence[int], y: Sequence[int]) -> float:
    """Chance-corrected agreement between two partitions. 0 = as good as a
    random partition with the same cluster sizes, 1 = identical."""
    n = len(x)
    if n < 2:
        return float("nan")
    cont = _contingency(x, y)
    a = Counter(x)
    b = Counter(y)
    index = sum(_c2(v) for v in cont.values())
    sa = sum(_c2(v) for v in a.values())
    sb = sum(_c2(v) for v in b.values())
    exp = sa * sb / _c2(n) if n > 1 else 0.0
    mx = (sa + sb) / 2.0
    return (index - exp) / (mx - exp) if mx != exp else 1.0


def fowlkes_mallows(x: Sequence[int], y: Sequence[int]) -> float:
    cont = _contingency(x, y)
    tp = sum(_c2(v) for v in cont.values())
    sa = sum(_c2(v) for v in Counter(x).values())
    sb = sum(_c2(v) for v in Counter(y).values())
    return tp / (sa * sb) ** 0.5 if sa and sb else float("nan")


def variation_of_information(x: Sequence[int], y: Sequence[int]) -> float:
    """VI in bits. A true metric on partitions: 0 iff identical, and bounded
    by log2(n), so it says how far apart two groupings are rather than how
    correlated they look."""
    import math
    n = len(x)
    if not n:
        return float("nan")
    cont = _contingency(x, y)
    a, b = Counter(x), Counter(y)
    hx = -sum((v / n) * math.log2(v / n) for v in a.values())
    hy = -sum((v / n) * math.log2(v / n) for v in b.values())
    mi = 0.0
    for (i, j), v in cont.items():
        pij = v / n
        mi += pij * math.log2(pij / ((a[i] / n) * (b[j] / n)))
    return hx + hy - 2 * mi


def cluster_comparison(agr_gt: Counter, att_gt: Counter,
                       agr_pr: Counter, att_pr: Counter,
                       universe: Sequence[str], ks: Sequence[int],
                       dist: str = "jaccard", linkage: str = "average",
                       ) -> Dict[str, Any]:
    """Cluster both grid tables the same way, then compare the partitions."""
    n = len(universe)
    Dg = distance_matrix(agr_gt, att_gt, universe, dist)
    Dp = distance_matrix(agr_pr, att_pr, universe, dist)
    mg, amb_g = hac(Dg, linkage)
    mp, amb_p = hac(Dp, linkage)

    Cg, Cp = cophenetic(mg, n), cophenetic(mp, n)
    flat_g = [Cg[i][j] for i in range(n) for j in range(i + 1, n)]
    flat_p = [Cp[i][j] for i in range(n) for j in range(i + 1, n)]

    rows = []
    for k in ks:
        if not 2 <= k <= n - 1:
            continue
        pg, pp = cut_tree(mg, n, k), cut_tree(mp, n, k)
        big = max(max(Counter(pg).values()), max(Counter(pp).values()))
        rows.append({
            "k": k,
            "degenerate": big >= 0.9 * n,
            "largest_share": big / n,
            "ARI": adjusted_rand(pg, pp),
            "FM": fowlkes_mallows(pg, pp),
            "VI": variation_of_information(pg, pp),
            "sizes_gt": sorted(Counter(pg).values(), reverse=True)[:6],
            "sizes_pred": sorted(Counter(pp).values(), reverse=True)[:6],
        })
    return {
        "n": n, "distance": dist, "linkage": linkage,
        "ambiguous_merges_gt": amb_g, "ambiguous_merges_pred": amb_p,
        "cophenetic_spearman": _spearman(flat_g, flat_p),
        "cophenetic_pearson": _pearson(flat_g, flat_p),
        "by_k": rows,
        "_partitions": {r["k"]: (cut_tree(mg, n, r["k"]), cut_tree(mp, n, r["k"]))
                        for r in rows},
        "_universe": list(universe),
    }


# ==========================================================================
# reporting helpers
# ==========================================================================

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return p, r, f


def aggregate(recs: Sequence[Rec]) -> Rec:
    tot = Rec("ALL")
    for r in recs:
        tot.merge(r)
    return tot


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation, ties averaged. No scipy dependency."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


# ==========================================================================
# the paper's named quantities, in one place
# ==========================================================================

def paper_metrics(tot: Rec, k: int = 3, min_degree: int = 1,
                  min_positive: int = 0, ties: str = "expected",
                  global_cer: Optional[float] = None,
                  global_cer_source: str = "n/a") -> Dict[str, Any]:
    m: Dict[str, Any] = {}

    # --- 1. apparatus structure -------------------------------------------
    tp = tot.ent_tp["__all__"]
    fp = tot.ent_fp["__all__"]
    fn = tot.ent_fn["__all__"]
    p, r, f = prf(tp, fp, fn)
    m["entry_precision"] = p
    m["entry_recall"] = r
    m["entry_F1"] = f
    m["_entry_counts"] = {"matched": tp, "spurious": fp, "dropped": fn}
    per_type = {}
    for t in set(list(tot.ent_tp) + list(tot.ent_fp) + list(tot.ent_fn)):
        if t == "__all__":
            continue
        pp, rr, ff = prf(tot.ent_tp[t], tot.ent_fp[t], tot.ent_fn[t])
        per_type[t] = {"gold": tot.ent_tp[t] + tot.ent_fn[t],
                       "matched": tot.ent_tp[t], "spurious": tot.ent_fp[t],
                       "dropped": tot.ent_fn[t],
                       "entry_precision": pp, "entry_recall": rr, "entry_F1": ff}
    m["entry_by_variant_type"] = per_type

    # --- 2. witness extraction --------------------------------------------
    n_grp, ok_grp = tot.exact_n["witness_set"], tot.exact_ok["witness_set"]
    m["sigla_group_exact_match"] = ok_grp / n_grp if n_grp else float("nan")
    m["_sigla_group_counts"] = {"segments": n_grp, "exact": ok_grp}
    for cat, label in (("witnesses", "witnesses"), ("missing", "missing")):
        pp, rr, ff = prf(tot.tp[cat], tot.fp[cat], tot.fn[cat])
        m["sigla_precision__" + label] = pp
        m["sigla_recall__" + label] = rr
        m["sigla_F1__" + label] = ff
    tp_s = tot.tp["witnesses"] + tot.tp["missing"]
    fp_s = tot.fp["witnesses"] + tot.fp["missing"]
    fn_s = tot.fn["witnesses"] + tot.fn["missing"]
    pp, rr, ff = prf(tp_s, fp_s, fn_s)
    m["sigla_precision"] = pp
    m["sigla_recall"] = rr
    m["sigla_F1"] = ff
    gold_s = tp_s + fn_s
    prod_s = tp_s + fp_s
    m["sigla_exact_match"] = tp_s / gold_s if gold_s else float("nan")
    m["sigla_strict_accuracy"] = (tp_s / (gold_s + fp_s)
                                  if gold_s + fp_s else float("nan"))
    m["_sigla_counts"] = {"gold": gold_s, "produced": prod_s,
                          "tp": tp_s, "fp": fp_s, "fn": fn_s}

    # --- 2 bis. the same, restricted to correctly detected units -----------
    # `witnesses` only: `missing` is scored per verse, never per entry, so it
    # is not conditioned by entry alignment and is left out of this block.
    tp_a = tot.tp["witnesses"]
    fp_a = tot.fp["witnesses"] - tot.fp["witnesses_unaligned"]
    fn_a = tot.fn["witnesses"] - tot.fn["witnesses_unaligned"]
    pa, ra, fa = prf(tp_a, fp_a, fn_a)
    m["sigla_precision_aligned"] = pa
    m["sigla_recall_aligned"] = ra
    m["sigla_F1_aligned"] = fa
    m["sigla_exact_match_aligned"] = (tp_a / (tp_a + fn_a)
                                      if tp_a + fn_a else float("nan"))
    n_ga = tot.exact_n["witness_set_aligned"]
    ok_ga = tot.exact_ok["witness_set_aligned"]
    m["sigla_group_exact_match_aligned"] = ok_ga / n_ga if n_ga else float("nan")
    m["_sigla_aligned_counts"] = {
        "gold": tp_a + fn_a, "produced": tp_a + fp_a,
        "tp": tp_a, "fp": fp_a, "fn": fn_a,
        "segments": n_ga, "exact_segments": ok_ga,
        "lost_with_entry": tot.fn["witnesses_unaligned"],
        "invented_with_entry": tot.fp["witnesses_unaligned"]}

    # --- 3. reading transcription -----------------------------------------
    for f_ in PAPER_TEXT_FIELDS:
        cer = tot.dist[f_] / tot.length[f_] if tot.length[f_] else float("nan")
        m["CER_" + f_] = cer
        m["exact_" + f_] = (tot.exact_ok[f_] / tot.exact_n[f_]
                            if tot.exact_n[f_] else float("nan"))
        m["_n_" + f_] = tot.exact_n[f_]
        m["_len_" + f_] = tot.length[f_]
    d = sum(tot.dist[f_] for f_ in PAPER_TEXT_FIELDS)
    n = sum(tot.length[f_] for f_ in PAPER_TEXT_FIELDS)
    m["CER_reading_fields"] = d / n if n else float("nan")

    # --- 3 bis. the same, restricted to correctly detected units -----------
    for f_ in PAPER_TEXT_FIELDS:
        ka = "aligned:" + f_
        m["CER_aligned_" + f_] = (tot.dist[ka] / tot.length[ka]
                                  if tot.length[ka] else float("nan"))
        m["exact_aligned_" + f_] = (tot.exact_ok[ka] / tot.exact_n[ka]
                                    if tot.exact_n[ka] else float("nan"))
        m["_n_aligned_" + f_] = tot.exact_n[ka]
        m["_len_aligned_" + f_] = tot.length[ka]
    da = sum(tot.dist["aligned:" + f_] for f_ in PAPER_TEXT_FIELDS)
    na = sum(tot.length["aligned:" + f_] for f_ in PAPER_TEXT_FIELDS)
    m["CER_aligned_reading_fields"] = da / na if na else float("nan")

    # --- 4. witness agreements --------------------------------------------
    universe = sorted({w for pair in tot.agr_gt for w in pair})
    kn = knn_overlap(tot.agr_gt, tot.agr_pr, universe, k=k, min_degree=min_degree,
                     min_positive=min_positive, ties=ties)
    m["knn_overlap"] = kn["knn_overlap"]
    for mode in TIE_MODES:
        m["knn_overlap_" + mode] = kn["knn_overlap_" + mode]
    m["knn_jaccard"] = kn["knn_jaccard"]
    m["row_spearman"] = row_agreement_correlation(tot.agr_gt, tot.agr_pr,
                                                  universe, min_degree=min_degree)
    m["_knn"] = {kk: vv for kk, vv in kn.items() if kk != "per_witness"}
    m["_knn_per_witness"] = kn["per_witness"]
    m["_agreement"] = matrix_stats(tot.agr_gt, tot.agr_pr, universe)

    # --- 5. clustering (filled in by report(), which owns the CLI options) --
    m["_clustering"] = None

    # --- global CER --------------------------------------------------------
    m["Global_CER"] = global_cer if global_cer is not None else float("nan")
    m["_Global_CER_source"] = global_cer_source
    return m


def print_paper_metrics(m: Dict[str, Any]) -> None:
    def pc(x):
        return "%.4f" % x if x == x else "n/a"

    print("\n" + "=" * 74)
    print("PAPER-READY VALUES  (names as they appear in the evaluation section)")
    print("=" * 74)

    c = m["_entry_counts"]
    print("\n[1] apparatus structure detection")
    print("      matched %d | dropped (in GT, absent from output) %d | spurious %d"
          % (c["matched"], c["dropped"], c["spurious"]))
    print("      entry_precision            %s   = matched / (matched + spurious)"
          % pc(m["entry_precision"]))
    print("      entry_recall               %s   = matched / (matched + dropped)"
          % pc(m["entry_recall"]))
    print("      entry_F1                   %s" % pc(m["entry_F1"]))
    print("      NOTE the paper's prose attaches `entry_precision` to the missing")
    print("           entries and `entry_recall` to the spurious ones. That is the")
    print("           other way round: missing = false negatives = recall.")
    print("\n      by variant type:")
    print("      %-18s %6s %8s %8s %8s %8s %8s %8s"
          % ("type", "gold", "matched", "dropped", "spur.", "P", "R", "F1"))
    for t, d in sorted(m["entry_by_variant_type"].items(),
                       key=lambda kv: -kv[1]["gold"]):
        print("      %-18s %6d %8d %8d %8d %8s %8s %8s"
              % (t, d["gold"], d["matched"], d["dropped"], d["spurious"],
                 pc(d["entry_precision"]), pc(d["entry_recall"]), pc(d["entry_F1"])))

    g = m["_sigla_group_counts"]
    s = m["_sigla_counts"]
    print("\n[2] witness extraction")
    print("      sigla_group_exact_match    %s   (%d/%d segments strictly correct)"
          % (pc(m["sigla_group_exact_match"]), g["exact"], g["segments"]))
    print("      sigla_exact_match          %s   (%d/%d gold sigla recovered)"
          % (pc(m["sigla_exact_match"]), s["tp"], s["gold"]))
    print("      sigla_precision            %s   (%d gold, %d produced)"
          % (pc(m["sigla_precision"]), s["gold"], s["produced"]))
    print("      sigla_recall               %s" % pc(m["sigla_recall"]))
    print("      sigla_F1                   %s" % pc(m["sigla_F1"]))
    print("      sigla_strict_accuracy      %s   (exact / |gold u produced|)"
          % pc(m["sigla_strict_accuracy"]))
    print("      NOTE with substitutions counted as one miss + one false alarm,")
    print("           sigla_exact_match and sigla_recall are the same number by")
    print("           construction. Report sigla_strict_accuracy if the paper")
    print("           wants a distinct 'exact match' figure.")
    print("      %-12s %10s %10s %10s" % ("field", "P", "R", "F1"))
    for lab in ("witnesses", "missing"):
        print("      %-12s %10s %10s %10s"
              % (lab, pc(m["sigla_precision__" + lab]),
                 pc(m["sigla_recall__" + lab]), pc(m["sigla_F1__" + lab])))

    a = m["_sigla_aligned_counts"]
    print("\n      -- restricted to correctly detected units (`witnesses` only) --")
    print("      sigla_group_exact_match_al %s   (%d/%d aligned segments)"
          % (pc(m["sigla_group_exact_match_aligned"]),
             a["exact_segments"], a["segments"]))
    print("      sigla_exact_match_aligned  %s   (%d/%d gold sigla recovered)"
          % (pc(m["sigla_exact_match_aligned"]), a["tp"], a["gold"]))
    print("      sigla_precision_aligned    %s   (%d gold, %d produced)"
          % (pc(m["sigla_precision_aligned"]), a["gold"], a["produced"]))
    print("      sigla_recall_aligned       %s" % pc(m["sigla_recall_aligned"]))
    print("      sigla_F1_aligned           %s" % pc(m["sigla_F1_aligned"]))
    print("      excluded: %d sigla lost with a dropped entry, %d invented with"
          % (a["lost_with_entry"], a["invented_with_entry"]))
    print("      a spurious one. This block conditions on successful detection")
    print("      and is NOT comparable across systems with different entry")
    print("      recall; quote it to attribute error to siglum recognition.")

    print("\n[3] reading transcription")
    print("      %-12s %8s %9s %10s %10s"
          % ("field", "n", "union", "CER", "exact"))
    for f_ in PAPER_TEXT_FIELDS:
        print("      %-12s %8d %9d %10s %10s"
              % (f_, m["_n_" + f_], m["_len_" + f_],
                 pc(m["CER_" + f_]), pc(m["exact_" + f_])))
    print("      exact_reading              %s" % pc(m["exact_reading"]))
    print("      CER over the three fields  %s" % pc(m["CER_reading_fields"]))
    print("      (CER is normalised by max(|gold|, |pred|); it is bounded by 1")
    print("       and computed over the same items as the exact-match rate.)")

    print("\n      -- restricted to correctly detected units --")
    print("      %-12s %8s %9s %10s %10s"
          % ("field", "n", "union", "CER", "exact"))
    for f_ in PAPER_TEXT_FIELDS:
        print("      %-12s %8d %9d %10s %10s"
              % (f_, m["_n_aligned_" + f_], m["_len_aligned_" + f_],
                 pc(m["CER_aligned_" + f_]), pc(m["exact_aligned_" + f_])))
    print("      exact_aligned_reading      %s" % pc(m["exact_aligned_reading"]))
    print("      CER over the three fields  %s"
          % pc(m["CER_aligned_reading_fields"]))
    print("      Dropped and spurious entries are excluded here: their fields")
    print("      score 1.0 by construction and measure unit detection, not")
    print("      transcription. Same caveat on comparability as [2].")

    kn = m["_knn"]
    ag = m["_agreement"]
    print("\n[4] witness agreements (Quentin's grid tables)")
    print("      agreements: %d in GT, %d in the OCR output, over %d witness pairs"
          % (ag["agreements_gt"], ag["agreements_pred"], ag["pairs"]))
    print("      relative L1 distance between matrices   %s" % pc(ag["relative_L1"]))
    print("      Spearman / Pearson, all %d pairs        %s / %s"
          % (ag["pairs"], pc(ag["spearman"]), pc(ag["pearson"])))
    print("      Spearman / Pearson, %d attested pairs   %s / %s"
          % (ag["pairs_nonzero_union"], pc(ag["spearman_nonzero_union"]),
             pc(ag["pearson_nonzero_union"])))
    print("        (%d structural zeros excluded — pairs never co-occurring "
          "on either side)" % ag["structural_zeros"])
    print("      knn_overlap (k=%d, N=%d witnesses, ties=%s)  %s"
          % (kn["k"], kn["N"], kn["ties_mode"], pc(m["knn_overlap"])))
    print("        pessimistic %s   expected %s   optimistic %s"
          % (pc(m["knn_overlap_pessimistic"]), pc(m["knn_overlap_expected"]),
             pc(m["knn_overlap_optimistic"])))
    print("      knn_jaccard  (tie-free)      %s" % pc(m["knn_jaccard"]))
    print("      row Spearman (tie-free)      %s" % pc(m["row_spearman"]))
    if kn["excluded"]:
        print("        (%d witnesses excluded by --knn-min-degree / "
              "--knn-min-positive)" % kn["excluded"])
    if kn["witnesses_with_ties"]:
        print("        (%d of %d witnesses have an ambiguous k-th neighbour, "
              "median tie width %d;" % (kn["witnesses_with_ties"], kn["N"],
                                        kn["median_tie_width"]))
        print("         quote the interval, not a single figure)")
    if kn["witnesses_zero_boundary"]:
        print("        (%d have a k-th GT neighbour with ZERO agreements: their"
              % kn["witnesses_zero_boundary"])
        print("         neighbourhood is not defined by the data at all. Use")
        print("         --knn-min-positive %d to drop them.)" % kn["k"])

    if m.get("_clustering"):
        cl = m["_clustering"]
        print("\n[5] clustering agreement (CAH, %s distance, %s linkage)"
              % (cl["distance"], cl["linkage"]))
        print("      cophenetic Spearman        %s   (K-free)"
              % pc(cl["cophenetic_spearman"]))
        for r in cl["by_k"]:
            print("      ARI at K=%-3d               %s   (VI %.3f bits)"
                  % (r["k"], pc(r["ARI"]), r["VI"]))

    print("\n[G] Global_CER                  %s   (%s)"
          % (pc(m["Global_CER"]), m["_Global_CER_source"]))
    print("=" * 74)


# ==========================================================================
# [S] failures of one variant type, printed before everything else
# ==========================================================================

def print_focus_failures(tot: Rec, focus: str = "substitution",
                         detail: int = 12) -> Dict[str, Any]:
    """Everything that went wrong with one variant type, printed first.

    Four disjoint failure modes:
      dropped    gold unit of this type absent from the output
      typed-out  gold unit of this type given another type
      typed-in   unit of another gold type read as this type
      spurious   unit of this type invented by the OCR

    Only `dropped` and `spurious` are scored by [1]: entry detection ignores
    the type, so a mistyped unit still counts as matched there.
    """
    gold = tot.ent_tp[focus] + tot.ent_fn[focus]
    prod = tot.ent_tp[focus] + tot.ent_fp[focus]
    dropped = [r for r in tot.fails if r[1] == "dropped" and r[2] == focus]
    spurious = [r for r in tot.fails if r[1] == "spurious" and r[3] == focus]
    out_ = [r for r in tot.fails if r[1] == "type" and r[2] == focus]
    in_ = [r for r in tot.fails if r[1] == "type" and r[3] == focus]

    print("\n[S] `%s` — detection and classification failures" % focus)
    if not (gold or prod or dropped or spurious or out_ or in_):
        print("      no unit of this type on either side")
        return {}
    # `prod` counts only units whose GOLD type was `focus`; a unit read as
    # `focus` from another gold type is in `in_` and is added here so that the
    # header agrees with what the pipeline actually emitted.
    print("      %d gold units, %d produced (%d of them from another gold type)"
          % (gold, prod + len(in_), len(in_)))
    print("      %-38s %4d" % ("dropped (absent from the output)", len(dropped)))
    print("      %-38s %4d" % ("typed as something else", len(out_)))
    print("      %-38s %4d" % ("another type read as this one", len(in_)))
    print("      %-38s %4d" % ("spurious (unit invented)", len(spurious)))
    print("      [1] scores only the first and the last: entry detection")
    print("      ignores the type, so a mistyped unit counts there as matched.")

    if out_:
        print("      gold `%s` was typed:" % focus)
        for t, n in Counter(r[3] for r in out_).most_common():
            print("        %-22s %4d" % (t, n))
    if in_:
        print("      read as `%s`, gold was:" % focus)
        for t, n in Counter(r[2] for r in in_).most_common():
            print("        %-22s %4d" % (t, n))

    rows = sorted(dropped + out_ + in_ + spurious,
                  key=lambda r: _verse_sort(r[0]))
    if rows and detail:
        shown = rows if detail < 0 else rows[:detail]
        print("      detail (%d of %d):" % (len(shown), len(rows)))
        for key, kind, gt_, pt_, gtx, ptx in shown:
            print("        [%-7s] %-9s %s -> %s" % (key, kind, gt_, pt_))
            if gtx:
                print("            GT %s" % gtx)
            if ptx:
                print("            OC %s" % ptx)
        if len(rows) > len(shown):
            print("        ... %d more (--focus-detail N, or -1 for all)"
                  % (len(rows) - len(shown)))

    return {"focus": focus, "gold": gold, "produced": prod,
            "dropped": len(dropped), "typed_out": len(out_),
            "typed_in": len(in_), "spurious": len(spurious),
            "typed_out_by_type": dict(Counter(r[3] for r in out_)),
            "typed_in_by_type": dict(Counter(r[2] for r in in_))}


# ==========================================================================
# [3c] the worst text fields, for manual error analysis
# ==========================================================================

def print_text_errors(tot: Rec, fields: Sequence[str] = PAPER_TEXT_FIELDS,
                      worst: int = 10, min_len: int = 0,
                      include_unaligned: bool = False,
                      width: int = 56) -> Dict[str, Any]:
    """List the fields that came back worst, ranked by per-field CER.

    Only `differs` rows are ranked by default. A dropped or spurious entry has
    CER 1.0 by construction and would fill the whole listing without saying
    anything about transcription: those are a unit-detection failure and are
    counted in [0] and [S]. --text-errors-all puts them back.

    `min_len` guards the other end: a two-character field with one wrong
    character scores 0.5 and is not evidence of anything.
    """
    rows = [r for r in tot.texterr if r[1] in fields
            and (include_unaligned or r[2] == "differs")]
    print("\n[3c] worst text fields (for error analysis)")
    if not rows:
        print("      no field differs: nothing to analyse")
        return {}
    kinds = Counter(r[2] for r in tot.texterr if r[1] in fields)
    print("      %d imperfect fields: %s"
          % (sum(kinds.values()),
             ", ".join("%s %d" % (k, n) for k, n in sorted(kinds.items()))))
    if not include_unaligned and (kinds["dropped"] or kinds["spurious"]):
        print("      (%d from unaligned entries excluded; --text-errors-all "
              "to include)" % (kinds["dropped"] + kinds["spurious"]))

    out: Dict[str, Any] = {}
    for f in fields:
        sub = [r for r in rows if r[1] == f and r[6] >= min_len]
        if not sub:
            continue
        sub.sort(key=lambda r: (-(r[5] / r[6] if r[6] else 0.0), -r[6], r[0]))
        shown = sub if worst < 0 else sub[:worst]
        print("      -- %s: %d imperfect, %d shown (min length %d) --"
              % (f, len(sub), len(shown), min_len))
        for key, _f, kind, g, p, d, n in shown:
            print("        [%-7s] cer=%.3f  d=%d/%d  %s"
                  % (key, d / n if n else float("nan"), d, n,
                     "" if kind == "differs" else "(" + kind + ")"))
            print("            GT %s" % g[:width])
            print("            OC %s" % p[:width])
        out[f] = {"imperfect": len(sub),
                  "worst": [{"verse": r[0], "kind": r[2], "gold": r[3],
                             "pred": r[4], "d": r[5], "n": r[6],
                             "cer": r[5] / r[6] if r[6] else None}
                            for r in shown]}
    return out


def dump_text_errors(path: str, tot: Rec) -> int:
    cols = ["verse", "field", "kind", "d", "n", "cer", "ground_truth", "pipeline"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for key, f, kind, g, p, d, n in sorted(
                tot.texterr, key=lambda r: (r[1], -(r[5] / r[6] if r[6] else 0))):
            fh.write("%s\t%s\t%s\t%d\t%d\t%.6f\t%s\t%s\n"
                     % (key, f, kind, d, n, (d / n if n else float("nan")),
                        g.replace("\t", " "), p.replace("\t", " ")))
    return len(tot.texterr)


# ==========================================================================
# [2d] sigla scored on aligned entries only
# ==========================================================================

def print_conditioned_sigla(tot: Rec) -> Dict[str, Any]:
    """Witness P/R/F1 restricted to correctly detected units.

    The headline figures in [2] are unconditioned: a siglum lost with its whole
    entry is a false negative and a siglum invented with a spurious entry is a
    false positive. That is the comparable quantity — conditioning on successful
    detection lets a model that drops a third of the apparatus score well on the
    third it kept, and hides the compound failure between the two stages.

    The conditioned figure below is nevertheless the one that attributes error
    to siglum recognition rather than to unit detection, so it is the number to
    quote in an ablation of the glyph classifier. Both are reported; neither
    replaces the other. Note that `missing` (Desunt) is scored per verse and
    never per entry, so it is unaffected by the conditioning and is excluded
    here.
    """
    tp = tot.tp["witnesses"]
    fn_all, fp_all = tot.fn["witnesses"], tot.fp["witnesses"]
    fn_un, fp_un = tot.fn["witnesses_unaligned"], tot.fp["witnesses_unaligned"]
    fn_c, fp_c = fn_all - fn_un, fp_all - fp_un

    p_u, r_u, f_u = prf(tp, fp_all, fn_all)
    p_c, r_c, f_c = prf(tp, fp_c, fn_c)

    print("\n[2d] witness sigla: unconditioned vs aligned-entries-only")
    print("      %-16s %6s %6s %6s %9s %9s %9s"
          % ("population", "tp", "fp", "fn", "P", "R", "F1"))
    print("      %-16s %6d %6d %6d %9.4f %9.4f %9.4f"
          % ("all entries", tp, fp_all, fn_all, p_u, r_u, f_u))
    print("      %-16s %6d %6d %6d %9.4f %9.4f %9.4f"
          % ("aligned only", tp, fp_c, fn_c, p_c, r_c, f_c))
    print("      %d sigla lost with a dropped entry, %d invented with a "
          "spurious one" % (fn_un, fp_un))
    if fn_all:
        print("      -> %.1f%% of all misses are unit-detection failures, "
              "not misreadings" % (100 * fn_un / fn_all))
    print("      Report the first row as the headline: the second conditions on")
    print("      successful detection and is not comparable across systems with")
    print("      different entry recall. Quote the second only to attribute")
    print("      error to the glyph classifier.")

    return {"tp": tp,
            "unconditioned": {"fp": fp_all, "fn": fn_all,
                              "P": p_u, "R": r_u, "F1": f_u},
            "aligned_only": {"fp": fp_c, "fn": fn_c,
                             "P": p_c, "R": r_c, "F1": f_c},
            "lost_with_entry": fn_un, "invented_with_entry": fp_un}


# ==========================================================================
# diagnostics kept from the previous version
# ==========================================================================

BINS = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, 10 ** 6)]


def _bin_label(n: int) -> str:
    for lo, hi in BINS:
        if lo <= n <= hi:
            return "%d" % lo if lo == hi else ("%d-%d" % (lo, hi) if hi < 10 ** 6
                                               else "%d+" % lo)
    return "?"


def length_analysis(tot: Rec, out: Dict[str, Any], bootstrap: int = 0,
                    seed: int = 17) -> None:
    """Does a longer witness run degrade the OCR, or does it just offer more
    chances to fail?"""
    pts = tot.points
    if not pts:
        return
    aligned = [q for q in pts if q[4]]
    gold_all = sum(q[0] for q in pts)
    miss_all = sum(q[1] for q in pts)
    p_hat = miss_all / gold_all if gold_all else 0.0

    print("\n[D] effect of witness-run length")
    print("  overall per-siglum miss rate p = %d/%d = %.4f" % (miss_all, gold_all, p_hat))
    print("  %-8s %6s %7s %9s %9s %11s %11s" %
          ("len", "units", "sigla", "miss/sig", "inv/sig", "unit exact", "(1-p)^L"))
    rows = {}
    for q in aligned:
        rows.setdefault(_bin_label(q[0]), []).append(q)
    order = [lbl for lbl in ("1", "2-3", "4-6", "7-10", "11-20", "21+") if lbl in rows]
    binned = {}
    for lbl in order:
        qs = rows[lbl]
        g = sum(q[0] for q in qs)
        miss = sum(q[1] for q in qs)
        inv = sum(q[2] for q in qs)
        ex = sum(q[3] for q in qs)
        mean_len = g / len(qs)
        pred = (1 - p_hat) ** mean_len
        print("  %-8s %6d %7d %9.4f %9.4f %11.4f %11.4f"
              % (lbl, len(qs), g, miss / g if g else float("nan"),
                 inv / g if g else float("nan"), ex / len(qs), pred))
        binned[lbl] = {"units": len(qs), "sigla": g,
                       "miss_rate": miss / g if g else None,
                       "invent_rate": inv / g if g else None,
                       "unit_exact": ex / len(qs), "independent_pred": pred}
    out["length_analysis"] = {"p_hat": p_hat, "bins": binned}

    xs = [q[0] for q in aligned if q[0]]
    ys = [q[1] / q[0] for q in aligned if q[0]]
    rho = _spearman(xs, ys)
    print("  Spearman rho(run length, per-siglum miss rate) = %.3f  over %d units"
          % (rho, len(xs)))
    out["length_analysis"]["spearman_rho"] = rho
    if bootstrap:
        rng = random.Random(seed)
        draws = []
        for _ in range(bootstrap):
            idx = [rng.randrange(len(xs)) for _ in xs]
            r = _spearman([xs[i] for i in idx], [ys[i] for i in idx])
            if r == r:
                draws.append(r)
        draws.sort()
        if len(draws) > 10:
            lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]
            print("    95%% CI [%.3f, %.3f]%s" % (lo, hi,
                  "" if lo > 0 or hi < 0 else "   (crosses 0: no length effect shown)"))
            out["length_analysis"]["rho_ci"] = [lo, hi]

    unal = [q for q in pts if not q[4]]
    if unal:
        print("\n  entry-alignment failure by gold run length:")
        print("  %-8s %8s %10s %9s" % ("len", "aligned", "unaligned", "fail rate"))
        for lbl in ("1", "2-3", "4-6", "7-10", "11-20", "21+"):
            a = sum(1 for q in aligned if _bin_label(q[0]) == lbl)
            u = sum(1 for q in unal if _bin_label(q[0]) == lbl)
            if a + u:
                print("  %-8s %8d %10d %9.4f" % (lbl, a, u, u / (a + u)))


# ==========================================================================
# the full report
# ==========================================================================

def report(recs: Sequence[Rec], args, extra: Dict[str, Any]) -> Dict[str, Any]:
    tot = aggregate(recs)
    out: Dict[str, Any] = {}

    out["focus_failures"] = print_focus_failures(
        tot, focus=args.focus_type, detail=args.focus_detail)

    out["config"] = {"sim_floor": args.sim_floor, "gap": args.gap,
                     "assign": args.assign, "weights": [W_TXT, W_WIT, W_TYPE],
                     "k": args.knn, "score_notes": bool(args.score_notes),
                     "text_fields_scored": list(TEXT_ORDER)}

    if extra.get("census_gt"):
        print("\n[C] corpus census (fills the 'X verses / X units / X segments' "
              "placeholders)")
        print_census("ground truth", extra["census_gt"])
        print_census("pipeline output", extra["census_pred"])
        out["census"] = {"gt": extra["census_gt"], "pred": extra["census_pred"]}

    print("\n[0] verse- and entry-level structure "
          "(OCR-induced divergence, not parser quality)")
    s = tot.struct
    print("  verses: %d gt / %d pred | dropped %d | spurious %d"
          % (s["verses_gt"], s["verses_pred"], s["verses_dropped"], s["verses_spurious"]))
    print("  entries: matched %d | dropped %d | spurious %d"
          % (s["entries_matched"], s["entries_dropped"], s["entries_spurious"]))
    denom = s["entries_matched"] + s["entries_dropped"] + s["entries_spurious"]
    print("  entry-alignment rate = %.4f" % (s["entries_matched"] / denom if denom else float("nan")))
    out["structure"] = dict(s)

    issues = getattr(args, "_issues", []) or []
    if issues:
        by_class = Counter(i["class"] for i in issues)
        print("\n  why entries failed to align:")
        for k in ("split", "merge", "near-miss", "crossed", "orphan"):
            if by_class[k]:
                print("    %-10s %4d" % (k, by_class[k]))
        if by_class["crossed"]:
            print("    (`crossed` can only occur with the order-preserving NW "
                  "aligner; try --assign greedy)")
        near = [i for i in issues if i["class"] == "near-miss"]
        if near:
            sims = ", ".join("%.2f" % x for x in
                             sorted((i["sim"] for i in near), reverse=True)[:8])
            print("    near-miss similarities (floor is %.2f): %s"
                  % (args.sim_floor, sims))
        worst = Counter(i["verse"] for i in issues).most_common(args.structure_report or 8)
        print("  verses with the most unaligned entries: %s"
              % ", ".join("%s (%d)" % (v, n) for v, n in worst))
        if args.structure_report:
            print("\n  unaligned entries in detail")
            keys = {v for v, _ in worst}
            for i in issues:
                if i["verse"] not in keys:
                    continue
                print("  [%s] %-9s %-9s sim=%.2f  type=%-16s wit=%d"
                      % (i["verse"], i["side"], i["class"], i["sim"],
                         i["variant_type"] or "-", i["n_wit"]))
                print("      this : %s" % i["text"])
                if i["best_text"]:
                    print("      near : %s" % i["best_text"])
        out["structure_issues"] = dict(by_class)

    print("\n[2b] set-valued categories (micro P / R / F1)")
    print("  %-12s %7s %6s %6s %6s %8s %8s %8s" %
          ("category", "gold", "tp", "fp", "fn", "P", "R", "F1"))
    for cat in Rec.SET_CATS:
        g = tot.tp[cat] + tot.fn[cat]
        if not g and not tot.fp[cat]:
            continue
        p, r, f = prf(tot.tp[cat], tot.fp[cat], tot.fn[cat])
        print("  %-12s %7d %6d %6d %6d %8.4f %8.4f %8.4f"
              % (cat, g, tot.tp[cat], tot.fp[cat], tot.fn[cat], p, r, f))
        out.setdefault("sets", {})[cat] = {"gold": g, "tp": tot.tp[cat],
                                           "fp": tot.fp[cat], "fn": tot.fn[cat],
                                           "P": p, "R": r, "F1": f}

    if tot.exact_n["witness_set"]:
        n, ok = tot.exact_n["witness_set"], tot.exact_ok["witness_set"]
        print("  strict per-segment witness list: %d/%d segments fully correct = %.4f"
              % (ok, n, ok / n))
        out.setdefault("sets", {})["sigla_group_exact_match"] = {"n": n, "ok": ok,
                                                                 "rate": ok / n}

    print("\n[3b] text fields (CER / exact-match)")
    print("  %-12s %7s %8s %8s %10s" % ("field", "n", "union", "CER", "exact"))
    for f in TEXT_ORDER:
        if not tot.exact_n[f]:
            continue
        cer = tot.dist[f] / tot.length[f] if tot.length[f] else float("nan")
        ex = tot.exact_ok[f] / tot.exact_n[f]
        print("  %-12s %7d %8d %8.4f %10.4f" % (f, tot.exact_n[f], tot.length[f], cer, ex))
        out.setdefault("text", {})[f] = {"n": tot.exact_n[f], "chars": tot.length[f],
                                         "CER": cer, "exact": ex}

    out["text_errors"] = print_text_errors(
        tot, worst=args.worst_cer, min_len=args.worst_cer_min_len,
        include_unaligned=args.text_errors_all)

    out["sigla_conditioned"] = print_conditioned_sigla(tot)

    print("\n[1b] categorical fields (accuracy)")
    for c in Rec.CLS_CATS:
        if tot.cls_n[c]:
            acc = tot.cls_ok[c] / tot.cls_n[c]
            print("  %-18s %6d  %.4f" % (c, tot.cls_n[c], acc))
            out.setdefault("classes", {})[c] = {"n": tot.cls_n[c], "acc": acc}

    print("\n[2c] siglum error taxonomy — the 'nature of the differences'")
    print("  (covers `witnesses` and `missing` together, i.e. the same population")
    print("   as sigla_precision / sigla_recall / sigla_F1 in the paper block)")
    tx = tot.taxo
    subs = sum(tx[k] for k in ("hand-mark", "superscript-letter", "base-glyph"))
    dropped = tx["dropped"] + tx["dropped-unaligned-entry"]
    spurious = tx["spurious"] + tx["spurious-unaligned-entry"]
    gold = tx["exact"] + subs + dropped
    produced = tx["exact"] + subs + spurious

    def pct(n, d):
        return 100 * n / d if d else float("nan")

    print("  -- what became of the %d gold sigla --" % gold)
    print("     %-26s %6d  (%.2f%%)" % ("recovered exactly", tx["exact"], pct(tx["exact"], gold)))
    for k in ("hand-mark", "superscript-letter", "base-glyph"):
        if tx[k]:
            print("     %-26s %6d  (%.2f%%)   misread as another siglum"
                  % (k, tx[k], pct(tx[k], gold)))
    if tx["dropped"]:
        print("     %-26s %6d  (%.2f%%)   entry aligned, siglum lost"
              % ("dropped", tx["dropped"], pct(tx["dropped"], gold)))
    if tx["dropped-unaligned-entry"]:
        print("     %-26s %6d  (%.2f%%)   lost with its whole entry"
              % ("dropped w/ entry", tx["dropped-unaligned-entry"],
                 pct(tx["dropped-unaligned-entry"], gold)))

    print("  -- what the pipeline produced: %d sigla --" % produced)
    print("     %-26s %6d  (%.2f%%)" % ("correct", tx["exact"], pct(tx["exact"], produced)))
    print("     %-26s %6d  (%.2f%%)" % ("wrong (substitutions)", subs, pct(subs, produced)))
    if tx["spurious"]:
        print("     %-26s %6d  (%.2f%%)   invented inside an aligned entry"
              % ("spurious", tx["spurious"], pct(tx["spurious"], produced)))
    if tx["spurious-unaligned-entry"]:
        print("     %-26s %6d  (%.2f%%)   came with a spurious entry"
              % ("spurious w/ entry", tx["spurious-unaligned-entry"],
                 pct(tx["spurious-unaligned-entry"], produced)))
    print("  -- reconciliation with sigla_precision / sigla_recall --")
    tp_c = tot.tp["witnesses"] + tot.tp["missing"]
    fp_c = tot.fp["witnesses"] + tot.fp["missing"]
    fn_c = tot.fn["witnesses"] + tot.fn["missing"]
    print("     tp = recovered exactly        = %d   (table: %d)" % (tx["exact"], tp_c))
    print("     fn = substitutions + dropped  = %d + %d = %d   (table: %d)"
          % (subs, dropped, subs + dropped, fn_c))
    print("     fp = substitutions + spurious = %d + %d = %d   (table: %d)"
          % (subs, spurious, subs + spurious, fp_c))
    out["siglum_taxonomy"] = dict(tx)

    wit_errs = [e for e in (getattr(args, "_errors", []) or [])
                if e[1] == "witnesses"]
    if wit_errs:
        KINDS = ("dropped", "dropped-unaligned-entry", "spurious",
                 "spurious-unaligned-entry", "base-glyph",
                 "superscript-letter", "hand-mark")
        print("\n  where the siglum errors are")
        loc = {}
        for key, _cat, kind, g, p in wit_errs:
            if kind.startswith("spurious"):
                label = p
            elif kind.startswith("dropped"):
                label = g
            else:
                label = "%s>%s" % (g, p)          # substitution: gold > produced
            loc.setdefault(kind, {}).setdefault(key, []).append(label)
        for kind in KINDS:
            if kind not in loc:
                continue
            verses = loc[kind]
            n = sum(len(v) for v in verses.values())
            print("  %s -- %d in %d verses" % (kind, n, len(verses)))
            items = sorted(verses.items(), key=lambda kv: (-len(kv[1]), _verse_sort(kv[0])))
            shown = items if args.where is True or args.where == 0 else items[:args.where]
            for key, sigla in shown:
                listing = ", ".join(sigla[:12]) + (" ..." if len(sigla) > 12 else "")
                print("      %-8s %3d   %s" % (key, len(sigla), listing))
            if len(items) > len(shown):
                print("      ... and %d more verses (see --dump-errors)"
                      % (len(items) - len(shown)))
        out["siglum_error_locations"] = {
            k: {vk: len(v) for vk, v in vs.items()} for k, vs in loc.items()}
    if tot.glyph:
        print("  base-glyph confusions:")
        for (a, b), n in tot.glyph.most_common(12):
            print("      %-8s -> %-8s %4d" % (a or "0", b or "0", n))
        out["glyph_confusions"] = {"%s>%s" % (a, b): n for (a, b), n in tot.glyph.items()}

    vt_by_gold = defaultdict(lambda: [0, 0])       # gold -> [n, correct]
    for (g, p), c in tot.confusion.items():
        vt_by_gold[g][0] += c
        if g == p:
            vt_by_gold[g][1] += c
    if vt_by_gold:
        print("\n[1c] variant_type success rate (per gold type, aligned entries only)")
        print("  %-20s %7s %8s %9s" % ("gold type", "n", "correct", "success"))
        for g, (n, ok) in sorted(vt_by_gold.items(), key=lambda kv: (-kv[1][0], kv[0])):
            print("  %-20s %7d %8d %9.4f" % (g, n, ok, ok / n if n else float("nan")))
            out.setdefault("variant_type_per_class", {})[g] = {
                "n": n, "correct": ok, "success": ok / n if n else None}

    conf = {k: v for k, v in tot.confusion.items() if k[0] != k[1]}
    if conf:
        print("\n  variant_type confusions (gt -> pred)")
        for (a, b), n in sorted(conf.items(), key=lambda kv: -kv[1]):
            print("  %-18s -> %-18s %4d" % (a, b, n))
        out["variant_type_confusions"] = {"%s>%s" % (a, b): n for (a, b), n in conf.items()}

    # ---- [4] agreements ---------------------------------------------------
    universe = sorted({w for pair in tot.agr_gt for w in pair})
    print("\n[4] witness agreement matrix (Quentin's grid tables)")
    if not universe:
        print("  no witness co-occurrence in the ground truth: nothing to compare")
    else:
        ag = matrix_stats(tot.agr_gt, tot.agr_pr, universe)
        print("  %d witnesses, %d unordered pairs" % (len(universe), ag["pairs"]))
        print("  total agreements   GT %d | OCR %d" % (ag["agreements_gt"],
                                                       ag["agreements_pred"]))
        print("  non-zero cells     GT %d | OCR %d" % (ag["nonzero_gt"],
                                                       ag["nonzero_pred"]))
        print("  relative L1        %.4f" % ag["relative_L1"])
        print("  Frobenius ||M_gt - M_ocr||_F / ||M_gt||_F        %.4f"
              % ag["frobenius_relative"])
        print("    after optimal rescaling (c = %.4f)              %.4f"
              % (ag["frobenius_optimal_scale"],
                 ag["frobenius_relative_rescaled"]))
        print("    the gap between the two is uniform shrinkage from dropped "
              "sigla, not structure")
        print("  %-32s %7s %10s %10s"
              % ("cells correlated", "pairs", "Spearman", "Pearson"))
        print("  %-32s %7d %10.4f %10.4f"
              % ("all pairs", ag["pairs"], ag["spearman"], ag["pearson"]))
        print("  %-32s %7d %10.4f %10.4f"
              % ("attested in GT or OCR", ag["pairs_nonzero_union"],
                 ag["spearman_nonzero_union"], ag["pearson_nonzero_union"]))
        print("  %-32s %7d %10.4f %10.4f"
              % ("attested in GT", ag["pairs_nonzero_gt"],
                 ag["spearman_nonzero_gt"], ag["pearson_nonzero_gt"]))
        if ag["structural_zeros"]:
            print("    %d of the %d cells (%.1f%%) are structural zeros — pairs "
                  "that never co-occur" % (ag["structural_zeros"], ag["pairs"],
                                           100 * ag["structural_zeros"] / ag["pairs"]))
            print("    on either side. They agree trivially; quote the restricted "
                  "rows.")
        out["agreement"] = ag
        print("  %-4s %6s %6s %11s %11s %11s %9s"
              % ("k", "N", "tied", "pessim.", "expected", "optimist.", "0-bound"))
        kn_all = {}
        for kk in sorted({1, 2, 3, 5, 10, args.knn}):
            if kk >= len(universe):
                continue
            r = knn_overlap(tot.agr_gt, tot.agr_pr, universe, k=kk,
                            min_degree=args.knn_min_degree,
                            min_positive=args.knn_min_positive, ties=args.ties)
            kn_all[kk] = {mo: r["knn_overlap_" + mo] for mo in TIE_MODES}
            mark = "  <- paper" if kk == args.knn else ""
            print("  %-4d %6d %6d %11.4f %11.4f %11.4f %9d%s"
                  % (kk, r["N"], r["witnesses_with_ties"],
                     r["knn_overlap_pessimistic"], r["knn_overlap_expected"],
                     r["knn_overlap_optimistic"], r["witnesses_zero_boundary"],
                     mark))
        out["knn_overlap_by_k"] = kn_all
        main = knn_overlap(tot.agr_gt, tot.agr_pr, universe, k=args.knn,
                           min_degree=args.knn_min_degree,
                           min_positive=args.knn_min_positive, ties=args.ties)
        out["knn_overlap"] = {kk: vv for kk, vv in main.items()
                              if kk != "per_witness"}
        out["knn_overlap_per_witness"] = main["per_witness"]
        if main["witnesses_with_ties"]:
            print("  tie diagnosis: %d/%d witnesses have an ambiguous k-th "
                  "neighbour," % (main["witnesses_with_ties"], main["N"]))
            print("    median tie width %d; %d of them tie at ZERO agreements."
                  % (main["median_tie_width"], main["witnesses_zero_boundary"]))
            bs = Counter(v["gt_boundary_score"] for v in main["per_witness"].values())
            print("    GT score at the k-th neighbour: %s"
                  % ", ".join("%s->%d wit." % (sc, n)
                              for sc, n in sorted(bs.items(),
                                                  key=lambda kv: (kv[0] is None,
                                                                  kv[0]))[:8]))
        ceil = knn_overlap(tot.agr_gt, tot.agr_gt, universe, k=args.knn,
                           min_degree=args.knn_min_degree,
                           min_positive=args.knn_min_positive, ties=args.ties)
        print("  ceiling — the SAME matrix scored against itself:")
        print("    pessim. %.4f | expected %.4f | optimist. %.4f | jaccard %.4f"
              % (ceil["knn_overlap_pessimistic"], ceil["knn_overlap_expected"],
                 ceil["knn_overlap_optimistic"], ceil["knn_jaccard"]))
        print("    a perfect OCR does not reach 1.0 because the two sides break "
              "the same tie")
        print("    independently. Read the OCR figure against this ceiling, not "
              "against 1.0.")
        out["knn_ceiling"] = {kk: vv for kk, vv in ceil.items()
                              if kk != "per_witness"}

        print("  tie-free alternatives (nothing arbitrary is chosen):")
        print("    knn_jaccard (tie-inclusive neighbourhoods)  %.4f"
              % main["knn_jaccard"])
        rho_row = row_agreement_correlation(tot.agr_gt, tot.agr_pr, universe,
                                            min_degree=args.knn_min_degree)
        print("    mean per-witness Spearman over the grid row  %.4f" % rho_row)
        out["knn_jaccard"] = main["knn_jaccard"]
        out["row_spearman"] = rho_row

        worst = sorted(main["per_witness"].items(),
                       key=lambda kv: (kv[1]["expected"], kv[0]))[:10]
        if worst:
            print("  witnesses whose neighbourhood is worst preserved (expected):")
            print("      " + ", ".join("%s (%.2f)" % (w, v["expected"])
                                       for w, v in worst))
        if args.scatter:
            rows = scatter_data(tot.agr_gt, tot.agr_pr, universe,
                                drop_zeros=not args.scatter_keep_zeros,
                                highlight=args.scatter_highlight)
            made = write_scatter(args.scatter, rows, ag,
                                 logscale=args.scatter_log)
            print("  scatter: %d pairs plotted%s"
                  % (len(rows),
                     "" if args.scatter_keep_zeros
                     else " (%d structural zeros at the origin omitted)"
                          % ag["structural_zeros"]))
            print("    -> %s" % ", ".join(made))
            out["scatter_files"] = made
        if args.dump_agreement:
            dump_matrix(args.dump_agreement + ".gt.tsv", tot.agr_gt, universe)
            dump_matrix(args.dump_agreement + ".pred.tsv", tot.agr_pr, universe)
            print("  matrices -> %s.gt.tsv, %s.pred.tsv"
                  % (args.dump_agreement, args.dump_agreement))

    # ---- [5] clustering ---------------------------------------------------
    if universe and not args.no_cluster and len(universe) >= 4:
        att_g, att_p = tot.att_gt, tot.att_pr
        keep = [w for w in universe if att_g.get(w, 0) >= args.cluster_min_attest]
        print("\n[5] clustering agreement — CAH on the two grid tables")
        print("  distance 1-%s | linkage %s | %d witnesses (%d dropped by "
              "--cluster-min-attest %d)"
              % (args.cluster_distance, args.linkage, len(keep),
                 len(universe) - len(keep), args.cluster_min_attest))
        if len(keep) < 4:
            print("  too few witnesses left to cluster")
        else:
            ks = [int(x) for x in args.clusters.split(",") if x.strip()]
            cl = cluster_comparison(tot.agr_gt, att_g, tot.agr_pr, att_p,
                                    keep, ks, dist=args.cluster_distance,
                                    linkage=args.linkage)
            print("  ambiguous merges (tied distances): GT %d / OCR %d out of %d"
                  % (cl["ambiguous_merges_gt"], cl["ambiguous_merges_pred"],
                     len(keep) - 1))
            print("  cophenetic correlation between the two dendrograms:"
                  "  Spearman %.4f | Pearson %.4f"
                  % (cl["cophenetic_spearman"], cl["cophenetic_pearson"]))
            print("    (K-free: compares the whole hierarchy, no cut to choose)")
            print("  %-4s %8s %8s %8s %3s  %s"
                  % ("K", "ARI", "FM", "VI(bits)", "", "cluster sizes GT | OCR"))
            for r in cl["by_k"]:
                print("  %-4d %8.4f %8.4f %8.4f %3s  %s | %s"
                      % (r["k"], r["ARI"], r["FM"], r["VI"],
                         "!!" if r["degenerate"] else "",
                         ",".join(map(str, r["sizes_gt"])),
                         ",".join(map(str, r["sizes_pred"]))))
            print("    ARI 1.0 = identical grouping, 0.0 = no better than a "
                  "random partition")
            print("    with the same cluster sizes. VI is a metric in bits, "
                  "0 = identical.")
            deg = [r for r in cl["by_k"] if r["degenerate"]]
            if deg:
                print("    !! DEGENERATE CUT: one cluster holds >=90%% of the "
                      "witnesses (K=%s)."
                      % ",".join(str(r["k"]) for r in deg))
                print("       ARI is not interpretable on these rows -- its "
                      "denominator (max - expected)")
                print("       nearly vanishes, so it reads as ~0 however well "
                      "the partitions agree.")
                print("       A low VI on the same row is the reliable signal. "
                      "Ignore these K, or use")
                print("       --linkage ward, which does not peel off one "
                      "witness at a time.")
            ok = [r for r in cl["by_k"] if not r["degenerate"]]
            if ok:
                print("    first informative cut: K=%d -- read the table from "
                      "there." % ok[0]["k"])
            if args.linkage_robustness:
                print("  robustness across linkages (ARI at each K):")
                print("    %-10s %s" % ("linkage",
                                        " ".join("K=%-6d" % r["k"] for r in cl["by_k"])))
                for lk in LINKAGES:
                    c2 = cluster_comparison(tot.agr_gt, att_g, tot.agr_pr, att_p,
                                            keep, ks, dist=args.cluster_distance,
                                            linkage=lk)
                    print("    %-10s %s" % (lk, " ".join("%-8.4f" % r["ARI"]
                                                         for r in c2["by_k"])))
            out["clustering"] = {kk: vv for kk, vv in cl.items()
                                 if not kk.startswith("_")}
            if args.dump_clusters:
                with open(args.dump_clusters, "w", encoding="utf-8") as fh:
                    fh.write("k\twitness\tcluster_gt\tcluster_pred\n")
                    for k, (pg, pp) in cl["_partitions"].items():
                        for w, a_, b_ in zip(cl["_universe"], pg, pp):
                            fh.write("%d\t%s\t%d\t%d\n" % (k, w, a_, b_))
                print("  partitions -> %s" % args.dump_clusters)
            extra["clustering"] = cl

    length_analysis(tot, out, bootstrap=args.bootstrap, seed=args.seed)

    # ---- the paper block --------------------------------------------------
    m = paper_metrics(tot, k=args.knn, min_degree=args.knn_min_degree,
                      min_positive=args.knn_min_positive, ties=args.ties,
                      global_cer=extra.get("global_cer"),
                      global_cer_source=extra.get("global_cer_source", "n/a"))
    if extra.get("clustering"):
        m["_clustering"] = {kk: vv for kk, vv in extra["clustering"].items()
                            if not kk.startswith("_")}
    print_paper_metrics(m)
    out["paper"] = {kk: vv for kk, vv in m.items() if not kk.startswith("_")}
    out["paper_detail"] = {kk: vv for kk, vv in m.items() if kk.startswith("_")}

    # ---- bootstrap on the paper's headline numbers ------------------------
    if args.bootstrap and len(recs) > 1:
        rng = random.Random(args.seed)
        keys = ["entry_F1", "sigla_F1", "sigla_group_exact_match",
                "CER_reading", "exact_reading", "knn_overlap",
                "sigla_F1_aligned", "sigla_group_exact_match_aligned",
                "CER_aligned_reading", "exact_aligned_reading"]
        draws = {k: [] for k in keys}
        for _ in range(args.bootstrap):
            sample = [recs[rng.randrange(len(recs))] for _ in recs]
            hh = paper_metrics(aggregate(sample), k=args.knn,
                               min_degree=args.knn_min_degree,
                               min_positive=args.knn_min_positive,
                               ties=args.ties)
            for k in keys:
                if hh.get(k) == hh.get(k):     # not NaN
                    draws[k].append(hh[k])
        print("\n[D] 95%% CI over %d bootstrap resamples of verses:" % args.bootstrap)
        out["bootstrap"] = {}
        for k in keys:
            d = sorted(draws[k])
            if len(d) < 10:
                continue
            lo, hi = d[int(0.025 * len(d))], d[int(0.975 * len(d)) - 1]
            print("    %-26s [%.4f, %.4f]" % (k, lo, hi))
            out["bootstrap"][k] = [lo, hi]

    if args.per_verse:
        print("\n[D] per verse")
        print("  %-10s %8s %8s %8s %8s" % ("verse", "entries", "CER", "wit F1", "errors"))
        for r in recs:
            d = sum(r.dist[f] for f in TEXT_ORDER)
            n = sum(r.length[f] for f in TEXT_ORDER)
            _, _, f1 = prf(r.tp["witnesses"], r.fp["witnesses"], r.fn["witnesses"])
            nerr = r.fp["witnesses"] + r.fn["witnesses"] + \
                sum(r.cls_n[c] - r.cls_ok[c] for c in Rec.CLS_CATS) + \
                r.struct["entries_dropped"] + r.struct["entries_spurious"]
            print("  %-10s %8d %8.4f %8.4f %8d"
                  % (r.key, r.struct["entries_matched"], d / n if n else float("nan"),
                     f1, nerr))
    return out


# ==========================================================================
# worked example for the figure
# ==========================================================================

def worked_example(gt: List[dict], pred: List[dict], key: str, args) -> None:
    gsel = [e for v in gt if verse_key(v) == key for e in (v.get("apparatus") or [])]
    psel = [e for v in pred if verse_key(v) == key for e in (v.get("apparatus") or [])]
    if not gsel and not psel:
        print("!! no verse %s on either side" % key, file=sys.stderr)
        return
    print("\n" + "=" * 74)
    print("WORKED EXAMPLE — verse %s  (material for fig:evaluation_metric)" % key)
    print("=" * 74)
    print("\nground truth: %d unit(s); pipeline: %d unit(s)" % (len(gsel), len(psel)))
    print("\nsimilarity matrix sigma = %.2f*txt + %.2f*wit + %.2f*type"
          % (W_TXT, W_WIT, W_TYPE))
    header = "        " + " ".join("  pi%-2d" % j for j in range(len(psel)))
    print(header)
    for i, g in enumerate(gsel):
        row = []
        for p in psel:
            s = entry_sim(g, p)
            row.append(("%6.3f" % s) if s >= args.sim_floor else ("(%5.3f)" % s))
        print("  ga%-2d %s" % (i, " ".join(row)))
    print("  values in parentheses are below the no-match threshold %.2f"
          % args.sim_floor)

    pairs = align_entries(gsel, psel, gap=args.gap, floor=args.sim_floor,
                          mode=args.assign)
    print("\nalignment (%s):" % args.assign)
    for gi, pi in pairs:
        if gi is not None and pi is not None:
            print("  ga%-2d <-> pi%-2d   sigma=%.3f" % (gi, pi, entry_sim(gsel[gi], psel[pi])))
        elif pi is None:
            print("  ga%-2d <-> --     dropped entry" % gi)
        else:
            print("  --   <-> pi%-2d   spurious entry" % pi)

    print("\nper-pair contributions:")
    rows = []
    for gi, pi in pairs:
        if gi is None or pi is None:
            continue
        g, p = gsel[gi], psel[pi]
        gtx, ptx = entry_texts(g), entry_texts(p)
        for f in PAPER_TEXT_FIELDS:
            if not gtx[f] and not ptx[f]:
                continue
            d = edit_distance(gtx[f], ptx[f])
            n = max(len(gtx[f]), len(ptx[f]))
            print("  ga%-2d %-9s GT %-38s" % (gi, f, gtx[f][:38]))
            print("       %-9s OC %-38s  d=%d / %d"
                  % ("", ptx[f][:38], d, n))
            rows.append((gi, f, gtx[f], ptx[f], d, n))
        gw, pw = entry_witnesses(g), entry_witnesses(p)
        cg, cp = Counter(gw), Counter(pw)
        inter = sum((cg & cp).values())
        print("  ga%-2d witnesses GT %s" % (gi, " ".join(sorted(gw))))
        print("       %-9s OC %s   tp=%d fp=%d fn=%d %s"
              % ("", " ".join(sorted(pw)), inter,
                 sum(cp.values()) - inter, sum(cg.values()) - inter,
                 "[group exact]" if cg == cp else "[group WRONG]"))

    print("\nLaTeX for the figure:")
    print(r"\begin{tabular}{llll r r}")
    print(r"\toprule")
    print(r"unit & field & ground truth & pipeline & $d$ & $\max(|\gamma|,|\pi|)$ \\")
    print(r"\midrule")
    for gi, f, a, b, d, n in rows:
        esc = lambda x: x.replace("_", r"\_").replace("&", r"\&")[:40]
        print(r"$\gamma_{%d}$ & \verb|%s| & %s & %s & %d & %d \\"
              % (gi, f, esc(a), esc(b), d, n))
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("=" * 74)


# ==========================================================================
# LaTeX tables
# ==========================================================================

def emit_latex(m: Dict[str, Any], census_gt: Optional[Dict[str, Any]]) -> None:
    def v(x):
        return "%.4f" % x if x == x else "--"

    print("\n%% ---- table 1: apparatus structure detection --------------------")
    print(r"\begin{tabular}{lrrrrrrr}")
    print(r"\toprule")
    print(r"Variant type & Gold & Matched & Dropped & Spurious & P & R & F$_1$ \\")
    print(r"\midrule")
    for t, d in sorted(m["entry_by_variant_type"].items(), key=lambda kv: -kv[1]["gold"]):
        print(r"%s & %d & %d & %d & %d & %s & %s & %s \\"
              % (t.replace("_", r"\_"), d["gold"], d["matched"], d["dropped"],
                 d["spurious"], v(d["entry_precision"]), v(d["entry_recall"]),
                 v(d["entry_F1"])))
    c = m["_entry_counts"]
    print(r"\midrule")
    print(r"\textbf{All} & %d & %d & %d & %d & %s & %s & %s \\"
          % (c["matched"] + c["dropped"], c["matched"], c["dropped"], c["spurious"],
             v(m["entry_precision"]), v(m["entry_recall"]), v(m["entry_F1"])))
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n%% ---- table 2: witness extraction -------------------------------")
    s = m["_sigla_counts"]
    g = m["_sigla_group_counts"]
    print(r"\begin{tabular}{lrr}")
    print(r"\toprule")
    print(r"Metric & Support & Value \\")
    print(r"\midrule")
    print(r"\verb|sigla_group_exact_match| & %d segments & %s \\"
          % (g["segments"], v(m["sigla_group_exact_match"])))
    print(r"\verb|sigla_exact_match| & %d sigla & %s \\"
          % (s["gold"], v(m["sigla_exact_match"])))
    print(r"\verb|sigla_precision| & %d produced & %s \\"
          % (s["produced"], v(m["sigla_precision"])))
    print(r"\verb|sigla_recall| & %d gold & %s \\" % (s["gold"], v(m["sigla_recall"])))
    print(r"\verb|sigla_F1| & & %s \\" % v(m["sigla_F1"]))
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n%% ---- table 3: reading transcription ----------------------------")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Field & $n$ & CER & Exact \\")
    print(r"\midrule")
    for f_ in PAPER_TEXT_FIELDS:
        print(r"\verb|%s| & %d & %s & %s \\"
              % (f_, m["_n_" + f_], v(m["CER_" + f_]), v(m["exact_" + f_])))
    print(r"\midrule")
    print(r"\verb|Global_CER| & & %s & \\" % v(m["Global_CER"]))
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n%% ---- table 4: witness agreements -------------------------------")
    ag = m["_agreement"]
    kn = m["_knn"]
    print(r"\begin{tabular}{lrr}")
    print(r"\toprule")
    print(r"Quantity & Support & Value \\")
    print(r"\midrule")
    print(r"Agreements, ground truth & %d pairs & %d \\" % (ag["pairs"], ag["agreements_gt"]))
    print(r"Agreements, pipeline & %d pairs & %d \\" % (ag["pairs"], ag["agreements_pred"]))
    print(r"Relative $L_1$ & & %s \\" % v(ag["relative_L1"]))
    print(r"Relative $\|\cdot\|_F$ & & %s \\" % v(ag["frobenius_relative"]))
    print(r"\quad after optimal rescaling & $c=%s$ & %s \\"
          % (v(ag["frobenius_optimal_scale"]), v(ag["frobenius_relative_rescaled"])))
    print(r"Spearman $\rho$, all cells & %d pairs & %s \\"
          % (ag["pairs"], v(ag["spearman"])))
    print(r"Spearman $\rho$, attested cells & %d pairs & %s \\"
          % (ag["pairs_nonzero_union"], v(ag["spearman_nonzero_union"])))
    print(r"\verb|knn_overlap| ($k=%d$) & %d witnesses & %s \\"
          % (kn["k"], kn["N"], v(m["knn_overlap"])))
    print(r"\quad tie interval & %d ambiguous & [%s, %s] \\"
          % (kn["witnesses_with_ties"], v(m["knn_overlap_pessimistic"]),
             v(m["knn_overlap_optimistic"])))
    print(r"\verb|knn_jaccard| (tie-free) & %d witnesses & %s \\"
          % (kn["N"], v(m["knn_jaccard"])))
    print(r"Mean per-witness $\rho$ (tie-free) & %d witnesses & %s \\"
          % (kn["N"], v(m["row_spearman"])))
    print(r"\bottomrule")
    print(r"\end{tabular}")

    if census_gt:
        print("\n%% ---- ground-truth description (fills the placeholders) ------")
        print("%% %d chapters, %d verses, %d units, %d segments, %d characters"
              % (census_gt["chapters"], census_gt["verses"], census_gt["units"],
                 census_gt["segments"],
                 census_gt.get("chars_raw", census_gt["chars_text_fields"])))


# ==========================================================================
# main
# ==========================================================================

def compare_assignments(gt: List[dict], pred: List[dict], args) -> None:
    """Is the order-preserving aligner costing us matches?"""
    gt_by, pr_by = defaultdict(list), defaultdict(list)
    for v in gt:
        gt_by[verse_key(v)].append(v)
    for v in pred:
        pr_by[verse_key(v)].append(v)
    n_nw = n_gr = n_diff = 0
    for key in sorted(set(gt_by) | set(pr_by), key=_verse_sort):
        gs = [e for v in gt_by.get(key, []) for e in (v.get("apparatus") or [])]
        ps = [e for v in pr_by.get(key, []) for e in (v.get("apparatus") or [])]
        a = {(i, j) for i, j in align_entries_nw(gs, ps, args.gap, args.sim_floor)
             if i is not None and j is not None}
        b = {(i, j) for i, j in align_entries_greedy(gs, ps, args.gap, args.sim_floor)
             if i is not None and j is not None}
        n_nw += len(a)
        n_gr += len(b)
        n_diff += len(a ^ b)
    print("\n[D] aligner comparison (order-preserving NW vs order-free greedy)")
    print("  matched pairs: NW %d | greedy %d | symmetric difference %d"
          % (n_nw, n_gr, n_diff))
    if n_diff == 0:
        print("  the two agree exactly: the order-preserving assumption is harmless "
              "here and can be stated as such in the paper.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="corrected parsed JSON")
    ap.add_argument("--pred", required=True, help="parsed JSON of the pipeline output")
    ap.add_argument("--gt-text", metavar="FILE",
                    help="annotated ground truth as raw text/LaTeX, for Global_CER")
    ap.add_argument("--pred-text", metavar="FILE",
                    help="raw OCR output, for Global_CER")
    ap.add_argument("--per-verse", action="store_true")
    ap.add_argument("--no-census", action="store_true",
                    help="skip the corpus census")
    ap.add_argument("--score-notes", action="store_true",
                    help="include the `note` field in the text score "
                         "(excluded by default)")
    ap.add_argument("--no-missing", action="store_true",
                    help="skip the Desunt (`missing`) category")
    ap.add_argument("--focus-type", default="substitution", metavar="TYPE",
                    help="variant type whose failures are printed first "
                         "(default substitution)")
    ap.add_argument("--focus-detail", type=int, default=12, metavar="N",
                    help="individual failures to list (-1 all, 0 none)")
    ap.add_argument("--worst-cer", type=int, default=10, metavar="N",
                    help="list the N worst text fields per field, ranked by CER "
                         "(-1 all, 0 none)")
    ap.add_argument("--worst-cer-min-len", type=int, default=0, metavar="N",
                    help="ignore fields shorter than N characters in that "
                         "listing: a 2-character field with one wrong character "
                         "scores 0.5 and means nothing")
    ap.add_argument("--text-errors-all", action="store_true",
                    help="include fields from dropped and spurious entries in "
                         "the listing (they are CER 1.0 by construction and are "
                         "a unit-detection failure, not a transcription one)")
    ap.add_argument("--dump-text-errors", metavar="TSV",
                    help="write every imperfect text field with its distance, "
                         "union length and CER, for manual error analysis")
    ap.add_argument("--assign", choices=("nw", "greedy"), default="nw",
                    help="entry assignment inside a verse: order-preserving "
                         "Needleman-Wunsch (default) or order-free greedy")
    ap.add_argument("--assign-compare", action="store_true",
                    help="report how much the two assignments differ")
    ap.add_argument("--knn", type=int, default=3, metavar="K",
                    help="k for knn_overlap (paper uses 3)")
    ap.add_argument("--knn-min-degree", type=int, default=1, metavar="N",
                    help="ignore witnesses with fewer than N agreements in the GT")
    ap.add_argument("--knn-min-positive", type=int, default=0, metavar="N",
                    help="ignore witnesses with fewer than N strictly positive "
                         "GT neighbours; set to K so that a witness counts only "
                         "when its k-NN set is defined by the data and not by a "
                         "tie at zero")
    ap.add_argument("--ties", choices=TIE_MODES, default="expected",
                    help="how to resolve an ambiguous k-th neighbour when "
                         "reporting the headline knn_overlap: worst case, "
                         "expectation under random tie-breaking (default), or "
                         "best case. All three are always printed.")
    ap.add_argument("--scatter", metavar="PREFIX",
                    help="scatter of gold vs predicted shared-reading count, one "
                         "point per witness pair; writes PREFIX.tsv always and "
                         "PREFIX.pdf/.png when matplotlib is available")
    ap.add_argument("--scatter-keep-zeros", action="store_true",
                    help="keep pairs that never co-occur on either side; they "
                         "pile up at the origin and swamp the figure")
    ap.add_argument("--scatter-highlight", metavar="REGEX",
                    help="draw in a second colour the pairs whose two sigla both "
                         "match REGEX (e.g. '^Phi' to pick out one group). "
                         "Off by default: witnesses are treated as independent "
                         "and no grouping is inferred from the sigla.")
    ap.add_argument("--scatter-log", action="store_true",
                    help="symlog axes, for a heavy-tailed count distribution")
    ap.add_argument("--no-cluster", action="store_true",
                    help="skip the clustering comparison")
    ap.add_argument("--cluster-distance", choices=DIST_KINDS, default="jaccard",
                    help="co-occurrence -> distance (default jaccard; `raw` is "
                         "unnormalised and only there for comparison)")
    ap.add_argument("--linkage", choices=LINKAGES, default="average",
                    help="CAH linkage (default average / UPGMA)")
    ap.add_argument("--linkage-robustness", action="store_true",
                    help="repeat the comparison under every linkage")
    ap.add_argument("--clusters", default="2,3,4,5,6,8,10,12", metavar="K,K,...",
                    help="cluster counts at which to cut both dendrograms")
    ap.add_argument("--cluster-min-attest", type=int, default=2, metavar="N",
                    help="drop witnesses cited in fewer than N segments of the "
                         "ground truth: they cannot be placed by any method")
    ap.add_argument("--dump-clusters", metavar="TSV",
                    help="write the GT and OCR cluster label of every witness")
    ap.add_argument("--dump-agreement", metavar="PREFIX",
                    help="write PREFIX.gt.tsv and PREFIX.pred.tsv grid tables")
    ap.add_argument("--example", metavar="CH:V",
                    help="print the worked metric computation for one verse")
    ap.add_argument("--where", type=int, default=6, metavar="N",
                    help="list the N worst verses per siglum-error kind "
                         "(0 = all, default 6)")
    ap.add_argument("--structure-report", type=int, default=0, metavar="N",
                    help="print the unaligned entries of the N worst verses")
    ap.add_argument("--dump-structure", metavar="TSV",
                    help="write every unaligned entry with its best counterpart")
    ap.add_argument("--dump-errors", metavar="TSV",
                    help="write every discrepancy for manual error analysis")
    ap.add_argument("--json-out", metavar="FILE", help="machine-readable results")
    ap.add_argument("--latex", action="store_true", help="print booktabs tables")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="95%% CI from N resamples of the verse list")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--gap", type=float, default=0.45,
                    help="entry-alignment: cost of leaving an entry unmatched")
    ap.add_argument("--sim-floor", type=float, default=0.6,
                    help="entry-alignment: below this similarity two entries are "
                         "never considered the same apparatus unit "
                         "(the paper's 0.20)")
    args = ap.parse_args()

    if args.score_notes and "note" not in TEXT_ORDER:
        TEXT_ORDER.insert(3, "note")

    gt = load_json(args.gt)
    pred = load_json(args.pred)

    extra: Dict[str, Any] = {}

    raw_gt = raw_pred = None
    if args.gt_text:
        raw_gt = open(args.gt_text, encoding="utf-8").read()
    if args.pred_text:
        raw_pred = open(args.pred_text, encoding="utf-8").read()

    if not args.no_census:
        extra["census_gt"] = census(gt, raw_gt)
        extra["census_pred"] = census(pred, raw_pred)

    # Global_CER
    if raw_gt is not None and raw_pred is not None:
        a, b = canon_text(raw_gt), canon_text(raw_pred)
        d = edit_distance_long(a, b)
        extra["global_cer"] = d / max(len(a), len(b)) if (a or b) else float("nan")
        extra["global_cer_source"] = ("raw OCR vs annotated GT, %d chars, %d edits"
                                      % (max(len(a), len(b)), d))
    else:
        a = " ".join(entry_text_blob(e) for _k, e in all_entries(gt))
        b = " ".join(entry_text_blob(e) for _k, e in all_entries(pred))
        d = edit_distance_long(a, b)
        extra["global_cer"] = d / max(len(a), len(b)) if (a or b) else float("nan")
        extra["global_cer_source"] = (
            "recomputed from parsed text fields (%d chars); pass --gt-text/"
            "--pred-text for the raw-OCR figure the paper describes"
            % max(len(a), len(b)))

    recs, errors, issues = compare(gt, pred, count_missing=not args.no_missing,
                                   gap=args.gap, floor=args.sim_floor,
                                   mode=args.assign)
    args._issues = issues
    args._errors = errors
    out = report(recs, args, extra)

    if args.assign_compare:
        compare_assignments(gt, pred, args)

    if args.example:
        worked_example(gt, pred, args.example, args)

    if args.dump_errors:
        with open(args.dump_errors, "w", encoding="utf-8") as fh:
            fh.write("verse\tcategory\tkind\tground_truth\tpipeline\n")
            for row in errors:
                fh.write("\t".join(str(x).replace("\t", " ") for x in row) + "\n")
        print("\n%d discrepancies -> %s" % (len(errors), args.dump_errors))

    if args.dump_text_errors:
        n = dump_text_errors(args.dump_text_errors, aggregate(recs))
        print("%d imperfect text fields -> %s" % (n, args.dump_text_errors))

    if args.dump_structure:
        cols = ["verse", "side", "class", "sim", "variant_type", "n_wit",
                "text", "best_text"]
        with open(args.dump_structure, "w", encoding="utf-8") as fh:
            fh.write("\t".join(cols) + "\n")
            for i in issues:
                fh.write("\t".join(str(i[c]).replace("\t", " ") for c in cols) + "\n")
        print("%d unaligned entries -> %s" % (len(issues), args.dump_structure))

    if args.latex:
        tot = aggregate(recs)
        m = paper_metrics(tot, k=args.knn, min_degree=args.knn_min_degree,
                          min_positive=args.knn_min_positive, ties=args.ties,
                          global_cer=extra.get("global_cer"),
                          global_cer_source=extra.get("global_cer_source", "n/a"))
        emit_latex(m, extra.get("census_gt"))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2, default=str)
        print("results -> %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
