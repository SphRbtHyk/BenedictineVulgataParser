#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_apparatus.py
=================

Three-way evaluation of critical-apparatus transcriptions:

    gold  (manual ground truth)
    vlm   (stage-1 VLM output only)
    pipe  (full pipeline: VLM + glyph classifier + write-back)

All three files are expected in the same region-segmented Markdown format::

    ## page 1 region 11 (paragraph_title)

    Desunt $\\mathrm{G}^{\\mathrm{*}}$$\\Sigma^{\\mathrm{T}}$...

Regions are matched across files by their ``(page, region)`` key, so no
document-level alignment is needed.

What it reports
---------------
1.  Character error rate on a notation-normalised rendering of each region.
2.  Witness-level precision / recall / F1, where a *witness* is an atomic
    siglum after expansion of superscript contractions and Fraktur runs
    (``Phi^{RA^2ZGP}`` -> Phi^R, Phi^A2, Phi^Z, Phi^G, Phi^P).
3.  Witness-group exact-match rate.
4.  Editorial-mark recovery (the marks that drive variant typing: + ~ = ] | dagger para).
5.  **Three-way regression table**: for every gold witness, whether it was
    correct in the VLM output and whether it was correct in the pipeline
    output.  The cell "correct in VLM, wrong in pipeline" is the false-override
    rate -- the only unrecoverable failure mode of a write-back design.
6.  Retrieval accuracy bucketed by witness-group length.
7.  Counts of the systematic VLM error categories (Fraktur flattening,
    superscript case-lowering, markup pathologies, glyph confusions).

Usage
-----
    python3 eval_apparatus.py --gold gold.md --vlm vlm.md --pipe pipe.md
    python3 eval_apparatus.py --gold gold.md --vlm vlm.md          # two-way
    python3 eval_apparatus.py ... --json results.json --loose-fraktur

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 1.  Inventory (from the edition's praefatio / Elenchus codicum)
# --------------------------------------------------------------------------

GREEK_BASES = "ΘΦΨΩΛΣΠ"
LATIN_BASES = "ABCGMOPTX"
ALL_BASES = GREEK_BASES + LATIN_BASES

FRAKTUR_LETTERS = set("agrelvscwm")   # printed editions
MARKS = set("*†")                     # hand / correction marks attaching to a modifier
EDITORIAL_MARKS = {"]", "†", "~", "+", "¶", "*", "|", "="}

# LaTeX control sequences -> canonical single characters.
# NB: deliberately NOT normalising \circ -> O or 0 -> O; those are genuine
# VLM misreadings and must be counted as errors.
LATEX_SYMBOL = {
    r"\Theta": "Θ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
    r"\Lambda": "Λ", r"\Sigma": "Σ", r"\Pi": "Π",
    r"\phi": "Φ", r"\varphi": "Φ", r"\sum": "Σ", r"\prod": "Π",
    r"\dagger": "†", r"\P": "¶", r"\S": "§",
    r"\sim": "~", r"\mid": "|", r"\equiv": "=",
    r"\O": "O", r"\emptyset": "O",
    r"\ldots": "...", r"\cdot": ".",
}

# Wrappers whose content is transparent for our purposes.
TRANSPARENT_CMDS = {
    "mathrm", "text", "textup", "textit", "mathbf", "mathit", "mathsf",
    "mathtt", "emph", "textbf", "operatorname", "mbox", "hbox",
}

# Latin words that would false-positive as Fraktur runs under --loose-fraktur.
FRAKTUR_STOPWORDS = {
    "ras", "res", "sacra", "cras", "esse", "sale", "vel", "sec", "ave",
    "vers", "gr", "cf", "cum", "al", "sc", "cra", "acre", "clare", "err",
    "vae", "eras", "aves", "seu", "lae", "vas", "ars", "arces",
}


# --------------------------------------------------------------------------
# 2.  Region parsing
# --------------------------------------------------------------------------

REGION_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*(?:page|p\.?)\s*(\d+)[\s,;:_-]*"
    r"(?:region|reg|r)\.?\s*(\d+)\s*(?:\(([^)]*)\))?\s*$",
    re.I,
)

# Any markdown heading at all, for --inspect diagnostics.
ANY_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*$")

# A verse anchor: a 1-3 digit number followed by '.' at the start of an entry.
# Tolerates a leading list marker, math delimiters and whitespace.
VERSE_RE = re.compile(r"(?:(?<=^)|(?<=[\s|;$]))(\d{1,3})\.\s")

ROMAN_RE = re.compile(r"^\s*\$?\\?[A-Za-z]*\{?([IVXLC]{1,7})\}?\$?\.?\s*$")

ROMAN_VALS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def roman_to_int(s: str) -> Optional[int]:
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        v = ROMAN_VALS.get(ch)
        if v is None:
            return None
        total += -v if v < prev else v
        prev = max(prev, v)
    return total if 0 < total < 200 else None


def parse_regions(path: str) -> "OrderedDict[Tuple[int,int], Dict[str,str]]":
    """
    Split a region-segmented Markdown file into {(page, region): {...}}.

    If the file carries no recognisable region headings, the whole file is
    returned as a single pseudo-region so that verse anchoring can still run.
    """
    regions: "OrderedDict[Tuple[int,int], Dict[str,str]]" = OrderedDict()
    key: Optional[Tuple[int, int]] = None
    kind = ""
    buf: List[str] = []
    preamble: List[str] = []

    def flush() -> None:
        if key is not None:
            regions[key] = {"kind": kind, "text": "\n".join(buf).strip()}

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = REGION_RE.match(line.rstrip("\n"))
            if m:
                flush()
                key = (int(m.group(1)), int(m.group(2)))
                kind = (m.group(3) or "").strip()
                buf = []
            elif key is not None:
                buf.append(line.rstrip("\n"))
            else:
                preamble.append(line.rstrip("\n"))
    flush()

    if not regions:
        body = "\n".join(preamble).strip()
        # strip a leading title line if present
        body = re.sub(r"^\s{0,3}#{1,6}[^\n]*\n", "", body, count=1)
        regions[(0, 0)] = {"kind": "whole_file", "text": body}
    return regions


def to_verse_units(
    regions: "OrderedDict[Tuple[int,int], Dict[str,str]]"
) -> "OrderedDict[Tuple[int,int], Dict[str,str]]":
    """
    Re-key a document by (chapter, verse), merging text across region and page
    boundaries.  This is segmentation-independent: it works whether or not the
    file carries layout-detector region IDs, and it repairs verses split by a
    page break.

    Chapter is taken from a standalone Roman numeral line where present, and
    otherwise incremented whenever the verse counter resets downward.
    """
    units: "OrderedDict[Tuple[int,int], List[str]]" = OrderedDict()
    chapter = 1
    verse = 0
    last_verse = 0

    for _, reg in regions.items():
        text = reg["text"]
        if not text:
            continue
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            rm = ROMAN_RE.match(line)
            if rm:
                r = roman_to_int(rm.group(1))
                if r is not None:
                    chapter = r
                    last_verse = 0
                    continue
            pos = 0
            for m in VERSE_RE.finditer(line):
                head = line[pos:m.start()]
                if head.strip():
                    units.setdefault((chapter, verse), []).append(head)
                v = int(m.group(1))
                if v < last_verse:          # counter reset -> new chapter
                    chapter += 1
                last_verse = v
                verse = v
                pos = m.end()
            tail = line[pos:]
            if tail.strip():
                units.setdefault((chapter, verse), []).append(tail)

    return OrderedDict(
        (k, {"kind": "verse", "text": " ".join(v).strip()})
        for k, v in units.items()
    )


def choose_alignment(
    docs: "Dict[str, OrderedDict]", mode: str, verbose: bool = True
) -> Tuple[str, "Dict[str, OrderedDict]"]:
    """
    Pick a keying scheme shared by all documents.

    'region' is used when the region IDs genuinely overlap across files;
    otherwise we fall back to content-derived verse anchors.
    """
    names = list(docs)
    if mode in ("region", "auto"):
        common = set(docs[names[0]])
        for n in names[1:]:
            common &= set(docs[n])
        smallest = min(len(docs[n]) for n in names)
        if common and len(common) >= 0.5 * smallest:
            return "region", docs
        if mode == "region":
            raise SystemExit(
                "error: --align region requested but only %d region keys are "
                "shared across the input files (smallest file has %d). The "
                "files are not segmented alike; use --align verse."
                % (len(common), smallest)
            )
        if verbose:
            print(
                "note: region IDs are not shared across files (%d common keys); "
                "falling back to verse anchoring." % len(common),
                file=sys.stderr,
            )
    return "verse", {n: to_verse_units(d) for n, d in docs.items()}


def inspect(path: str, label: str) -> None:
    """Diagnostics: what structure does this file actually have?"""
    raw = open(path, encoding="utf-8").read()
    lines = raw.split("\n")
    headings = [l for l in lines if ANY_HEADING_RE.match(l)]
    regions = parse_regions(path)
    verses = to_verse_units(regions)
    print("--- %s: %s" % (label, path))
    print("    bytes=%d lines=%d" % (len(raw), len(lines)))
    print("    markdown headings found: %d" % len(headings))
    for h in headings[:4]:
        print("      %r" % h[:78])
    matched = sum(1 for l in lines if REGION_RE.match(l))
    print("    parsed as 'page N region M': %d" % matched)
    print("    regions: %d   verse units: %d" % (len(regions), len(verses)))
    if verses:
        ks = list(verses)
        print("    verse key range: %s ... %s" % (ks[0], ks[-1]))
    print()



# --------------------------------------------------------------------------
# 3.  Notation normalisation
# --------------------------------------------------------------------------

def _match_brace(s: str, i: int) -> int:
    """Given s[i] == '{', return index just past the matching '}'."""
    depth = 0
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(s)


def flatten_latex(s: str) -> str:
    """
    Recursively resolve LaTeX wrappers and symbol commands into a flat string.

    ``\\mathrm{X}``            -> ``X``
    ``\\mathfrak{agrelvsc}``   -> ``<agrelvsc>``   (Fraktur sentinel)
    ``\\Sigma``                -> ``Σ``
    ``<sup>c</sup>``           -> ``^{c}``
    """
    # HTML superscript -> LaTeX form, before anything else.
    s = re.sub(r"<sup>(.*?)</sup>", r"^{\1}", s, flags=re.S)
    s = re.sub(r"<sup>([^<]*)$", r"^{\1}", s)          # unclosed <sup>
    s = s.replace("</sup>", "")

    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if m:
                cmd = m.group(1)
                j = i + m.end()
                # skip whitespace between command and its argument
                k = j
                while k < n and s[k] in " \t":
                    k += 1
                if cmd == "mathfrak" and k < n and s[k] == "{":
                    end = _match_brace(s, k)
                    out.append("<" + flatten_latex(s[k + 1:end - 1]) + ">")
                    i = end
                    continue
                if cmd in TRANSPARENT_CMDS and k < n and s[k] == "{":
                    end = _match_brace(s, k)
                    out.append(flatten_latex(s[k + 1:end - 1]))
                    i = end
                    continue
                sym = LATEX_SYMBOL.get("\\" + cmd)
                if sym is not None:
                    out.append(sym)
                    i = j
                    continue
                # unknown command: drop the backslash, keep the name
                out.append(" ")
                i = j
                continue
            # escaped punctuation
            out.append(s[i + 1] if i + 1 < n else "")
            i += 2
            continue
        if ch == "$":
            out.append(" ")          # math delimiters are token boundaries
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def normalize_for_cer(s: str) -> str:
    """Notation-neutral rendering used for character error rate."""
    s = flatten_latex(s)
    s = s.replace("<", "").replace(">", "")      # drop Fraktur sentinels
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# --------------------------------------------------------------------------
# 4.  Witness extraction
# --------------------------------------------------------------------------

def _split_modifiers(payload: str) -> List[str]:
    """
    Split a superscript payload into atomic modifiers.

    ``RA^{2}ZGVP`` -> ['R', 'A2', 'Z', 'G', 'V', 'P']
    ``B*``         -> ['B*']
    ``AM*``        -> ['A', 'M*']
    ``*``          -> ['*']
    """
    payload = payload.replace("{", "").replace("}", "")
    mods: List[str] = []
    i = 0
    n = len(payload)
    while i < n:
        c = payload[i]
        if c.isspace():
            i += 1
            continue
        if c.isalpha():
            cur = c.upper()
            i += 1
            while i < n:
                d = payload[i]
                if d in MARKS:
                    cur += d
                    i += 1
                elif d.isdigit():
                    cur += d
                    i += 1
                elif d == "^":
                    i += 1
                else:
                    break
            mods.append(cur)
        elif c in MARKS or c.isdigit():
            if mods:
                mods[-1] += c
            else:
                mods.append(c)
            i += 1
        else:
            i += 1
    return mods


def extract_witnesses(text: str, loose_fraktur: bool = False) -> List[str]:
    """
    Return the ordered list of atomic witnesses in a region.

    A witness is either ``BASE^MOD`` (contractions expanded), a bare Latin
    base capital, or ``f:x`` for a Fraktur letter.
    """
    s = flatten_latex(text)
    s = unicodedata.normalize("NFKC", s)
    wits: List[str] = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        # ---- Fraktur run in explicit markup: <agrelvsc>
        if ch == "<":
            end = s.find(">", i)
            if end == -1:
                i += 1
                continue
            for letter in s[i + 1:end]:
                if letter.lower() in FRAKTUR_LETTERS:
                    wits.append("f:" + letter.lower())
            i = end + 1
            continue

        # ---- siglum base
        if ch in ALL_BASES:
            j = i + 1
            mods: List[str] = []
            # consume any number of consecutive ^ groups
            while j < n and s[j] == "^":
                j += 1
                if j < n and s[j] == "{":
                    end = _match_brace(s, j)
                    mods.extend(_split_modifiers(s[j + 1:end - 1]))
                    j = end
                else:
                    k = j
                    while k < n and (s[k].isalnum() or s[k] in MARKS):
                        k += 1
                    mods.extend(_split_modifiers(s[j:k]))
                    j = k
            # a mark glued immediately after the group belongs to the last mod
            while j < n and s[j] in MARKS and mods:
                mods[-1] += s[j]
                j += 1
            if mods:
                for mod in mods:
                    wits.append(f"{ch}^{mod}")
            else:
                # bare capital: a witness only if isolated (not inside a word)
                prev_ok = i == 0 or not s[i - 1].isalpha()
                nxt = s[j] if j < n else " "
                nxt_ok = not (nxt.isalpha() and nxt.islower())
                if prev_ok and nxt_ok:
                    wits.append(ch)
            i = j
            continue

        # ---- bare lowercase run that may be unmarked Fraktur
        if loose_fraktur and ch.isalpha() and ch.islower():
            k = i
            while k < n and s[k].isalpha() and s[k].islower():
                k += 1
            run = s[i:k]
            if (
                len(run) >= 3
                and set(run) <= FRAKTUR_LETTERS
                and run not in FRAKTUR_STOPWORDS
            ):
                for letter in run:
                    wits.append("f:" + letter)
            i = k
            continue

        i += 1

    return wits


def extract_marks(text: str) -> List[str]:
    """Editorial marks that drive variant typing."""
    s = flatten_latex(text)
    return [c for c in s if c in EDITORIAL_MARKS]


def witness_groups(text: str, loose_fraktur: bool = False) -> List[List[str]]:
    """
    Approximate witness lists: maximal runs of witnesses separated only by
    whitespace / math delimiters in the flattened string.
    """
    s = flatten_latex(text)
    s = unicodedata.normalize("NFKC", s)
    groups: List[List[str]] = []
    for chunk in re.split(r"[;|]|—|\bcum\b|\bcf\.|\bom\.", s):
        wits = extract_witnesses(chunk, loose_fraktur=loose_fraktur)
        if wits:
            groups.append(wits)
    return groups


# --------------------------------------------------------------------------
# 5.  Alignment and scoring primitives
# --------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def align(a: Sequence[str], b: Sequence[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Needleman-Wunsch over token sequences. match 0, mismatch 1, gap 1."""
    na, nb = len(a), len(b)
    d = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(1, na + 1):
        d[i][0] = i
    for j in range(1, nb + 1):
        d[0][j] = j
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            d[i][j] = min(
                d[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
            )
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = na, nb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = 100.0 * tp / (tp + fp) if tp + fp else 0.0
    r = 100.0 * tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# --------------------------------------------------------------------------
# 6.  System-level evaluation
# --------------------------------------------------------------------------

def evaluate_system(
    gold: Dict, sysout: Dict, loose_fraktur: bool = False
) -> Dict:
    """Score one system (vlm or pipe) against gold on shared regions."""
    keys = [k for k in gold if k in sysout]

    cer_num = cer_den = 0
    w_tp = w_fp = w_fn = 0
    m_tp = m_fp = m_fn = 0
    grp_ok = grp_tot = 0
    per_region = {}
    length_bucket: Dict[int, List[int]] = defaultdict(lambda: [0, 0])  # len -> [hit, tot]

    for k in keys:
        g_txt, s_txt = gold[k]["text"], sysout[k]["text"]

        gn, sn = normalize_for_cer(g_txt), normalize_for_cer(s_txt)
        dist = levenshtein(gn, sn)
        cer_num += dist
        cer_den += max(len(gn), 1)

        gw = extract_witnesses(g_txt, loose_fraktur)
        sw = extract_witnesses(s_txt, loose_fraktur)
        gc, sc = Counter(gw), Counter(sw)
        inter = sum((gc & sc).values())
        w_tp += inter
        w_fp += len(sw) - inter
        w_fn += len(gw) - inter

        gm, sm = Counter(extract_marks(g_txt)), Counter(extract_marks(s_txt))
        mi = sum((gm & sm).values())
        m_tp += mi
        m_fp += sum(sm.values()) - mi
        m_fn += sum(gm.values()) - mi

        gg = witness_groups(g_txt, loose_fraktur)
        sg = witness_groups(s_txt, loose_fraktur)
        sg_sets = [Counter(x) for x in sg]
        for grp in gg:
            grp_tot += 1
            gcnt = Counter(grp)
            if gcnt in sg_sets:
                grp_ok += 1
            hit = max((sum((gcnt & c).values()) for c in sg_sets), default=0)
            b = min(len(grp), 12)
            length_bucket[b][0] += hit
            length_bucket[b][1] += len(grp)

        per_region[f"{k[0]}:{k[1]}"] = {
            "cer": 100.0 * dist / max(len(gn), 1),
            "gold_witnesses": len(gw),
            "sys_witnesses": len(sw),
            "witness_hits": inter,
        }

    wp, wr, wf = prf(w_tp, w_fp, w_fn)
    mp, mr, mf = prf(m_tp, m_fp, m_fn)
    return {
        "n_regions": len(keys),
        "cer": 100.0 * cer_num / max(cer_den, 1),
        "witness_precision": wp,
        "witness_recall": wr,
        "witness_f1": wf,
        "witness_tp": w_tp, "witness_fp": w_fp, "witness_fn": w_fn,
        "mark_precision": mp, "mark_recall": mr, "mark_f1": mf,
        "group_exact_match": 100.0 * grp_ok / max(grp_tot, 1),
        "n_groups": grp_tot,
        "length_buckets": {
            str(k): {
                "recovered": v[0], "total": v[1],
                "accuracy": 100.0 * v[0] / max(v[1], 1),
            }
            for k, v in sorted(length_bucket.items())
        },
        "per_region": per_region,
    }


def three_way(gold: Dict, vlm: Dict, pipe: Dict, loose_fraktur: bool = False) -> Dict:
    """
    Per gold witness: correct in VLM? correct in pipeline?

    The cell (vlm_ok=True, pipe_ok=False) is the false-override count: the
    pipeline destroyed a reading the VLM had right.
    """
    keys = [k for k in gold if k in vlm and k in pipe]
    cells = Counter()
    regressed_examples: List[Dict] = []
    repaired_examples: List[Dict] = []
    spurious = Counter()          # pipeline insertions absent from gold
    net_loss_by_len: Dict[int, List[int]] = defaultdict(lambda: [0, 0])

    for k in keys:
        gw = extract_witnesses(gold[k]["text"], loose_fraktur)
        vw = extract_witnesses(vlm[k]["text"], loose_fraktur)
        pw = extract_witnesses(pipe[k]["text"], loose_fraktur)

        v_ok = [False] * len(gw)
        p_ok = [False] * len(gw)
        for gi, si in align(gw, vw):
            if gi is not None and si is not None and gw[gi] == vw[si]:
                v_ok[gi] = True
        for gi, si in align(gw, pw):
            if gi is not None and si is not None and gw[gi] == pw[si]:
                p_ok[gi] = True

        for idx, w in enumerate(gw):
            cells[(v_ok[idx], p_ok[idx])] += 1
            if v_ok[idx] and not p_ok[idx] and len(regressed_examples) < 40:
                regressed_examples.append({"region": f"{k[0]}:{k[1]}", "witness": w})
            if not v_ok[idx] and p_ok[idx] and len(repaired_examples) < 40:
                repaired_examples.append({"region": f"{k[0]}:{k[1]}", "witness": w})

        gc, pc = Counter(gw), Counter(pw)
        for w, c in (pc - gc).items():
            spurious[w] += c

        for grp in witness_groups(gold[k]["text"], loose_fraktur):
            b = min(len(grp), 12)
            net_loss_by_len[b][0] += len(grp)
            net_loss_by_len[b][1] += 1

    total = sum(cells.values())
    preserved = cells[(True, True)]
    repaired = cells[(False, True)]
    regressed = cells[(True, False)]
    unfixed = cells[(False, False)]
    vlm_correct = preserved + regressed
    return {
        "n_regions": len(keys),
        "gold_witnesses": total,
        "preserved": preserved,
        "repaired": repaired,
        "regressed": regressed,
        "unfixed": unfixed,
        "repair_rate_pct": 100.0 * repaired / max(total - vlm_correct, 1),
        "false_override_rate_pct": 100.0 * regressed / max(vlm_correct, 1),
        "net_gain": repaired - regressed,
        "spurious_insertions": sum(spurious.values()),
        "spurious_top": spurious.most_common(15),
        "regressed_examples": regressed_examples,
        "repaired_examples": repaired_examples,
    }


# --------------------------------------------------------------------------
# 7.  VLM error taxonomy (backs the qualitative categories with counts)
# --------------------------------------------------------------------------

def error_taxonomy(gold: Dict, sysout: Dict) -> Dict:
    keys = [k for k in gold if k in sysout]
    tax = Counter()
    for k in keys:
        g, s = gold[k]["text"], sysout[k]["text"]

        # 1. Fraktur flattening: gold marks it, system does not
        gf = len(re.findall(r"\\mathfrak", g))
        sf = len(re.findall(r"\\mathfrak", s))
        if gf > sf:
            tax["fraktur_unmarked"] += gf - sf

        # 2. superscript case-lowering
        gsup = re.findall(r"\^\{?\\?[a-zA-Z]", g)
        ssup_lower = len(re.findall(r"\^\{?\\?mathrm\{?[a-z]", s)) + \
            len(re.findall(r"\^\{[a-z]", s)) + len(re.findall(r"\^[a-z]\b", s))
        gsup_lower = len(re.findall(r"\^\{?\\?mathrm\{?[a-z]", g)) + \
            len(re.findall(r"\^\{[a-z]", g)) + len(re.findall(r"\^[a-z]\b", g))
        if ssup_lower > gsup_lower:
            tax["superscript_case_lowered"] += ssup_lower - gsup_lower

        # 3. markup pathologies
        if s.count("$") % 2:
            tax["unbalanced_math_delim"] += 1
        if s.count("{") != s.count("}"):
            tax["unbalanced_braces"] += 1
        if "<sup>" in s and s.count("<sup>") != s.count("</sup>"):
            tax["unclosed_html_sup"] += 1
        tax["stacked_carets"] += len(re.findall(r"\}\s*\^\{", s))

        # 4. glyph confusions the classifier is designed to fix
        tax["sigma_as_sum"] += len(re.findall(r"\\sum", s))
        tax["phi_lowercase"] += len(re.findall(r"\\phi", s))
        tax["O_as_circ"] += len(re.findall(r"\\circ", s))
        tax["O_as_zero"] += len(re.findall(r"\^\{?0\}?", s)) - \
            len(re.findall(r"\^\{?0\}?", g))
        tax["para_as_eta"] += len(re.findall(r"\\eta", s))
    return {k: v for k, v in sorted(tax.items()) if v > 0}


# --------------------------------------------------------------------------
# 8.  Reporting
# --------------------------------------------------------------------------

def fmt_table(rows: List[List[str]], header: List[str]) -> str:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows
              else len(header[i]) for i in range(len(header))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    out = [line, "-" * len(line)]
    for r in rows:
        out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def report(res: Dict) -> str:
    o: List[str] = []
    a = o.append
    a("=" * 72)
    a("APPARATUS TRANSCRIPTION EVALUATION")
    a("=" * 72)

    systems = [(n, res[n]) for n in ("vlm", "pipe") if n in res]
    cfg = res.get("config", {})
    a("")
    a("Alignment: %s   |   units compared: %d"
      % (cfg.get("alignment", "?"), systems[0][1]["n_regions"]))
    a("Gold witnesses: %d   Gold witness groups: %d"
      % (systems[0][1]["witness_tp"] + systems[0][1]["witness_fn"],
         systems[0][1]["n_groups"]))
    a("")
    a("-- Table A: overall ------------------------------------------------")
    hdr = ["metric"] + [n for n, _ in systems]
    rows = []
    for label, key in [
        ("CER (%)", "cer"),
        ("witness precision (%)", "witness_precision"),
        ("witness recall (%)", "witness_recall"),
        ("witness F1 (%)", "witness_f1"),
        ("witness group exact (%)", "group_exact_match"),
        ("editorial mark F1 (%)", "mark_f1"),
    ]:
        rows.append([label] + ["%.2f" % s[key] for _, s in systems])
    rows.append(["witnesses missed (FN)"] + [str(s["witness_fn"]) for _, s in systems])
    rows.append(["witnesses spurious (FP)"] + [str(s["witness_fp"]) for _, s in systems])
    a(fmt_table(rows, hdr))

    a("")
    a("-- Table B: retrieval accuracy by witness-group length -------------")
    lens = sorted({int(k) for _, s in systems for k in s["length_buckets"]})
    rows = []
    for L in lens:
        row = ["%d%s" % (L, "+" if L == 12 else "")]
        for _, s in systems:
            b = s["length_buckets"].get(str(L))
            row.append("%.1f (%d/%d)" % (b["accuracy"], b["recovered"], b["total"])
                       if b else "-")
        rows.append(row)
    a(fmt_table(rows, ["group len"] + [n for n, _ in systems]))

    if "three_way" in res:
        t = res["three_way"]
        a("")
        a("-- Table C: three-way regression analysis --------------------------")
        a("Every gold witness, cross-tabulated by correctness in each system.")
        a("")
        rows = [
            ["correct in VLM, correct in pipeline (preserved)", str(t["preserved"])],
            ["wrong in VLM,   correct in pipeline (REPAIRED)", str(t["repaired"])],
            ["correct in VLM, wrong in pipeline (FALSE OVERRIDE)", str(t["regressed"])],
            ["wrong in VLM,   wrong in pipeline (unfixed)", str(t["unfixed"])],
        ]
        a(fmt_table(rows, ["cell", "n"]))
        a("")
        a("  gold witnesses            : %d" % t["gold_witnesses"])
        a("  repair rate               : %.2f%% of VLM errors corrected"
          % t["repair_rate_pct"])
        a("  FALSE-OVERRIDE RATE       : %.2f%% of correct VLM readings destroyed"
          % t["false_override_rate_pct"])
        a("  net witnesses gained      : %+d" % t["net_gain"])
        a("  spurious insertions       : %d" % t["spurious_insertions"])
        if t["regressed_examples"]:
            a("")
            a("  sample false overrides:")
            for e in t["regressed_examples"][:10]:
                a("    region %-8s  %s" % (e["region"], e["witness"]))

    if "taxonomy_vlm" in res and res["taxonomy_vlm"]:
        a("")
        a("-- Table D: VLM error categories (counts) --------------------------")
        a(fmt_table([[k, str(v)] for k, v in res["taxonomy_vlm"].items()],
                    ["category", "n"]))
    a("")
    return "\n".join(o)


# --------------------------------------------------------------------------
# 9.  CLI
# --------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--pipe", default=None,
                    help="full pipeline output; enables the regression table")
    ap.add_argument("--align", choices=("auto", "region", "verse"), default="auto",
                    help="how to match units across files. 'region' needs shared "
                         "page/region IDs; 'verse' anchors on verse numbers in "
                         "the content and works across different segmentations. "
                         "Default 'auto' tries region then falls back to verse.")
    ap.add_argument("--inspect", action="store_true",
                    help="report the structure detected in each file and exit")
    ap.add_argument("--loose-fraktur", action="store_true",
                    help="also treat unmarked lowercase runs over the Fraktur "
                         "alphabet as Fraktur witnesses (noisy; off by default)")
    ap.add_argument("--json", default=None, help="write full results as JSON")
    args = ap.parse_args(argv)

    if args.inspect:
        inspect(args.gold, "gold")
        inspect(args.vlm, "vlm")
        if args.pipe:
            inspect(args.pipe, "pipe")
        return 0

    docs: "Dict[str, OrderedDict]" = {
        "gold": parse_regions(args.gold),
        "vlm": parse_regions(args.vlm),
    }
    if args.pipe:
        docs["pipe"] = parse_regions(args.pipe)

    scheme, docs = choose_alignment(docs, args.align)

    gold, vlm = docs["gold"], docs["vlm"]
    pipe = docs.get("pipe")

    # Every system must be scored on the SAME unit set, otherwise the
    # columns of Table A are not comparable.
    common = [k for k in gold if k in vlm and (pipe is None or k in pipe)]
    if not common:
        raise SystemExit(
            "error: no units shared between the files under '%s' alignment.\n"
            "Run with --inspect to see what structure was detected in each file."
            % scheme
        )
    dropped = len(gold) - len(common)
    if dropped:
        print("note: %d/%d gold units absent from at least one system file; "
              "scoring on the %d shared units."
              % (dropped, len(gold), len(common)), file=sys.stderr)
    gold = OrderedDict((k, gold[k]) for k in common)
    vlm = OrderedDict((k, vlm[k]) for k in common)
    if pipe is not None:
        pipe = OrderedDict((k, pipe[k]) for k in common)

    res: Dict = {
        "vlm": evaluate_system(gold, vlm, args.loose_fraktur),
        "taxonomy_vlm": error_taxonomy(gold, vlm),
        "config": {"loose_fraktur": args.loose_fraktur, "alignment": scheme,
                   "n_units": len(common)},
    }
    if pipe is not None:
        res["pipe"] = evaluate_system(gold, pipe, args.loose_fraktur)
        res["three_way"] = three_way(gold, vlm, pipe, args.loose_fraktur)

    print(report(res))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        print("wrote %s" % args.json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())