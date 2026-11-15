#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apparatus_cer.py — CER between a critical-apparatus ground truth (LaTeX-ish)
and a pipeline output (Markdown with `## page N region M (type)` headers).

Two things it does:

  1. Page-aligned CER (and optional WER), after a normalisation that erases
     everything that is *LaTeX quality* rather than *content*:
       - font/style commands are stripped, argument kept
         (\\mathrm{G} == G, \\mathfrak{a} == a, \\textit{...} == ...)
       - Greek control sequences are mapped to unicode (\\Lambda == Λ)
       - $ { } and stray backslashes disappear, whitespace is collapsed
       - end-of-line hyphenation is re-joined (uni-\\nuerso -> uniuerso)
     Layout is dropped on both sides: `\\emph{Page N}` markers, `## page N
     region M (...)` headers, and every *Desunt* block (running notice of
     missing manuscripts).

  2. A cleaned pipeline file for the parser: same removals, LaTeX left intact,
     regions re-joined into one paragraph per apparatus entry (a new entry
     starts at `12.` / `II. 1.` ...), so entries broken across regions or
     across pages come back together.

Usage
-----
    python3 apparatus_cer.py --gt gt.tex --pred pipeline.md \
        --clean-out cleaned_for_parser.txt --per-page --diff 3

    # cleaning only
    python3 apparatus_cer.py --pred pipeline.md --clean-out cleaned.txt --no-score

rapidfuzz is used if installed (fast); otherwise a pure-python Levenshtein
is used, which is fine at page granularity.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import unicodedata
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# edit distance backend
# --------------------------------------------------------------------------
try:
    from rapidfuzz.distance import Levenshtein as _RF

    def edit_distance(a: str, b: str) -> int:
        return _RF.distance(a, b)

    _BACKEND = "rapidfuzz"
except ImportError:  # pragma: no cover
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

    _BACKEND = "pure-python"


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

# commands whose *argument* is content and whose *name* is pure presentation
STYLE_CMDS = [
    "mathrm", "mathit", "mathbf", "mathsf", "mathtt", "mathfrak", "mathcal",
    "mathbb", "frak", "textrm", "textit", "textbf", "textsf", "texttt",
    "textsc", "textnormal", "emph", "text", "mbox", "hbox", "operatorname",
    "underline", "textup", "textmd", "bm", "boldsymbol", "rm", "it", "bf",
    "sc", "sl", "cal", "up",
]
_STYLE_RE = re.compile(r"\\(?:%s)\s*\{" % "|".join(STYLE_CMDS))

# accent-ish commands: \c{e} -> e, \'e -> e, \"u -> u ...
_ACCENT_BRACED = re.compile(r"\\[cvuHrbdtk]\s*\{([^{}]*)\}")
_ACCENT_PLAIN = re.compile(r"\\['`^\"~=.]\s*\{?([A-Za-z])\}?")

GREEK = {
    "Alpha": "Α", "Beta": "Β", "Gamma": "Γ", "Delta": "Δ", "Epsilon": "Ε",
    "Zeta": "Ζ", "Eta": "Η", "Theta": "Θ", "Iota": "Ι", "Kappa": "Κ",
    "Lambda": "Λ", "Mu": "Μ", "Nu": "Ν", "Xi": "Ξ", "Omicron": "Ο",
    "Pi": "Π", "Rho": "Ρ", "Sigma": "Σ", "Tau": "Τ", "Upsilon": "Υ",
    "Phi": "Φ", "Chi": "Χ", "Psi": "Ψ", "Omega": "Ω",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "varphi": "φ",
    "varsigma": "ς", "vartheta": "θ",
}

SYMBOLS = {
    "mid": "|", "vert": "|", "|": "|",
    "sim": "~", "thicksim": "~", "backsim": "~",
    "P": "¶", "S": "§", "dagger": "†", "ddagger": "‡",
    "ast": "*", "star": "*", "cdot": ".", "ldots": "...", "dots": "...",
    "prime": "'", "circ": "°", "times": "*", "equiv": "=", "approx": "=",
    "leftarrow": "<-", "rightarrow": "->", "to": "->", "gets": "<-",
    "quad": " ", "qquad": " ", ",": " ", ";": " ", ":": " ", "!": "",
    "ge": ">=", "le": "<=", "neq": "!=", "pm": "+-", "cup": "u", "cap": "n",
}

# characters that are typographic noise once content is what we compare
DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
QUOTES = {ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"',
          ord("\u201d"): '"', ord("\u00ab"): '"', ord("\u00bb"): '"'}

_MD_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+.*$", re.M)
_MD_FENCE = re.compile(r"^\s*```.*$", re.M)
_PAGE_MARK = re.compile(r"\\emph\s*\{?\s*Page\s*\d+\s*[:}]*\s*\}?\s*:?", re.I)
_HYPHEN_BREAK = re.compile(r"([A-Za-zÀ-ÿ])-\s*\n\s*([a-zà-ÿ])")


def _strip_braced_style(s: str) -> str:
    """Remove \\mathrm{...} & friends, keeping the braced argument (nested-safe)."""
    while True:
        m = _STYLE_RE.search(s)
        if not m:
            return s
        i = m.end() - 1  # position of '{'
        depth, j = 0, i
        while j < len(s):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= len(s):  # unbalanced: drop the command name only
            s = s[: m.start()] + s[m.end() - 1:]
            continue
        s = s[: m.start()] + s[i + 1: j] + s[j + 1:]


def _map_commands(s: str) -> str:
    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name in GREEK:
            return GREEK[name]
        if name in SYMBOLS:
            return SYMBOLS[name]
        return " "  # unknown command: pure LaTeX noise
    s = re.sub(r"\\([A-Za-z]+)\s*", sub, s)
    # non-letter control sequences left over (\\, \$, \&, \%, \_, \{ ...)
    s = re.sub(r"\\(.)", lambda m: "" if m.group(1) in "\\ " else m.group(1), s)
    return s


_SUP_MERGE = re.compile(r"([\^_])\{([^{}]*)\}\s*\1\s*\{([^{}]*)\}")
_SPACE_BEFORE = re.compile(r"\s+([.,;:!?)\]}»])")
_SPACE_AFTER = re.compile(r"([(\[{«])\s+")
_HTML_SUP   = re.compile(r"<sup>(.*?)</sup>", re.I | re.S)
_HTML_SUB   = re.compile(r"<sub>(.*?)</sub>", re.I | re.S)
_HTML_STRAY = re.compile(r"</?su[bp]>", re.I)


def normalize(text: str, keep_case: bool = True, ignore_space: bool = False) -> str:
    """Content-only view of an apparatus chunk, for scoring."""
    s = text
    s = _HYPHEN_BREAK.sub(r"\1\2", s)
    s = _HTML_SUP.sub(r"^{\1}", s)
    s = _HTML_SUB.sub(r"_{\1}", s)
    s = _HTML_STRAY.sub("", s)   # drop any unbalanced leftover tag (e.g. Ψ<sup>d})
    s = _strip_braced_style(s)
    s = _ACCENT_BRACED.sub(r"\1", s)
    s = _ACCENT_PLAIN.sub(r"\1", s)
    s = _map_commands(s)
    # \Phi^{Z}^{\dagger}^{V} == \Phi^{Z\dagger V}: merge adjacent sup/sub groups
    prev = None
    while prev != s:
        prev = s
        s = _SUP_MERGE.sub(r"\1{\2\3}", s)
    s = s.replace("$", " ").replace("{", " ").replace("}", " ")
    s = unicodedata.normalize("NFKC", s)
    # strip combining marks left by NFKC/accents so ę == e, à == a
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if not unicodedata.combining(c))
    s = s.translate(DASHES).translate(QUOTES)
    if not keep_case:
        s = s.lower()
    # no space around superscript/subscript markers, so `G ^ C` == `G^C`
    s = re.sub(r"\s*([\^_])\s*", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    # spacing artefacts left by `$ X $` vs `X`
    s = _SPACE_BEFORE.sub(r"\1", s)
    s = _SPACE_AFTER.sub(r"\1", s)
    if ignore_space:
        s = re.sub(r"\s+", "", s)
    return s


def tokens(text: str) -> List[str]:
    return normalize(text).split()


# --------------------------------------------------------------------------
# parsing the two file formats into pages of blocks
# --------------------------------------------------------------------------

_DESUNT = re.compile(r"^\W*Desunt\b", re.I)
_PRED_HEADER = re.compile(
    r"^\s{0,3}#{1,6}\s*page\s*(\d+)\s*region\s*(\d+)\s*(?:\(([^)]*)\))?\s*$",
    re.I | re.M,
)
_GT_PAGE = re.compile(r"\\emph\s*\{\s*Page\s*(\d+)\s*\}?\s*:?\s*\}?", re.I)

ENTRY_START = re.compile(r"^\s*(?:[IVXLCDM]{1,6}\.\s*)?\d{1,3}\.\s")


def is_desunt(block: str) -> bool:
    probe = _strip_braced_style(block).lstrip()
    probe = re.sub(r"^[\\{}$\s]*", "", probe)
    return bool(_DESUNT.match(probe))


def _blocks(chunk: str) -> List[str]:
    """Split a page body into paragraph blocks (blank-line separated)."""
    chunk = _MD_FENCE.sub("", chunk)
    parts = re.split(r"\n\s*\n", chunk)
    return [p.strip() for p in parts if p.strip()]


def parse_gt(text: str) -> List[Tuple[int, List[Tuple[bool, str]]]]:
    """Ground truth -> ordered list of (printed label, [(is_desunt, block)]).

    *Desunt* blocks are kept here and filtered out at scoring time, so that the
    same parse can feed both the scores and a cleaned rendering.
    The label is informative only: it is *not* used to align anything, because
    the numbering restarts in the source.
    """
    pages: List[Tuple[int, List[Tuple[bool, str]]]] = []
    marks = list(_GT_PAGE.finditer(text))
    if not marks:
        return [(1, [(is_desunt(b), b) for b in _blocks(text)])]
    for k, m in enumerate(marks):
        pno = int(m.group(1))
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        body = text[m.end(): end]
        body = _PAGE_MARK.sub("", body)
        body = re.sub(r"\\\\", "\n", body)
        pages.append((pno, [(is_desunt(b), b) for b in _blocks(body)]))
    return pages


def parse_pred(text: str) -> List[Tuple[int, List[Tuple[int, str, str, bool]]]]:
    """Pipeline -> ordered list of (printed label, [(region, type, body, is_desunt)]).

    A new page starts whenever the page number in the `## page N region M`
    header changes; regions of one page are assumed contiguous in the file.
    Labels are kept for reporting only, never for alignment.
    """
    pages: List[Tuple[int, List[Tuple[int, str, str, bool]]]] = []
    marks = list(_PRED_HEADER.finditer(text))
    if not marks:
        body = _MD_HEADER.sub("", text)
        return [(1, [(0, "text", b, is_desunt(b)) for b in _blocks(body)])]
    last_label = None
    for k, m in enumerate(marks):
        pno, rno = int(m.group(1)), int(m.group(2))
        rtype = (m.group(3) or "text").strip()
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        raw = text[m.end(): end]
        raw = _MD_HEADER.sub("", raw)  # any stray markdown heading
        if pno != last_label:
            pages.append((pno, []))
            last_label = pno
        for b in _blocks(raw):
            pages[-1][1].append((rno, rtype, b, is_desunt(b)))
    return [p for p in pages if p[1]]


# --------------------------------------------------------------------------
# cleaning for the parser
# --------------------------------------------------------------------------

def light_fix(block: str) -> str:
    """Conservative repairs of pipeline noise. Spacing and sigla are left alone.

    Deliberately does NOT touch `$A$$B$` (two adjacent math groups is valid and
    frequent here); it only balances a trailing stray `$` when the block has an
    odd number of them.
    """
    s = block
    s = _HYPHEN_BREAK.sub(r"\1\2", s)
    s = _HTML_SUP.sub(r"^{\1}", s)
    s = _HTML_SUB.sub(r"_{\1}", s)
    s = _HTML_STRAY.sub("", s)   # drop any unbalanced leftover tag (e.g. Ψ<sup>d})
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)        # one physical line per block
    s = re.sub(r"[ \t]{2,}", " ", s)
    if s.count("$") % 2 == 1:                     # e.g. `$\Psi^{\mathrm{D}}$$.`
        s = re.sub(r"\$\$(?!\$)", "$", s, count=1) if "$$" in s else s
    return s.strip()


def clean_pred(pages: List[Tuple[int, List[Tuple[int, str, str]]]],
               merge_entries: bool = True,
               page_comments: bool = True,
               fix: bool = True) -> str:
    entries: List[List[str]] = []
    origin: List[Tuple[int, int]] = []          # (sequential index, printed label)
    last_content = None                          # index of the last non-Desunt entry
    for idx, (label, regions) in enumerate(pages, 1):
        for _rno, _rtype, body, des in regions:
            body = light_fix(body) if fix else body.strip()
            if not body:
                continue
            if des:
                # kept in the output, never merged with an apparatus entry and
                # never a merge target for the text that follows it
                entries.append([body])
                origin.append((idx, label))
                continue
            if merge_entries and last_content is not None \
                    and not ENTRY_START.match(body):
                entries[last_content].append(body)   # continuation of that entry
            else:
                entries.append([body])
                origin.append((idx, label))
                last_content = len(entries) - 1
    out: List[str] = []
    last = None
    for i, parts in enumerate(entries):
        if page_comments and origin[i] != last:
            idx, label = origin[i]
            tag = "%%%% page %d" % idx
            if label != idx:
                tag += " (printed label: %s)" % label
            out.append(tag)
            last = origin[i]
        out.append(" ".join(parts))
    return "\n\n".join(out) + "\n"


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

NORM_OPTS = {"keep_case": True, "ignore_space": False}


def page_text_gt(blocks: List[Tuple[bool, str]]) -> str:
    """Scoring view of a GT page: *Desunt* blocks excluded."""
    return normalize(" ".join(b for des, b in blocks if not des), **NORM_OPTS)


def page_text_pred(items: List[Tuple[int, str, str, bool]]) -> str:
    """Scoring view of a pipeline page: *Desunt* regions excluded."""
    return normalize(" ".join(b for _, _, b, des in items if not des), **NORM_OPTS)


def show_diff(ref: str, hyp: str, context: int = 28, limit: int = 25) -> None:
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    shown = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        shown += 1
        if shown > limit:
            print("      ... (more)")
            break
        left = ref[max(0, i1 - context): i1]
        right = ref[i2: i2 + context]
        print("      [%s] ...%s«%s»%s..." % (tag.upper(), left, ref[i1:i2], right))
        print("           %s>>%s<<%s" % (" " * len(left), hyp[j1:j2], ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", help="ground-truth file (LaTeX-ish, \\emph{Page N} markers)")
    ap.add_argument("--pred", required=True, help="pipeline output (markdown)")
    ap.add_argument("--clean-out", help="write the cleaned pipeline text here")
    ap.add_argument("--no-score", action="store_true", help="only clean, do not score")
    ap.add_argument("--per-page", action="store_true", help="per-page table")
    ap.add_argument("--align", choices=["order", "whole"], default="order",
                    help="order: pair the Nth page of each file (default; printed "
                         "page numbers are ignored, they are not comparable). "
                         "whole: score both files as a single stream.")
    ap.add_argument("--wer", action="store_true", help="also report token error rate")
    ap.add_argument("--diff", type=int, default=0, metavar="N",
                    help="show character-level diffs for the N worst pages")
    ap.add_argument("--lower", action="store_true", help="case-insensitive scoring")
    ap.add_argument("--ignore-space", action="store_true",
                    help="scoring: drop all whitespace (spacing inside sigla runs "
                         "then costs nothing)")
    ap.add_argument("--no-merge", action="store_true",
                    help="cleaning: keep one block per region instead of per entry")
    ap.add_argument("--no-page-comments", action="store_true",
                    help="cleaning: do not emit %%%% page N markers")
    ap.add_argument("--dump-norm", metavar="PREFIX",
                    help="write the normalised strings (PREFIX.gt.txt / PREFIX.pred.txt)")
    args = ap.parse_args()

    pred_raw = open(args.pred, encoding="utf-8").read()
    pred_pages = parse_pred(pred_raw)

    if args.clean_out:
        cleaned = clean_pred(pred_pages,
                             merge_entries=not args.no_merge,
                             page_comments=not args.no_page_comments)
        with open(args.clean_out, "w", encoding="utf-8") as fh:
            fh.write(cleaned)
        print("cleaned pipeline text -> %s (%d chars)" % (args.clean_out, len(cleaned)))

    if args.no_score or not args.gt:
        if not args.gt and not args.no_score:
            print("no --gt given: cleaning only", file=sys.stderr)
        return 0

    gt_raw = open(args.gt, encoding="utf-8").read()
    gt_pages = parse_gt(gt_raw)

    NORM_OPTS["keep_case"] = not args.lower
    NORM_OPTS["ignore_space"] = args.ignore_space

    # ---- alignment ------------------------------------------------------
    # Printed page numbers are NOT usable as keys (the GT numbering restarts),
    # so pages are paired by their position in the file, or merged into one
    # stream with --align whole.
    if args.align == "whole":
        units = [("all", page_text_gt([b for _, blocks in gt_pages for b in blocks]),
                  page_text_pred([r for _, regs in pred_pages for r in regs]))]
    else:
        if len(gt_pages) != len(pred_pages):
            print("!! page-count mismatch: %d in GT vs %d in the pipeline output.\n"
                  "   Pairing the first %d in file order; use --align whole to "
                  "score the two files as a single stream instead."
                  % (len(gt_pages), len(pred_pages),
                     min(len(gt_pages), len(pred_pages))), file=sys.stderr)
        units = []
        for i, ((glab, gblocks), (plab, pregs)) in enumerate(
                zip(gt_pages, pred_pages), 1):
            tag = "%d" % i
            if glab != plab:
                tag += " (gt:%s/pred:%s)" % (glab, plab)
            units.append((tag, page_text_gt(gblocks), page_text_pred(pregs)))

    rows, tot_d, tot_n, tot_wd, tot_wn = [], 0, 0, 0, 0
    norm_gt_all, norm_pred_all = [], []
    for tag, ref, hyp in units:
        norm_gt_all.append(ref)
        norm_pred_all.append(hyp)
        d = edit_distance(ref, hyp)
        tot_d += d
        tot_n += len(ref)
        cer = d / len(ref) if ref else float("nan")
        wer = float("nan")
        if args.wer:
            rt, ht = ref.split(), hyp.split()
            wd = edit_distance_tokens(rt, ht)
            tot_wd += wd
            tot_wn += len(rt)
            wer = wd / len(rt) if rt else float("nan")
        rows.append((tag, len(ref), len(hyp), d, cer, wer, ref, hyp))

    if args.per_page:
        print("\npage        len(gt)  len(pred)   dist      CER" +
              ("      WER" if args.wer else ""))
        for tag, lr, lh, d, cer, wer, *_ in rows:
            line = "%-11s %7d %10d %6d  %7.4f" % (tag, lr, lh, d, cer)
            if args.wer:
                line += "  %7.4f" % wer
            print(line)

    print("\nbackend: %s | pages compared: %d" % (_BACKEND, len(rows)))
    print("global CER = %d / %d = %.4f  (%.2f%%)" %
          (tot_d, tot_n, tot_d / tot_n if tot_n else float("nan"),
           100 * tot_d / tot_n if tot_n else float("nan")))
    if args.wer:
        print("global WER = %d / %d = %.4f  (%.2f%%)" %
              (tot_wd, tot_wn, tot_wd / tot_wn if tot_wn else float("nan"),
               100 * tot_wd / tot_wn if tot_wn else float("nan")))

    if args.diff:
        worst = sorted(rows, key=lambda r: r[4], reverse=True)[: args.diff]
        for p, _lr, _lh, _d, cer, _wer, ref, hyp in worst:
            print("\n--- page %s (CER %.4f) ---" % (p, cer))
            show_diff(ref, hyp)

    if args.dump_norm:
        with open(args.dump_norm + ".gt.txt", "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(norm_gt_all) + "\n")
        with open(args.dump_norm + ".pred.txt", "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(norm_pred_all) + "\n")
        print("normalised strings -> %s.{gt,pred}.txt" % args.dump_norm)
    return 0


def edit_distance_tokens(a: List[str], b: List[str]) -> int:
    """Levenshtein over token lists (for WER)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        cur = [i]
        for j, tb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = cur
    return prev[-1]


if __name__ == "__main__":
    sys.exit(main())