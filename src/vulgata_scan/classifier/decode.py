"""Two-pass token decoding: the classifier-side decision logic.

Pass 1 (`clf.predict` on the whole token crop) fixes the family and, for sigla, the base character.
Pass 2 (`clf.predict_glyph` per component, left to right) reads the modifier stack or fraktur letters
one glyph at a time, so the output preserves the printed sequence.
All confidence fallback rules live in this file.
"""
from __future__ import annotations

from .segment import Component, Token, crop_token_image

GLYPH_CONF_FLOOR = 0.7      # below this, a per-component decode is flagged with a warning
FRAKTUR_FALLBACK_CONF = 0.6  # below this, fall back to the cluster set head
REJECT_CONF_FLOOR = 0.5     # a reject below this is itself untrustworthy: re-decode without it


def compose_fraktur_cluster(letters: frozenset[str], tax) -> str:
    """Render a set-head prediction as a string in the taxonomy's fraktur alphabet order."""
    ordered = [l for l in tax.fraktur_letters if l in letters]
    return "".join(tax.fraktur_char(l) for l in ordered)


def _render_glyph(g: str, tax) -> str:
    if g in ("*", "†"):
        return g
    if g in ("1", "2", "3"):
        return tax.trailing_numeral_char(g)
    if g.islower():
        return tax.fraktur_char(g)
    return tax.superscript_char(g)


def _component_crop(line_image, comp, pad: int = 3):
    return crop_token_image(line_image, Token(bbox=comp.bbox, components=[comp]), pad=pad)


def _merge_overlapping(comps):
    """Merge components whose x-ranges overlap.
    Real adjacent superscripts never overlap horizontally,
    so an overlap means one broken glyph (typically a cracked '*'), not two."""
    merged = []
    for c in sorted(comps, key=lambda c: c.bbox[0]):
        if merged and c.bbox[0] < merged[-1].bbox[2]:
            p = merged[-1]
            merged[-1] = Component(bbox=(min(p.bbox[0], c.bbox[0]), min(p.bbox[1], c.bbox[1]),
                                         max(p.bbox[2], c.bbox[2]), max(p.bbox[3], c.bbox[3])),
                                   is_small=True)
        else:
            merged.append(c)
    return merged


def decode_token(clf, tax, line_image, token, token_crop, modifier_classes, fraktur_classes):
    """Decode one segmented token with the two-pass scheme.
    Returns (pred, recon, seq, warnings):
    the whole-token prediction dict, the reconstructed Unicode string ("" for background),
    the per-component decode sequence, and diagnostic warnings."""
    pred = clf.predict(token_crop)
    fam = pred["family"]
    warnings = []
    seq = []
    recon = ""

    if fam == "manuscript_siglum":
        mods = _merge_overlapping(c for c in token.components if c.is_small)
        tok_h = token.bbox[3] - token.bbox[1]
        base_c = next((c for c in token.components if not c.is_small), token.components[0])
        base_h = base_c.bbox[3] - base_c.bbox[1]
        for c in mods:
            crop = _component_crop(line_image, c)
            # Numerals sit higher than letter modifiers (bottom clears ~0.44 of the
            # base height vs ~0.28, measured), and letters never stack on letters,
            # so a high-band component can only be a numeral or '*'.
            raised_band = (base_c.bbox[3] - c.bbox[3]) >= 0.375 * base_h
            allowed_here = ["1", "2", "3", "*"] if raised_band else modifier_classes
            g, conf = clf.predict_glyph(crop, allowed=allowed_here)
            if g == "reject" and conf < REJECT_CONF_FLOOR:
                g, conf = clf.predict_glyph(crop, allowed=allowed_here, allow_reject=False)
                warnings.append(f"weak reject overridden -> {g} {conf:.2f}")
            elif g == "reject" and (c.bbox[3] - c.bbox[1]) >= 0.35 * tok_h:
                # Genuine punctuation fragments are small (under 0.25 of the token
                # height), so a modifier-sized reject is a degraded real glyph.
                g, conf = clf.predict_glyph(crop, allowed=allowed_here, allow_reject=False)
                warnings.append(f"large rejected component recovered -> {g} {conf:.2f}")
            seq.append({"glyph": g, "confidence": round(conf, 3), "bbox": list(c.bbox)})
            if g == "reject":
                # a wrongly attached punctuation fragment:
                # kept in the sequence for diagnosis, excluded from the reconstruction
                warnings.append(f"rejected attached component (conf {conf:.2f})")
            elif conf < GLYPH_CONF_FLOOR:
                warnings.append(f"low glyph confidence: {g} {conf:.2f}")
        # Low-confidence downgrade. A real word misframed as a siglum decodes as many
        # near-zero-confidence modifiers, while a genuine siglum has at most one
        # degraded component (measured), so we trigger on >= 2 near-zero components
        # or a majority under 0.5.
        n_zero = sum(1 for s in seq if s["glyph"] != "reject" and s["confidence"] < 0.1)
        n_low = sum(1 for s in seq if s["glyph"] == "reject" or s["confidence"] < 0.5)
        if n_zero >= 2 or (len(seq) >= 2 and n_low * 2 > len(seq)):
            pred["downgraded_from"] = fam
            pred["family"] = "background"
            warnings.append(f"low-confidence modifiers ({n_zero} near-zero, "
                            f"{n_low}/{len(seq)} low): downgraded to background")
            return pred, "", seq, warnings
        seq_kept = [s for s in seq if s["glyph"] != "reject"]
        if not seq_kept and tax.bases[pred["base"]].plain_latin:
            # A plain-Latin base with every component rejected is a bare capital,
            # which is ordinary text by definition, so downgrade and keep the OCR output.
            pred["downgraded_from"] = fam
            pred["family"] = "background"
            return pred, "", seq, warnings
        recon = pred["base"] + "".join(_render_glyph(s["glyph"], tax) for s in seq_kept)
        aux_set = set(pred["superscripts"])
        comp_set = {s["glyph"] for s in seq_kept if s["glyph"].isupper()}
        if aux_set != comp_set:
            warnings.append(f"aux set head {sorted(aux_set)} != per-component {sorted(comp_set)}")
    elif fam == "fraktur_cluster":
        comps = sorted(token.components, key=lambda c: c.bbox[0])
        ok = True
        for c in comps:
            crop = _component_crop(line_image, c)
            g, conf = clf.predict_glyph(crop, allowed=fraktur_classes)
            if g == "reject" and conf < REJECT_CONF_FLOOR:
                g, conf = clf.predict_glyph(crop, allowed=fraktur_classes, allow_reject=False)
                warnings.append(f"weak reject overridden -> {g} {conf:.2f}")
            seq.append({"glyph": g, "confidence": round(conf, 3), "bbox": list(c.bbox)})
            if g == "reject" or conf < FRAKTUR_FALLBACK_CONF:
                ok = False
        n_rej = sum(1 for s in seq if s["glyph"] == "reject")
        if n_rej * 2 > len(seq):
            # Majority reject means the family call itself is wrong (e.g. a connected
            # italic word), so downgrade to background and keep the OCR output.
            # A strict majority is required so that a real cluster with one degraded
            # component survives via the set-head fallback below.
            pred["downgraded_from"] = fam
            pred["family"] = "background"
            warnings.append(f"majority reject ({n_rej}/{len(seq)}): downgraded to background")
            return pred, "", seq, warnings
        seq_kept = [s for s in seq if s["glyph"] != "reject"]
        if ok and seq:
            recon = "".join(tax.fraktur_char(s["glyph"]) for s in seq)
        elif seq_kept and all(s["confidence"] >= FRAKTUR_FALLBACK_CONF for s in seq_kept):
            # Confident letters arrived with cleanly rejected junk (e.g. an attached
            # punctuation dot), so drop the junk and keep the letters.
            # Falling back to the set head here would let the junk's texture
            # inject a spurious letter into the cluster.
            recon = "".join(tax.fraktur_char(s["glyph"]) for s in seq_kept)
        else:  # touching letters merged into one blob, which the set head saw in training
            warnings.append("glyph head unsure; fell back to cluster set head")
            recon = compose_fraktur_cluster(frozenset(pred["letters"]), tax)
    elif fam == "declaratio_signorum":
        recon = pred["symbol"]
    elif fam == "latin_letter":
        recon = pred["letter"]

    return pred, recon, seq, warnings
