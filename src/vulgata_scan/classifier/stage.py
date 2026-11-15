"""Stage 2: preprocessing, apparatus-zone selection, segmentation and CNN decoding.

Pipeline per page: apparatus zone selection (Desunt anchor + column fit,
printed rules as a protection layer) -> region deskew + crop -> line and token
segmentation -> per-token CNN decoding (decode.py) -> one region/token report
consumed by the reconciliation stage (vulgata_scan.reconcile).

Inputs are the multi-page stage-1 artifact (``model_pages`` + ``page_sizes``)
plus the native page renders.  Per-region debug PNGs are opt-in via ``debug``;
the page-level zones PNG is always written.

Adapted unchanged from ``pipeline_final_code/preprocessing/run_real_page.py``
of the paper's codebase; only the imports and the chunk-level driver are new.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .apparatus_filter import ZONES, assign_zone, desunt_line_index, detect_rules, \
    filter_apparatus_regions, is_apparatus_text, region_band_gaps
from .decode import decode_token
from .infer import SiglaClassifier
from .segment import Component, Token, crop_region_from_native_page, crop_token_image, \
    ink_mask, measure_skew, segment_line_into_tokens, segment_lines
from .taxonomy import load_taxonomy

import numpy as np

ZONE_COLORS = {"main": (120, 125, 135), "apparatus": (22, 163, 74),
               "footnote": (234, 88, 12)}


def draw_zone_overview(native_page: Image.Image, rule_ys: list[int],
                       gaps: list[tuple[int, int]], vis: list[tuple[int, list[float]]],
                       zone_sub: dict, out_path: Path, note: str | None = None,
                       selected: set[int] | None = None) -> None:
    """One PNG per page showing what the zoning decided, with
    tinted zone bands, the detected rules, the region-gap cross-check as a right sidebar,
    and a green box around each FINAL selected region (witness line drawn thick).
    Drawn from the live decision data, so a fallback page looks different by construction.
    """
    im = native_page.convert("RGB")
    w, h = im.size
    tint = im.copy()
    td = ImageDraw.Draw(tint)
    if len(rule_ys) == 2:
        for (y0, y1), col in zip([(0, rule_ys[0]), (rule_ys[0], rule_ys[1]), (rule_ys[1], h)],
                                 [ZONE_COLORS["main"], ZONE_COLORS["apparatus"], ZONE_COLORS["footnote"]]):
            td.rectangle([0, y0, w, y1], fill=col)
        im = Image.blend(im, tint, 0.16)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 44)
        font_s = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 34)
    except OSError:
        font = font_s = ImageFont.load_default()
    for ry in rule_ys:
        draw.line([(0, ry), (w, ry)], fill=(220, 38, 38), width=6)
        draw.text((16, ry + 10), f"rule y={ry}", fill=(220, 38, 38), font=font)
    for g0, g1 in gaps:  # layout-gap cross-check sidebar
        draw.rectangle([w - 44, g0, w - 10, g1], outline=(37, 99, 235), width=5)
    green = (22, 163, 74)
    for idx, bbox in vis:
        if selected is not None and idx not in selected:
            continue
        witness = zone_sub.get(idx) == "witness_line"
        draw.rectangle(bbox, outline=green, width=10 if witness else 6)
        label = f"r{idx}" + (" WITNESS" if witness else "")
        draw.text((bbox[0] + 8, bbox[1] + 4), label, fill=green, font=font_s)
    header = note or ("zones: main / apparatus / footnote; red = rules, "
                      "blue sidebar = layout gaps")
    draw.text((16, 12), header, fill=(17, 17, 17), font=font)
    scale = 1400 / w
    im.resize((1400, round(h * scale)), Image.LANCZOS).save(out_path)


def load_model_json_page(model_json_path: Path, page_no: int,
                         page_size: tuple[int, int]) -> dict:
    """Adapter for the multi-page model.json (a list of pages with norm-1000 bboxes),
    returning one page in the single-page shape ({page_image_size, regions:[...]}).
    norm-1000 * page_size/1000 reproduces the page-pixel bboxes to within a pixel (verified).
    """
    pages = json.loads(Path(model_json_path).read_text(encoding="utf-8"))
    return page_from_model_pages(pages, page_no, page_size)


def page_from_model_pages(pages: list, page_no: int,
                          page_size: tuple[int, int]) -> dict:
    """One page of a stage-1 ``model_pages`` structure, in the single-page shape
    consumed by :func:`run` (``{page_image_size, regions: [...]}``)."""
    w, h = page_size
    regions = []
    for r in pages[page_no]:
        x0, y0, x1, y1 = r["bbox_norm_1000"] if "bbox_norm_1000" in r else r["bbox_2d"]
        regions.append({"index": r["index"], "label": r.get("label"),
                        "content": r.get("content", ""),
                        "bbox": [x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000]})
    return {"page_image_size": [w, h], "regions": regions}


def run(glm_output: dict, native_page_path: Path, clf: SiglaClassifier, out_dir: Path,
        page_no: int = 0, debug_dir: Path | None = None,
        zones: frozenset[str] = frozenset({"apparatus"}),
        debug: bool = False) -> list[dict]:
    # The zones argument selects which page bands are processed. The default is the
    # middle apparatus only (the parsing target); pass {"apparatus","footnote"} to
    # also correct the bottom footnote-style apparatus.
    # When debug is True, an annotated per-region debug PNG is written next to the
    # zones PNG; the zones PNG itself is always written (make_report embeds it).
    debug_dir = debug_dir or out_dir
    debug_dir.mkdir(parents=True, exist_ok=True)
    tax = load_taxonomy()
    modifier_classes = [*clf.ls.superscripts, *clf.ls.trailing_numerals, "*", "†"]
    fraktur_classes = list(clf.ls.fraktur_letters)

    native_page = Image.open(native_page_path).convert("L")

    # bbox scale derived from the actual image sizes rather than a hardcoded DPI ratio
    glm_w, _ = glm_output["page_image_size"]
    scale_to_native = native_page.width / glm_w

    # The primary selection path is the Desunt witness-line anchor plus two-column fit,
    # driven entirely by GLM content and bboxes.
    # The printed rules act as a protection layer that vetoes regions outside the ruled middle band.
    # The legacy rules -> layout-gaps -> content-filter pipeline is the fallback
    # when no witness line is readable.
    bboxes_native = [[v * scale_to_native for v in r["bbox"]] for r in glm_output["regions"]]
    # Page-level deskew (skew reaches ~1 deg in this corpus), measured before selection
    # so that region crops, the classifier's geometry and detect_rules all consume the same angle.
    skew_deg = measure_skew(native_page)
    if abs(skew_deg) >= 0.1:
        print(f"NOTE p{page_no + 1}: page skew {skew_deg:+.2f} deg -- region crops deskewed")
    rule_ys = detect_rules(native_page, deskew_deg=skew_deg)
    gaps = region_band_gaps(bboxes_native)
    zone_note = None
    zone_sub: dict[int, str] = {}
    W = native_page.width
    geo = list(zip(glm_output["regions"], bboxes_native))
    zone_vis = [(r["index"], b) for r, b in geo]
    yc = lambda b: (b[1] + b[3]) / 2
    hits = [(r, b, li) for r, b in geo
            if (li := desunt_line_index(r.get("content", ""))) is not None]
    if hits:
        # ---- PRIMARY: Desunt anchor + column fit ----
        # The witness line is the edition's standing header (present on every calibrated page).
        # Everything above it is first-apparatus carry-over;
        # width is NO criterion there, the observed ranges overlap.
        # Below it, the second apparatus is the only two-column matter;
        # footnote blocks are full-measure and are cut by the per-page gutter test.
        anchor, a_bbox, a_line = min(hits, key=lambda h: yc(h[1]))
        zone_sub[anchor["index"]] = "witness_line"
        if a_line:
            print(f"WARNING p{page_no + 1} r{anchor['index']}: witness line is line "
                  f"{a_line + 1} of a merged region -- carry-over text above it "
                  f"is retained, page needs review")
        below = [(r, b) for r, b in geo if yc(b) > yc(a_bbox) and r is not anchor]
        # Right-column starters open at >= 0.46 W, left-column boxes end at <= 0.28 W (measured),
        # so 0.40 W splits with wide margin.
        right_x0 = [b[0] for _, b in below if b[0] >= 0.40 * W]
        gutter = min(right_x0) if right_x0 else None
        if gutter is None:
            print(f"WARNING p{page_no + 1}: no right-column start below the witness "
                  f"line -- column fit unavailable, keeping everything below the "
                  f"anchor, page needs review")
        sel = [(anchor, a_bbox)]  # anchor exempt from the column test (a witness list can be full-measure)
        for r, b in below:
            if re.fullmatch(r"\d+", r.get("content", "").strip()):
                continue  # signature/page numbers at the page foot ('10', '11')
            if gutter is not None and b[0] < 0.40 * W and b[2] > gutter:
                zone_sub[r["index"]] = "carry_over_dropped"
                print(f"NOTE p{page_no + 1} r{r['index']}: full-measure block below "
                      f"the witness line -- footnote apparatus, dropped: "
                      f"{r.get('content', '')[:50]!r}")
                continue
            sel.append((r, b))
        # ---- PROTECTION: printed-rules cross-check (veto) ----
        if len(rule_ys) == 2:
            y1r, y2r = rule_ys
            if not (y1r < yc(a_bbox) < y2r):
                print(f"WARNING p{page_no + 1}: witness line outside the ruled middle "
                      f"band (rules y={rule_ys}) -- page needs review")
            kept = []
            for r, b in sel:
                if r is not anchor and not (y1r < yc(b) < y2r):
                    zone_sub[r["index"]] = "carry_over_dropped"
                    print(f"WARNING p{page_no + 1} r{r['index']}: selected region "
                          f"outside the ruled middle band (rules y={rule_ys}) -- "
                          f"vetoed, page needs review")
                else:
                    kept.append((r, b))
            sel = kept
        else:
            print(f"WARNING p{page_no + 1}: {len(rule_ys)} divider rules detected -- "
                  f"selection runs on the Desunt+columns primary alone, unchecked")
        zoned = ([(r, "apparatus") for r, _ in sel] if "apparatus" in zones else [])
        if "footnote" in zones and len(rule_ys) == 2:
            zoned += [(r, "footnote") for r, b in geo if yc(b) > rule_ys[1]
                      and not re.fullmatch(r"\d+", r.get("content", "").strip())]
        zone_note = ("DESUNT+COLUMNS primary; rules cross-check "
                     + ("ok" if len(rule_ys) == 2 else f"unavailable ({len(rule_ys)} rules)"))
        draw_note = zone_note
    else:
        # ---- rule diagnostics, then the legacy FALLBACK selection ----
        if len(rule_ys) == 2:
            for ry in rule_ys:
                # A +-20 px tolerance is needed because rule ys are in the deskewed frame
                # while gaps come from unrotated GLM bboxes, so up to ~13 px of skew
                # disparity is expected.
                if not any(g0 - 20 <= ry <= g1 + 20 for g0, g1 in gaps):
                    print(f"WARNING p{page_no + 1}: rule at y={ry} falls inside a layout "
                          f"region, not a region gap {gaps} -- rules and layout disagree "
                          f"(merged region across the divider?), page needs review")
        else:
            cand = [g for g in gaps if (g[0] + g[1]) / 2 >= 0.3 * native_page.height]
            if len(rule_ys) == 1:
                # A hybrid stitch handles the case where only one rule was detected
                # (worn print broke the other).
                # The detected rule is trusted pixel evidence, so keep it
                # and take only the MISSING rule from the layout gaps.
                # Its height says which one it is (upper rules sit at 0.41-0.46 H, lower at 0.72-0.78 H),
                # so the complement is a candidate gap on the other side.
                ry = rule_ys[0]
                lower = ry >= 0.6 * native_page.height
                side = [g for g in cand
                        if ((g[0] + g[1]) / 2 < ry) == lower
                        and not (g[0] - 5 <= ry <= g[1] + 5)]
                if side:
                    # The widest gap wins because rule gaps carry the rule's own
                    # whitespace (>= 75 px) while paragraph-spacing gaps run ~55 px.
                    g0, g1 = max(side, key=lambda g: g[1] - g[0])
                    rule_ys = sorted([ry, (g0 + g1) // 2])
                    zone_note = (f"STITCHED: rule y={ry} detected, "
                                 f"{'upper' if lower else 'lower'} rule taken from "
                                 f"layout gap ({g0},{g1})")
                    print(f"WARNING p{page_no + 1}: {zone_note} -- page needs review")
            if len(rule_ys) != 2:
                if len(cand) == 2:
                    print(f"WARNING p{page_no + 1}: detected {len(rule_ys)} divider rules "
                          f"(expected 2) at {rule_ys} -- using region-gap pseudo-rules "
                          f"instead, page needs review")
                    rule_ys = [(g0 + g1) // 2 for g0, g1 in cand]
                    zone_note = "PSEUDO-RULES: no rules detected, both taken from layout gaps"
                else:
                    print(f"WARNING p{page_no + 1}: {len(rule_ys)} divider rules at {rule_ys} "
                          f"AND {len(cand)} candidate region gaps -- both geometric signals "
                          f"failed, falling back to the content filter, zones unknown")

        # ---- FALLBACK: legacy zone pipeline (rules -> gaps -> content scan).
        # Only reachable when GLM garbled the word 'Desunt' itself,
        # which never happened on the calibrated pages.
        print(f"WARNING p{page_no + 1}: no 'Desunt' witness line anywhere on the "
              f"page -- falling back to the legacy zone pipeline, page needs review")
        zoned = []
        if len(rule_ys) == 2:
            n_zone = dict.fromkeys(ZONES, 0)
            for r, bbox_n in geo:
                zone = assign_zone(bbox_n, rule_ys)
                n_zone[zone] += 1
                hit = is_apparatus_text(r.get("content", ""))
                if zone == "main":
                    if hit:
                        print(f"WARNING p{page_no + 1} r{r['index']}: sigla-like content in the "
                              f"main zone -- skipped; check rule detection: "
                              f"{r.get('content', '')[:60]!r}")
                    continue
                if zone not in zones:
                    continue  # tagged out by design (e.g. footnote band), not an anomaly
                if not hit:
                    print(f"NOTE p{page_no + 1} r{r['index']}: {zone} region without sigla-like "
                          f"content -- rescued by zoning (content filter alone would skip it)")
                for ry in rule_ys:
                    if bbox_n[1] < ry - 15 and bbox_n[3] > ry + 15:
                        print(f"WARNING p{page_no + 1} r{r['index']}: bbox straddles the divider "
                              f"rule at y={ry} -- zone by center may be wrong")
                zoned.append((r, zone))
            for z, n in n_zone.items():
                if n == 0:
                    print(f"WARNING p{page_no + 1}: no regions in the {z} zone -- "
                          f"zoning suspect, page needs review")
        else:
            zoned = [(r, None) for r in filter_apparatus_regions(glm_output)]
        # Without an anchor the carry-over cut falls back to width, the best remaining signal.
        apparatus = [(r, z) for r, z in zoned if z == "apparatus"]
        for r, _ in sorted(apparatus, key=lambda t: t[0]["bbox"][1]):
            if r["bbox"][2] - r["bbox"][0] >= 0.45 * glm_w:
                zone_sub[r["index"]] = "carry_over_dropped"
                print(f"NOTE p{page_no + 1} r{r['index']}: wide leading region "
                      f"dropped ({(r['bbox'][2] - r['bbox'][0]) / glm_w:.2f} W)")
            else:
                break
        zoned = [(r, z) for r, z in zoned
                 if zone_sub.get(r["index"]) != "carry_over_dropped"]
        draw_note = zone_note if len(rule_ys) == 2 else \
            "FALLBACK: rules not found -- content-filter selection, no zones"
    draw_zone_overview(
        native_page, rule_ys, gaps, zone_vis, zone_sub,
        debug_dir / f"p{page_no + 1}_zones.png",
        note=draw_note, selected={r["index"] for r, _ in zoned})
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(zoned)} regions selected (zones={sorted(zones)})"
          + (f" (rules at y={rule_ys})" if len(rule_ys) == 2 else ""))

    report = []
    for region, zone in zoned:
        region_idx = region["index"]
        region_crop = crop_region_from_native_page(native_page, tuple(region["bbox"]),
                                                   scale_to_native, deskew_deg=skew_deg)
        region_arr = np.array(region_crop)
        mask = ink_mask(region_arr)
        # 1.6x measured line height as the two-lines-bridged threshold
        max_line_h = round(1.6 * tax.measured_scale_px_native["full_line_height"])
        line_bands = segment_lines(mask, max_line_h=max_line_h)

        if debug:
            debug_img = region_crop.convert("RGB")
            draw = ImageDraw.Draw(debug_img)

        region_tokens = []
        for line_i, (y0, y1) in enumerate(line_bands):
            line_mask = mask[y0:y1]
            line_image = region_crop.crop((0, y0, region_crop.width, y1))
            tokens = segment_line_into_tokens(line_mask)

            results = []
            for token in tokens:
                token_crop = crop_token_image(line_image, token)
                # Measure the INK rather than the crop, because crop_token_image pads
                # 4 px a side, so a crop-size test never fires and 1-2 px scanner
                # specks reach the classifier.
                # A real | is 3-5 px wide, so < 3 px is only noise.
                if min(token.bbox[2] - token.bbox[0], token.bbox[3] - token.bbox[1]) < 3:
                    continue
                results.append((token, *decode_token(
                    clf, tax, line_image, token, token_crop, modifier_classes, fraktur_classes)))

            # Broken-glyph repair is needed because a cracked wide glyph (𝔴 above all)
            # prints as two components and segments into two tokens.
            # Merge each adjacent pair with a crack-sized gap and re-decode the union as ONE glyph.
            # At least one half must be background, but never both halves,
            # because a half can decode as a confident fraktur single
            # (a cracked 𝔴 half was observed reading as 𝔳 at 0.94),
            # while requiring one background half keeps genuine adjacent fraktur letters apart.
            def _half_ok(t, p, recon):
                if len(t.components) != 1:
                    return False
                return p["family"] == "background" or \
                    (p["family"] == "fraktur_cluster" and len(recon) == 1)

            i = 0
            while i < len(results) - 1:
                t1, p1, r1 = results[i][0], results[i][1], results[i][2]
                t2, p2, r2 = results[i + 1][0], results[i + 1][1], results[i + 1][2]
                gap = t2.bbox[0] - t1.bbox[2]
                n_bg = (p1["family"] == "background") + (p2["family"] == "background")
                if _half_ok(t1, p1, r1) and _half_ok(t2, p2, r2) and n_bg >= 1 \
                        and 0 <= gap <= 4:
                    merged = Token(
                        bbox=(min(t1.bbox[0], t2.bbox[0]), min(t1.bbox[1], t2.bbox[1]),
                              max(t1.bbox[2], t2.bbox[2]), max(t1.bbox[3], t2.bbox[3])),
                        components=t1.components + t2.components)
                    crop = crop_token_image(line_image, merged)
                    mpred = clf.predict(crop)
                    if mpred["family"] == "fraktur_cluster" and mpred["family_confidence"] >= 0.9:
                        g, conf = clf.predict_glyph(crop, allowed=fraktur_classes)
                        # Only 𝔴 is accepted, because it is the one glyph that physically
                        # prints in two parts. Relaxing to any fraktur letter read merged
                        # pairs of PLAIN letters as 𝔢/𝔞 at 0.9+ (25 false merges),
                        # so the letter itself is the gate.
                        if g == "w" and conf >= 0.8:
                            recon = tax.fraktur_char(g)
                            seq = [{"glyph": g, "confidence": round(conf, 3), "bbox": list(merged.bbox)}]
                            halves = "+".join(p["family"] for p in (p1, p2))
                            results[i:i + 2] = [(merged, mpred, recon, seq,
                                                 [f"broken-glyph merge: {halves} -> {recon} {conf:.2f}"])]
                            continue
                    # Ω gets a double gate because its bottom hairlines crack it like 𝔴.
                    # Accept a merged siglum only as a near-certain bare Ω
                    # corroborated by GLM reading an Ω in this very region,
                    # so a false merge can never introduce a base GLM never saw.
                    if mpred["family"] == "manuscript_siglum" \
                            and mpred.get("base") == "Ω" \
                            and mpred.get("family_confidence", 0.0) >= 0.99 \
                            and mpred.get("base_confidence", 0.0) >= 0.99 \
                            and ("Ω" in region.get("content", "")
                                 or "\\Omega" in region.get("content", "")):
                        halves = "+".join(p["family"] for p in (p1, p2))
                        results[i:i + 2] = [(merged, mpred, "Ω", [],
                                             [f"broken-Omega merge: {halves} "
                                              f"{mpred['base_confidence']:.2f}"])]
                        continue
                i += 1

            for tok_i, (token, pred, recon, mod_seq, warnings) in enumerate(results):
                abs_x0, abs_y0 = token.bbox[0], y0 + token.bbox[1]
                abs_x1, abs_y1 = token.bbox[2], y0 + token.bbox[3]
                for s in mod_seq:
                    # Region coords, same frame as bbox_in_region;
                    # downstream consumers cut crops straight from these.
                    s["bbox"] = [s["bbox"][0], y0 + s["bbox"][1],
                                 s["bbox"][2], y0 + s["bbox"][3]]
                if debug:
                    color = "red" if pred["family"] == "background" else "lime"
                    draw.rectangle([abs_x0, abs_y0, abs_x1, abs_y1], outline=color, width=1)
                    if pred["family"] != "background":
                        draw.text((abs_x0, max(0, abs_y0 - 12)), recon, fill="blue")

                region_tokens.append({
                    "line": line_i, "token": tok_i,
                    "bbox_in_region": [abs_x0, abs_y0, abs_x1, abs_y1],
                    "prediction": pred,
                    "modifier_sequence": mod_seq,
                    "decode_warnings": warnings,
                    "reconstructed_string": recon,
                })

        record = {
            "page": page_no + 1,
            "region_index": region_idx,
            "region_label": region.get("label"),
            "page_skew_deg": skew_deg,
            "zone": zone,
            **({"zone_sub": zone_sub[region_idx]} if region_idx in zone_sub else {}),
            "ocr_content": region.get("content"),
            "n_lines": len(line_bands),
            "tokens": region_tokens,
        }
        if debug:
            debug_path = debug_dir / f"p{page_no + 1}_region{region_idx:03d}_debug.png"
            debug_img.save(debug_path)
            record["debug_image"] = str(debug_path.name)
        report.append(record)
        print(f"p{page_no + 1} region {region_idx}: {len(line_bands)} lines, "
              f"{len(region_tokens)} tokens")

    return report


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best.pt"


def run_chunk(model_pages: list, page_sizes: list, native_pages: list,
              out_dir: Path, checkpoint: Path | str | None = None,
              debug: bool = False, debug_dir: Path | None = None,
              zones: frozenset[str] = frozenset({"apparatus"})) -> list[dict]:
    """Chunk-level stage-2 driver: segment + classify every page of a chunk.

    Library equivalent of the original ``run_real_page.py`` command.  Loads the
    frozen checkpoint once, then runs :func:`run` over each page.

    Args:
        model_pages: the pages-of-regions structure of the stage-1 artifact.
        page_sizes: ``[[width, height], ...]`` per page, index-aligned with
            ``model_pages`` and ``native_pages`` (the bboxes are norm-1000).
        native_pages: paths of the native 330 PPI grayscale renders,
            index-aligned with ``model_pages``.
        out_dir: output dir for the zones PNGs.
        checkpoint: path of the classifier checkpoint
            (defaults to the frozen checkpoint shipped with the package).
        debug: also write an annotated per-region debug PNG per page.
        debug_dir: where the debug images go (default ``<out_dir>/debug``).
        zones: which page bands to process (the parsing target is the middle
            apparatus band).

    Returns:
        The list of region/token records consumed by the reconciliation stage.
    """
    # Stage timing: classifier time = model load + every forward pass
    # (clf.predict / clf.predict_glyph are the only two inference entry points,
    # covering decode_token and the broken-glyph repair calls alike);
    # everything else in this process is preprocessing+segmentation.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = Path(debug_dir) if debug_dir else out_dir / "debug"

    t_start = time.perf_counter()
    clf = SiglaClassifier(str(checkpoint or DEFAULT_CHECKPOINT))
    clf_seconds = [time.perf_counter() - t_start]

    def _timed(fn):
        def wrap(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                clf_seconds[0] += time.perf_counter() - t0
        return wrap
    clf.predict = _timed(clf.predict)
    clf.predict_glyph = _timed(clf.predict_glyph)

    full_report = []
    for i, page_png in enumerate(native_pages):
        glm_output = page_from_model_pages(model_pages, i, tuple(page_sizes[i]))
        full_report += run(glm_output, Path(page_png), clf, out_dir, page_no=i,
                           debug_dir=debug_dir, zones=frozenset(zones), debug=debug)

    total = time.perf_counter() - t_start
    print(f"TIMING stage2 preprocessing+segmentation={total - clf_seconds[0]:.1f}s "
          f"classifier={clf_seconds[0]:.1f}s")
    return full_report

