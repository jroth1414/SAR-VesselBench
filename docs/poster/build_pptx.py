"""Build the editable PowerPoint version of the AIPR poster (one 48 x 36 in slide).

The slide mirrors poster.tex: the same header, answer strip, four-column grid
(positions measured from the tikzposter PDF), section order, wording and
footer. Text, the detector pipeline and the protocol steps are native,
editable shapes; the two result figures are the 300-dpi PNGs that
``src.analysis.poster_figures`` writes beside the PDF versions. Every number
comes from ``generated/poster_numbers.json``; none is typed here.

Run from the repository root after generating the figures:
  python docs/poster/build_pptx.py
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
STATIC = HERE.parent / "class_report" / "figures" / "static"
OUT = HERE / "poster.pptx"

FONT = "Arial"  # metric-compatible with the PDF's Helvetica-class TeX Gyre Heros
NAVY = RGBColor(0x14, 0x21, 0x3D)
INK = RGBColor(0x1F, 0x23, 0x28)
MUTED = RGBColor(0x6B, 0x72, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RULE = RGBColor(0xB8, 0xBE, 0xC8)
ANSWER_BG = RGBColor(0xDC, 0xEF, 0xF8)
CITE = RGBColor(0x5B, 0x64, 0x77)
ROLE = {
    "floor": RGBColor(0x5D, 0x5D, 0x5D),
    "optical": RGBColor(0x9A, 0x67, 0x00),  # darker optical hue for text contrast
    "sar": RGBColor(0x00, 0x72, 0xB2),
    "imagenet": RGBColor(0x00, 0x9E, 0x73),
}
OPTICAL_LINE = RGBColor(0xC4, 0x87, 0x00)

# Matched to the PDF (body ~29.9 pt, headings 42.8 pt, captions 20.7 pt).
BODY_PT, SMALL_PT, FOOT_PT, REF_PT, HEAD_PT = 29, 22, 20, 17, 42

# Grid measured from poster.pdf: column (x, width) and section title tops.
COLUMNS = {1: (0.86, 9.63), 2: (11.20, 9.63), 3: (21.53, 15.27), 4: (37.51, 9.63)}
SECTION_H = 0.85  # heading band; the rule sits at its bottom edge


# --------------------------------------------------------------------------- primitives
def _rgb(fill, color: RGBColor) -> None:
    fill.solid()
    fill.fore_color.rgb = color


def _font(run, size: float, *, bold: bool = False, color: RGBColor = INK, italic: bool = False) -> None:
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def _list_marker(paragraph, xml: str, indent_in: float) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent_in))))
    pPr.set("indent", str(-int(Inches(indent_in))))
    pPr.append(parse_xml(f'<a:buClr {nsdecls("a")}><a:srgbClr val="14213D"/></a:buClr>'))
    pPr.append(parse_xml(f'<a:buFont {nsdecls("a")} typeface="Arial"/>'))
    pPr.append(parse_xml(xml))


def text(slide, x, y, w, h, paragraphs, *, size=BODY_PT, align=PP_ALIGN.LEFT, name="text",
         space_after=8, anchor=MSO_ANCHOR.TOP):
    """paragraphs: dicts with runs [(text, opts)] and optional bullet/numbered/refnum flags."""

    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    for index, spec in enumerate(paragraphs):
        para = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        para.alignment = spec.get("align", align)
        para.space_after = Pt(spec.get("space_after", space_after))
        if spec.get("bullet"):
            _list_marker(para, f'<a:buChar {nsdecls("a")} char="&#9632;"/>', 0.38)
            para._p.get_or_add_pPr().insert(1, parse_xml(f'<a:buSzPct {nsdecls("a")} val="70000"/>'))
        if spec.get("numbered"):
            _list_marker(para, f'<a:buAutoNum {nsdecls("a")} type="arabicPlain"/>', 0.45)
        if spec.get("refnum"):
            _list_marker(para, f'<a:buAutoNum {nsdecls("a")} type="arabicParenR"/>', 0.45)
        for chunk, opts in spec["runs"]:
            run = para.add_run()
            run.text = chunk
            _font(run, opts.get("size", spec.get("size", size)), bold=opts.get("bold", False),
                  color=opts.get("color", INK), italic=opts.get("italic", False))
    return box


def P(*runs, **kw):
    """Paragraph from runs: plain strings or (text, opts) tuples."""

    return {"runs": [(r, {}) if isinstance(r, str) else r for r in runs], **kw}


def B(t, **opts):
    return (t, {"bold": True, **opts})


def R(n: int):
    """Citation marker, styled like the PDF's \\refn."""

    return (f"[{n}]", {"color": CITE})


def section(slide, column: int, top: float, title: str):
    """Centered navy heading over a thin rule; returns (x, content_top, w)."""

    x, w = COLUMNS[column]
    text(slide, x, top, w, SECTION_H - 0.1, [P(B(title, size=HEAD_PT, color=NAVY), align=PP_ALIGN.CENTER,
                                              space_after=0)], name=f"heading: {title}", anchor=MSO_ANCHOR.BOTTOM)
    rule = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(top + SECTION_H),
                                      Inches(x + w), Inches(top + SECTION_H))
    rule.name = f"rule: {title}"
    rule.line.color.rgb = RULE
    rule.line.width = Pt(4.5)
    return x + 0.28, top + SECTION_H + 0.3, w - 0.56


def node(slide, x, y, w, h, label, *, fill, line, size=SMALL_PT, name="node",
         shape=MSO_SHAPE.ROUNDED_RECTANGLE, align=PP_ALIGN.CENTER):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.name = name
    _rgb(shp.fill, fill)
    shp.line.color.rgb = line
    shp.line.width = Pt(2)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for side in ("margin_left", "margin_right"):
        setattr(tf, side, Inches(0.12))
    lines = label if isinstance(label, list) else [label]
    for index, line_spec in enumerate(lines):
        para = tf.paragraphs[0] if index == 0 else tf.add_paragraph()
        para.alignment = align
        parts = line_spec if isinstance(line_spec, list) else [(line_spec, {})]
        for chunk, opts in parts:
            run = para.add_run()
            run.text = chunk
            _font(run, opts.get("size", size), bold=opts.get("bold", False), color=opts.get("color", INK))
    return shp


def arrow(slide, x1, y1, x2, y2, *, elbow: bool = False, name="arrow"):
    kind = MSO_CONNECTOR.ELBOW if elbow else MSO_CONNECTOR.STRAIGHT
    con = slide.shapes.add_connector(kind, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    con.name = name
    con.line.color.rgb = INK
    con.line.width = Pt(2.25)
    con.line._get_or_add_ln().append(parse_xml(f'<a:tailEnd {nsdecls("a")} type="triangle" w="med" len="med"/>'))
    return con


def picture(slide, path: Path, x, y, w, name):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    pic.name = name
    return pic, pic.height / 914400


def rect(slide, x, y, w, h, color: RGBColor, name: str):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.name = name
    _rgb(shp.fill, color)
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


# --------------------------------------------------------------------------- diagrams
def pipeline(slide, x: float, y: float, w: float) -> float:
    """Two-row detector diagram, native shapes (mirrors figures/detector_pipeline.tex)."""

    grey_fill, grey_line = RGBColor(0xF0, 0xF0, 0xF0), RGBColor(0x8C, 0x8C, 0x8C)
    blue_fill, blue_line = RGBColor(0xE6, 0xF1, 0xF7), RGBColor(0x00, 0x5B, 0x8E)
    neck_fill = RGBColor(0xF2, 0xF8, 0xFB)
    cal_fill = RGBColor(0xFC, 0xF1, 0xDB)
    end_fill, end_line = RGBColor(0xE0, 0xF4, 0xEE), RGBColor(0x00, 0x7E, 0x5C)
    row1, box_h = y + 0.4, 1.25
    node(slide, x, row1 + 0.35, 1.9, box_h, ["[VH, VV,", "VH−VV]", "512² crop"], fill=grey_fill, line=grey_line, name="input")
    enc_x, enc_w = x + 2.45, 3.6
    group = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(enc_x - 0.15), Inches(row1 - 0.05),
                                   Inches(enc_w + 0.3), Inches(2.15))
    group.name = "encoder group"
    group.fill.background()
    group.line.color.rgb = grey_line
    group.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    group.line.width = Pt(1.5)
    group.shadow.inherit = False
    text(slide, enc_x - 0.15, row1 - 0.45, enc_w + 0.3, 0.4, [P(("one encoder per track", {"size": 17, "color": MUTED}),
                                                                align=PP_ALIGN.CENTER)], name="encoder group label")
    node(slide, enc_x, row1 + 0.05, enc_w, 0.85, "ViT-B/16, stride 16", fill=blue_fill, line=blue_line, size=19, name="ViT encoder")
    node(slide, enc_x, row1 + 1.15, enc_w, 0.85, "ConvNeXt-V2-B, s32 + up", fill=blue_fill, line=blue_line, size=19, name="CNN encoder")
    head_x = enc_x + enc_w + 0.65
    head_w = x + w - head_x
    node(slide, head_x, row1 + 0.35, head_w, box_h, ["shared", "256-ch head"], fill=neck_fill, line=blue_line, name="head")
    arrow(slide, x + 1.9, row1 + 1.0, enc_x, row1 + 0.475, elbow=True, name="input to ViT")
    arrow(slide, x + 1.9, row1 + 1.0, enc_x, row1 + 1.575, elbow=True, name="input to CNN")
    arrow(slide, enc_x + enc_w + 0.15, row1 + 1.0, head_x, row1 + 1.0, name="encoders to head")
    row2 = row1 + 2.45
    seg_w = (w - 2 * 0.45) / 3
    boxes = [("F1, 200 m geo match", end_fill, end_line, "scoring"),
             ("threshold τ picked on dev", cal_fill, OPTICAL_LINE, "threshold"),
             ("heatmap peaks ≥ 0.05, 120 m NMS", neck_fill, blue_line, "decode")]
    for i, (label, fill, line, name) in enumerate(boxes):
        node(slide, x + i * (seg_w + 0.45), row2, seg_w, box_h, label, fill=fill, line=line, name=name)
    arrow(slide, head_x + head_w / 2, row1 + 0.35 + box_h, head_x + head_w / 2, row2, name="head to decode")
    arrow(slide, x + 2 * (seg_w + 0.45), row2 + box_h / 2, x + seg_w + 0.45 + seg_w, row2 + box_h / 2, name="decode to threshold")
    arrow(slide, x + seg_w + 0.45, row2 + box_h / 2, x + seg_w, row2 + box_h / 2, name="threshold to scoring")
    return row2 + box_h


# --------------------------------------------------------------------------- poster
def build(n: dict[str, str]) -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(48), Inches(36)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # header, answer strip, footer -----------------------------------------
    rect(slide, 0, 0, 48, 4.92, NAVY, "title bar")
    text(slide, 0.75, 0.35, 46.5, 2.5, [
        P(B("Label-Efficient Dark-Vessel Detection in SAR:", size=70, color=WHITE), align=PP_ALIGN.CENTER, space_after=0),
        P(B("Does SAR-Domain Pretraining Outperform Optical and ImageNet Transfer Across ViT and CNN?",
            size=70, color=WHITE), align=PP_ALIGN.CENTER, space_after=0),
    ], name="title")
    text(slide, 1.0, 2.95, 46.0, 0.8, [P(B("John Roth            Kyle Wagner            Liv d'Aliberti",
                                           size=44, color=WHITE), align=PP_ALIGN.CENTER)], name="authors")
    text(slide, 1.0, 3.85, 46.0, 0.6, [P(("Johns Hopkins University", {"size": 34, "color": WHITE}),
                                         align=PP_ALIGN.CENTER)], name="affiliation")
    rect(slide, 0.86, 5.70, 46.27, 1.04, ANSWER_BG, "answer strip")
    text(slide, 1.1, 5.70, 45.8, 1.04, [P(B("Short answer: not consistently. ", size=34),
        ("With 12 labeled scenes every pretrained encoder beats random initialization, but SAR pretraining beats "
         "optical only for the CNN, and only while labels are scarce.", {"size": 34}), align=PP_ALIGN.CENTER,
        space_after=0)], name="answer", anchor=MSO_ANCHOR.MIDDLE)
    rect(slide, 0, 34.85, 48, 1.15, NAVY, "footer bar")
    footer = {"size": 30, "color": WHITE}
    text(slide, 0.9, 34.85, 14, 1.15, [P(("Johns Hopkins University", footer), space_after=0)],
         name="footer left", anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 12, 34.85, 24, 1.15, [P(("Applied Imagery Pattern Recognition (AIPR) 2026", footer),
                                        align=PP_ALIGN.CENTER, space_after=0)], name="footer center", anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 33.1, 34.85, 14, 1.15, [P(("{jroth32, kwagne35, odalibe1}@jh.edu", footer), align=PP_ALIGN.RIGHT,
                                          space_after=0)], name="footer right", anchor=MSO_ANCHOR.MIDDLE)

    # column 1 ----------------------------------------------------------------
    x, y, w = section(slide, 1, 7.21, "Motivation")
    text(slide, x, y, w, 5.9, [
        P(B("Dark vessels"), " have no matching AIS report. Sentinel-1 SAR ", R(2),
          " sees them at night and through cloud.", bullet=True),
        P("Vessels fill a few pixels in a 25,000-pixel scene. Sea clutter and coastlines give strong false returns.",
          bullet=True),
        P(B("Human-verified labels are scarce."), " Pretraining can lower the label cost, but which source domain "
          "helps most, and does the answer depend on the backbone?", bullet=True),
    ], name="motivation")
    x, y, w = section(slide, 1, 14.60, "Data: xView3-SAR [1]")
    img_w = 0.82 * w
    _, img_h = picture(slide, STATIC / "scene_context.png", x + (w - img_w) / 2, y, img_w, "scene context")
    text(slide, x, y + img_h + 0.1, w, 0.6, [P(("One Sentinel-1 scene (about 300 × 220 km) and one 800-pixel chip "
                                               "(8 × 8 km).", {"size": FOOT_PT}), align=PP_ALIGN.CENTER)], name="scene caption")
    text(slide, x, y + img_h + 0.95, w, 26.0 - (y + img_h + 0.95), [
        P("150 frozen study scenes: ", B("111 train, 23 dev, 16 test"), ".", bullet=True),
        P("Nested budgets: ", B("12 ⊂ 28 ⊂ 56 ⊂ 111"), " training scenes (10, 25, 50, 100%).", bullet=True),
        P(B(f"{n['final_scenes']} near-shore, human-verified scenes"), ", opened once after test scoring: "
          f"{n['final_vessels']} vessels, {n['final_dark']} dark, {n['final_near_shore']} within 2 km of shore.",
          bullet=True),
        P("Input [VH, VV, VH−VV] in dB, 10 m pixels.", bullet=True),
    ], name="data bullets")
    x, y, w = section(slide, 1, 26.07, "References")
    refs = [
        "F. S. Paolo et al. xView3-SAR: Detecting dark fishing activity using SAR imagery. NeurIPS 2022.",
        "R. Torres et al. GMES Sentinel-1 mission. Remote Sens. Environ. 2012.",
        "A. Dosovitskiy et al. An image is worth 16×16 words. ICLR 2021.",
        "S. Woo et al. ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders. CVPR 2023.",
        "A. Steiner et al. How to train your ViT? TMLR 2022.",
        "J. Straka and I. Gruber. SatDINO: Self-supervised pretraining for remote sensing. arXiv:2508.21402, 2025.",
        "D. Liu et al. SARMAE: Masked autoencoder for SAR representation learning. CVPR 2026.",
        "K. N. Clasen et al. reBEN: Refined BigEarthNet dataset. IGARSS 2025.",
        "X. Zhou, D. Wang, and P. Krähenbühl. Objects as points. arXiv:1904.07850, 2019.",
        "T.-Y. Lin et al. Focal loss for dense object detection. ICCV 2017.",
    ]
    text(slide, x, y, w, 34.6 - y, [P((f"[{i}]  ", {"color": CITE, "size": REF_PT}), (ref, {"size": REF_PT}),
                                      space_after=3) for i, ref in enumerate(refs, 1)], name="references")

    # column 2 ----------------------------------------------------------------
    x, y, w = section(slide, 2, 7.32, f"8 arms × 4 budgets = {n['cells']} cells")
    rows = [("Role", "ViT-B/16 [3]", "ConvNeXt V2 [4]"),
            ("Random", "fresh init", "fresh init"),
            ("Optical", "SatDINO [6] (fMoW-RGB)", "BigEarthNet-S2 [8]"),
            ("SAR", "SARMAE [7] (SAR-1M)", "BigEarthNet-S1 [8]"),
            ("ImageNet", "AugReg [5]", "FCMAE [4]")]
    role_colors = [INK, ROLE["floor"], ROLE["optical"], ROLE["sar"], ROLE["imagenet"]]
    table = slide.shapes.add_table(len(rows), 3, Inches(x), Inches(y), Inches(w), Inches(4.4)).table
    table.columns[0].width, table.columns[1].width = Inches(2.25), Inches(3.3)
    table.columns[2].width = Inches(w - 5.55)
    table._tbl.tblPr.set("firstRow", "0")
    table._tbl.tblPr.set("bandRow", "0")
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.fill.background()
            cell.margin_left = cell.margin_right = Inches(0.06)
            run = cell.text_frame.paragraphs[0].add_run()
            run.text = value
            _font(run, BODY_PT - 3, bold=(c == 0 and r > 0), color=role_colors[r] if c == 0 else INK)
    text(slide, x, y + 4.7, w, 1.2, [P("Only the initialization changes inside a track; we compare domains ",
                                      B("within"), " an architecture.")], name="design note")
    x, y, w = section(slide, 2, 15.04, "One shared point detector")
    bottom = pipeline(slide, x, y, w)
    text(slide, x, bottom + 0.35, w, 26.3 - bottom - 0.35, [
        P("CenterNet-style heatmap ", R(9), " with focal loss ", R(10), "; F1 with 200 m geographic matching.",
          bullet=True),
        P(f"Same optimizer, schedule, crops, data order and seed (0) for every cell. Strict FP32: {n['gpu_hours']} "
          "H100 GPU-hours.", bullet=True),
        P("8 dev scenes pick each checkpoint and threshold; test and verified scenes are scored ", B("once"),
          ", with no retuning.", bullet=True),
    ], name="detector bullets")
    x, y, w = section(slide, 2, 26.43, "Why SARMAE dips at 28 scenes")
    _, img_h = picture(slide, GENERATED / "poster_sarmae_dynamics.png", x + 0.1, y, w - 0.2, "figure: SARMAE dynamics")
    text(slide, x, y + img_h + 0.12, w, 34.6 - (y + img_h + 0.12), [P((
        f"At 28 scenes SARMAE peaks as warm-up ends, then overfits (training loss {n['sarmae_f25_loss_first']} → "
        f"{n['sarmae_f25_loss_last']}; dev precision {n['sarmae_f25_dev_precision_first']} → "
        f"{n['sarmae_f25_dev_precision_last']}). Early stopping keeps the epoch-{n['sarmae_f25_best_epoch']} "
        "checkpoint.", {"size": FOOT_PT}))], name="SARMAE caption")

    # column 3 ----------------------------------------------------------------
    x, y, w = section(slide, 3, 7.33, "Pretraining helps most when labels are scarce")
    _, img_h = picture(slide, GENERATED / "poster_label_efficiency.png", x + 0.1, y, w - 0.2, "figure: label efficiency")
    text(slide, x, y + img_h + 0.15, w, 1.0, [P((
        f"F1 with 200 m matching. Each of the {n['cells']} cells is scored once with the threshold it picked on 8 dev "
        f"scenes. Top: {n['test_scenes']} held-out test scenes. Bottom: {n['final_scenes']} near-shore, "
        "human-verified scenes. Seed 0.", {"size": FOOT_PT}))], name="figure 1 caption")
    x, y, w = section(slide, 3, 20.47, "SAR vs. optical depends on backbone and budget")
    _, img_h = picture(slide, GENERATED / "poster_sar_minus_optical.png", x + 0.1, y, w - 0.2, "figure: SAR minus optical")
    text(slide, x, y + img_h + 0.15, w, 1.0, [P((
        f"Bars: 95% intervals from {n['bootstrap_resamples']} paired scene resamples. They cover scene sampling only: "
        f"retraining {n['rerun_cells']} unchanged cells moved test F1 by up to {n['rerun_max']}.",
        {"size": FOOT_PT}))], name="figure 2 caption")
    x, y, w = section(slide, 3, 29.57, "The verified scenes expose a coastal label gap")
    tiles = [
        (f"{n['train_near_shore_pct']} → {n['final_near_shore_pct']}",
         ["vessels within 2 km of shore:", "training labels → verified scenes"]),
        (f"{n['best_cell_near_shore_predictions']} / {n['best_cell_predictions']}",
         ["best cell's predictions", "within 2 km of shore"]),
        (n["best_cell_offshore_f1"], ["best cell's offshore verified F1", f"({n['best_cell_final']} overall)"]),
    ]
    tile_w = w / 3
    for i, (big, label) in enumerate(tiles):
        tx = x + i * tile_w
        text(slide, tx, y - 0.1, tile_w, 0.9, [P(B(big, size=52, color=NAVY), align=PP_ALIGN.CENTER, space_after=0)],
             name=f"stat {i + 1} value")
        text(slide, tx, y + 0.85, tile_w, 0.85, [P((line, {"size": FOOT_PT}), align=PP_ALIGN.CENTER, space_after=0)
                                                  for line in label], name=f"stat {i + 1} label")
    text(slide, x, y + 1.85, w, 0.9, [P((
        f"Every cell reaches at most {n['final_near_shore_recall_max']} near-shore recall and "
        f"{n['final_dark_recall_max']} dark-vessel recall, while precision stays at {n['final_precision_min']}–"
        f"{n['final_precision_max']}: the detectors learned to stay silent near shore.", {"size": FOOT_PT}))],
        name="stat note")

    # column 4 ----------------------------------------------------------------
    x, y, w = section(slide, 4, 7.33, "Findings")
    text(slide, x, y - 0.1, w, 1.1, [P(B(f"{n['gain12_ci_above_zero']} / {n['gain12_comparisons']}", size=60,
                                         color=ROLE["imagenet"]), align=PP_ALIGN.CENTER, space_after=0)],
         name="hero number")
    text(slide, x, y + 1.05, w, 1.1, [P("pretrained-vs-random comparisons at 12 training scenes have a 95% interval "
                                       "above zero.", align=PP_ALIGN.CENTER)], name="hero caption")
    text(slide, x, y + 2.45, w, 20.3 - (y + 2.45), [
        P(B("Pretraining pays off with few labels."), f" Gains at 12 scenes: {n['test_gain12_min']} to "
          f"{n['test_gain12_max']} test F1, {n['final_gain12_min']} to {n['final_gain12_max']} verified F1.",
          numbered=True, space_after=12),
        P(B("SAR beats optical only for the CNN, and only with few labels:"),
          f" {n['test_cnn_sar_minus_opt_10']} {n['test_cnn_sar_minus_opt_10_ci']} test F1 at 12 scenes, but "
          f"{n['final_cnn_sar_minus_opt_100']} {n['final_cnn_sar_minus_opt_100_ci']} verified F1 at 111. "
          "ViT: no stable order.", numbered=True, space_after=12),
        P(B("The input adapter matters."), " With a seeded 3-channel stem instead of band slicing, BigEarthNet-S2 "
          f"beats random at every budget ({n['cnn_optical_test_gain_min']} to {n['cnn_optical_test_gain_max']} "
          "test F1).", numbered=True, space_after=12),
        P(B("The coastal gap is a label gap."), f" Only {n['train_near_shore_pct']} of training vessels lie within "
          f"2 km of shore, against {n['final_near_shore_pct']} of verified ones.", numbered=True),
    ], name="findings")
    x, y, w = section(slide, 4, 20.36, "Limits")
    text(slide, x, y, w, 25.8 - y, [
        P("One seed per cell; 8 dev scenes pick checkpoints and thresholds.", bullet=True),
        P("Monotonicity gate failed: two SAR test curves drop by more than 0.02 as labels grow.", bullet=True),
        P("Checkpoints differ in objective and corpus, not only domain.", bullet=True),
    ], name="limits")
    x, y, w = section(slide, 4, 26.02, "Open questions & next steps")
    text(slide, x, y, w, 34.6 - y, [
        P("How do geospatial foundation models pretrained jointly on SAR and optical imagery fare against these "
          "single-source checkpoints?", bullet=True),
        P("Does SAR pretraining help where optical cannot: near shore and for small dark vessels?", bullet=True),
        P("Would per-arm learning rates or longer early-stopping patience remove the small-budget dips?", bullet=True),
        P(B("Next:"), " near-shore training labels, a shoreline input channel, several seeds, and matched pretraining "
          "objectives across sensors.", bullet=True),
    ], name="open questions")
    return prs


def main() -> int:
    payload = json.loads((GENERATED / "poster_numbers.json").read_text(encoding="utf-8"))
    prs = build(payload["display"])
    prs.core_properties.title = "Label-Efficient Dark-Vessel Detection in SAR (AIPR 2026 poster)"
    prs.core_properties.author = "John Roth, Kyle Wagner, Liv d'Aliberti"
    prs.save(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
