"""Build the editable PowerPoint version of the AIPR poster (one 48 x 36 in slide).

The slide mirrors the tikzposter grid in poster.tex (same columns, blocks and
order). Text, the detector pipeline and the protocol steps are native,
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

FONT = "Arial"
NAVY = RGBColor(0x14, 0x21, 0x3D)
INK = RGBColor(0x1F, 0x23, 0x28)
MUTED = RGBColor(0x6B, 0x72, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BODY_BG = RGBColor(0xF8, 0xF8, 0xF9)
CALLOUT_BG = RGBColor(0xFF, 0xF4, 0xDE)
CALLOUT_TITLE = RGBColor(0xC4, 0x87, 0x00)
ROLE = {
    "floor": RGBColor(0x5D, 0x5D, 0x5D),
    "optical": RGBColor(0xE6, 0x9F, 0x00),
    "sar": RGBColor(0x00, 0x72, 0xB2),
    "imagenet": RGBColor(0x00, 0x9E, 0x73),
}
TEXT_OPTICAL = RGBColor(0x9A, 0x67, 0x00)  # darker optical hue for text contrast

# Matched to the tikzposter PDF (Roboto body 29.9 pt, block titles 42.8 pt,
# captions 20.7 pt), one step smaller because Arial sets wider than Roboto.
BODY_PT, SMALL_PT, FOOT_PT, BLOCK_TITLE_PT, TITLE_PT = 27, 22, 20, 40, 60
TITLE_H = 0.95  # block title bar height (in)
PAD = 0.3  # block body padding (in)

# Grid measured from the tikzposter PDF: (x, width) per column, block (y, height).
COLUMNS = {1: (0.86, 9.63), 2: (11.20, 9.63), 3: (21.53, 15.27), 4: (37.51, 9.63)}


def _rgb(shape_fill, color: RGBColor) -> None:
    shape_fill.solid()
    shape_fill.fore_color.rgb = color


def block(slide, column: int, y: float, h: float, title: str, *, callout: bool = False):
    """Draw a titled block; return (x, y, w, h) of its content area in inches."""

    x, w = COLUMNS[column]
    body = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    body.adjustments[0] = 0.25 / min(w, h)  # ~0.25 in corner radius
    _rgb(body.fill, CALLOUT_BG if callout else BODY_BG)
    body.line.color.rgb = CALLOUT_TITLE if callout else NAVY
    body.line.width = Pt(1.5)
    body.shadow.inherit = False
    body.name = f"block: {title}"
    bar = slide.shapes.add_shape(MSO_SHAPE.ROUND_2_SAME_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(TITLE_H))
    bar.adjustments[0] = 0.25
    _rgb(bar.fill, CALLOUT_TITLE if callout else NAVY)
    bar.line.fill.background()
    bar.shadow.inherit = False
    bar.name = f"block title: {title}"
    tf = bar.text_frame
    tf.margin_left = Inches(PAD)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT
    run = tf.paragraphs[0].add_run()
    run.text = title
    _font(run, BLOCK_TITLE_PT, bold=True, color=WHITE)
    return x + PAD, y + TITLE_H + 0.25, w - 2 * PAD, h - TITLE_H - 0.45


def _font(run, size: float, *, bold: bool = False, color: RGBColor = INK, italic: bool = False,
          superscript: bool = False) -> None:
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    if superscript:
        run.font._element.set("baseline", "30000")


def _bullet(paragraph, char: str = "■", indent_in: float = 0.38) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent_in))))
    pPr.set("indent", str(-int(Inches(indent_in))))
    pPr.append(parse_xml(f'<a:buClr {nsdecls("a")}><a:srgbClr val="14213D"/></a:buClr>'))
    pPr.append(parse_xml(f'<a:buSzPct {nsdecls("a")} val="70000"/>'))
    pPr.append(parse_xml(f'<a:buFont {nsdecls("a")} typeface="Arial"/>'))
    pPr.append(parse_xml(f'<a:buChar {nsdecls("a")} char="{char}"/>'))


def _numbered(paragraph, indent_in: float = 0.45) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent_in))))
    pPr.set("indent", str(-int(Inches(indent_in))))
    pPr.append(parse_xml(f'<a:buClr {nsdecls("a")}><a:srgbClr val="14213D"/></a:buClr>'))
    pPr.append(parse_xml(f'<a:buFont {nsdecls("a")} typeface="Arial"/>'))
    pPr.append(parse_xml(f'<a:buAutoNum {nsdecls("a")} type="arabicPlain"/>'))


def text(slide, x, y, w, h, paragraphs, *, size=BODY_PT, align=PP_ALIGN.LEFT, name="text",
         space_after=8, anchor=MSO_ANCHOR.TOP):
    """paragraphs: list of dicts {runs: [(text, opts)], bullet|numbered: bool, size}."""

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
            _bullet(para)
        if spec.get("numbered"):
            _numbered(para)
        for chunk, opts in spec["runs"]:
            run = para.add_run()
            run.text = chunk
            _font(run, opts.get("size", spec.get("size", size)), bold=opts.get("bold", False),
                  color=opts.get("color", INK), italic=opts.get("italic", False),
                  superscript=opts.get("sup", False))
    return box


def P(*runs, **kw):
    """Paragraph from runs: plain strings or (text, opts) tuples."""

    return {"runs": [(r, {}) if isinstance(r, str) else r for r in runs], **kw}


def B(t, **opts):
    return (t, {"bold": True, **opts})


def node(slide, x, y, w, h, label, *, fill, line, size=SMALL_PT, bold_first=False, name="node",
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
        setattr(tf, side, Inches(0.08))
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
    ln = con.line._get_or_add_ln()
    ln.append(parse_xml(f'<a:tailEnd {nsdecls("a")} type="triangle" w="med" len="med"/>'))
    return con


def picture(slide, path: Path, x, y, w, name):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    pic.name = name
    return pic


def build(numbers: dict[str, str]) -> Presentation:
    n = numbers
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(48), Inches(36)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # ---------------------------------------------------------------- title bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(48), Inches(5.32))
    bar.name = "title bar"
    _rgb(bar.fill, NAVY)
    bar.line.fill.background()
    bar.shadow.inherit = False
    text(slide, 1.0, 0.45, 46.0, 2.6, [
        P(B("Label-Efficient Dark-Vessel Detection in SAR:", size=TITLE_PT, color=WHITE), align=PP_ALIGN.CENTER, space_after=0),
        P(B("Does SAR-Domain Pretraining Outperform Optical and ImageNet Transfer Across ViT and CNN?",
            size=TITLE_PT, color=WHITE), align=PP_ALIGN.CENTER, space_after=0),
    ], name="title")
    sup = {"size": 42, "color": WHITE, "sup": True}
    text(slide, 1.0, 3.35, 46.0, 0.8, [P(
        ("John Roth", {"size": 42, "color": WHITE}), ("1,2", sup),
        ("        Kyle Wagner", {"size": 42, "color": WHITE}), ("1,3", sup),
        ("        Liv d'Aliberti", {"size": 42, "color": WHITE}), ("1", sup),
        align=PP_ALIGN.CENTER)], name="authors")
    pale = RGBColor(0xDD, 0xE1, 0xE8)
    aff_sup = {"size": 30, "color": pale, "sup": True}
    aff = {"size": 30, "color": pale}
    text(slide, 1.0, 4.25, 46.0, 0.7, [P(
        ("1", aff_sup), ("Johns Hopkins University      ", aff), ("2", aff_sup), ("The MITRE Corporation      ", aff),
        ("3", aff_sup), ("Boeing – Phantom Works      {jroth32, kwagne35, odalibe1}@jh.edu", aff),
        align=PP_ALIGN.CENTER)], name="affiliations")

    # ---------------------------------------------------------------- column 1
    x, y, w, h = block(slide, 1, 6.11, 6.57, "Why it matters")
    text(slide, x, y, w, h, [
        P(B("Dark vessels"), " have no matching AIS report. Sentinel-1 SAR sees them at night and through cloud.", bullet=True),
        P("Vessels fill a few pixels in a 25,000-pixel scene. Sea clutter and coastlines give strong false returns.", bullet=True),
        P(B("Human-verified labels are scarce."), " Pretraining can lower the label cost, but which source domain helps most?", bullet=True),
    ], name="why")
    x, y, w, h = block(slide, 1, 13.30, 3.47, "Question")
    text(slide, x, y, w, h, [P(
        "Does ", B("SAR", color=ROLE["sar"]), " pretraining beat ", B("optical remote-sensing", color=TEXT_OPTICAL),
        " and ", B("ImageNet", color=ROLE["imagenet"]), " pretraining when labels are few? Does the answer hold for both a ViT and a CNN?")],
        name="question")
    x, y, w, h = block(slide, 1, 17.40, 11.91, "Data: xView3-SAR")
    pic = picture(slide, STATIC / "scene_context.png", x, y, w, "scene context")
    img_h = pic.height / 914400
    text(slide, x, y + img_h + 0.1, w, 0.6, [P((
        "One Sentinel-1 scene (about 300 × 220 km) and one 800-pixel chip (8 × 8 km).", {"size": FOOT_PT}))],
        name="scene caption")
    text(slide, x, y + img_h + 0.85, w, h - img_h - 0.85, [
        P("150 frozen study scenes: ", B("111 train, 23 dev, 16 test"), ".", bullet=True),
        P("Nested budgets: ", B("12 ⊂ 28 ⊂ 56 ⊂ 111"), " training scenes (10, 25, 50, 100%).", bullet=True),
        P(B(f"{n['final_scenes']} near-shore, human-verified scenes"), ", opened once after test scoring: "
          f"{n['final_vessels']} vessels, {n['final_dark']} dark, {n['final_near_shore']} within 2 km of shore.", bullet=True),
        P("Input [VH, VV, VH − VV] in dB, 10 m pixels.", bullet=True),
    ], name="data bullets")
    x, y, w, h = block(slide, 1, 29.94, 4.30, "Fix before the final cohort", callout=True)
    text(slide, x, y, w, h, [P(
        "BigEarthNet-S2 expects 10 optical bands. The first adapter kept 3 band kernels and scaled them by 10/3: "
        "first-layer weight norm ", B(n["stem_weight_norm_ratio"]), ". A seeded 3-channel stem plus every other "
        f"transferred tensor raised dev F1 from {n['stem_dev_before']} to {n['stem_dev_after']} (12 scenes). "
        "We then retrained ", B(f"all {n['cells']} cells"), " from initialization.")], name="stem fix")

    # ---------------------------------------------------------------- column 2
    x, y, w, h = block(slide, 2, 6.11, 9.17, f"8 arms × 4 budgets = {n['cells']} cells")
    rows = [("Role", "ViT-B/16 (86M)", "ConvNeXt-V2-B (89M)"),
            ("Random", "fresh init", "fresh init"),
            ("Optical", "SatDINO (fMoW-RGB)", "BigEarthNet-S2"),
            ("SAR", "SARMAE (SAR-1M)", "BigEarthNet-S1"),
            ("ImageNet", "AugReg", "FCMAE")]
    role_colors = [INK, ROLE["floor"], TEXT_OPTICAL, ROLE["sar"], ROLE["imagenet"]]
    table = slide.shapes.add_table(len(rows), 3, Inches(x), Inches(y), Inches(w), Inches(4.6)).table
    table.columns[0].width, table.columns[1].width, table.columns[2].width = Inches(1.85), Inches(3.4), Inches(w - 5.25)
    tbl_pr = table._tbl.tblPr
    tbl_pr.set("firstRow", "0")
    tbl_pr.set("bandRow", "0")
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.fill.background()
            cell.margin_left = cell.margin_right = Inches(0.06)
            para = cell.text_frame.paragraphs[0]
            run = para.add_run()
            run.text = value
            _font(run, BODY_PT - 1, bold=(c == 0 and r > 0), color=role_colors[r] if c == 0 else INK)
    text(slide, x, y + 4.85, w, h - 4.85, [P(
        "Only the initialization changes inside a track. We compare domains ", B("within"),
        " an architecture, then check whether the pattern repeats across the two.")], name="design note")

    x, y, w, h = block(slide, 2, 15.91, 9.25, "One shared point detector")
    pipeline(slide, x, y, w)
    text(slide, x, y + 4.15, w, h - 4.15, [
        P("CenterNet-style Gaussian heatmap and focal loss; F1 with 200 m geographic matching.", bullet=True),
        P(f"Same optimizer, schedule, crops, data order and seed (0) for every cell. Strict FP32: "
          f"{n['gpu_hours']} H100 GPU-hours.", bullet=True),
    ], name="detector bullets")

    x, y, w, h = block(slide, 2, 25.78, 6.44, "Once-only held-out protocol")
    steps = [
        (f"Train {n['cells']} cells.", " A sweep on 8 dev scenes picks each checkpoint and threshold.", RGBColor(0xF2, 0xF2, 0xF2), RGBColor(0x99, 0x99, 0x99)),
        ("Freeze the cohort.", " Hash-bind every marker and checkpoint.", RGBColor(0xEB, 0xF4, 0xFA), ROLE["sar"]),
        (f"Score {n['test_scenes']} test scenes once.", " Same thresholds, no retuning.", RGBColor(0xFC, 0xF1, 0xDB), CALLOUT_TITLE),
        (f"Open {n['final_scenes']} verified scenes once.", f" Same thresholds, all {n['cells']} cells.", RGBColor(0xE0, 0xF4, 0xEE), ROLE["imagenet"]),
    ]
    step_h, gap = 1.0, 0.32
    for i, (head, rest, fill, line) in enumerate(steps):
        sy = y + 0.05 + i * (step_h + gap)
        dot = node(slide, x + 0.2, sy + 0.2, 0.6, 0.6, [[(str(i + 1), {"bold": True, "color": WHITE, "size": SMALL_PT})]],
                   fill=NAVY, line=NAVY, shape=MSO_SHAPE.OVAL, name=f"step {i + 1} number")
        node(slide, x + 1.1, sy, w - 1.3, step_h, [[(head, {"bold": True}), (rest, {})]], fill=fill, line=line,
             name=f"step {i + 1}", align=PP_ALIGN.LEFT)
        if i < len(steps) - 1:
            arrow(slide, x + 1.1 + (w - 1.3) / 2, sy + step_h, x + 1.1 + (w - 1.3) / 2, sy + step_h + gap, name=f"step arrow {i + 1}")

    # ---------------------------------------------------------------- column 3
    x, y, w, h = block(slide, 3, 6.11, 14.09, "Pretraining helps most when labels are scarce")
    pic = picture(slide, GENERATED / "poster_label_efficiency.png", x + 0.2, y, w - 0.4, "figure: label efficiency")
    img_h = pic.height / 914400
    text(slide, x, y + img_h + 0.15, w, h - img_h - 0.15, [P((
        f"F1 with 200 m matching. Each of the {n['cells']} cells is scored once with the threshold it picked on 8 dev "
        f"scenes. Top: {n['test_scenes']} held-out test scenes. Bottom: {n['final_scenes']} near-shore, human-verified "
        "scenes. Seed 0; no intervals.", {"size": FOOT_PT}))], name="figure 1 caption")

    x, y, w, h = block(slide, 3, 20.83, 8.91, "SAR vs. optical depends on backbone and budget")
    pic = picture(slide, GENERATED / "poster_sar_minus_optical.png", x + 0.2, y, w - 0.4, "figure: SAR minus optical")
    img_h = pic.height / 914400
    text(slide, x, y + img_h + 0.15, w, h - img_h - 0.15, [P((
        f"Gray band: the largest test-F1 change across {n['rerun_cells']} cells that we retrained with the same recipe "
        f"and seed (±{n['rerun_max']}). A contrast inside the band is within rerun variation.", {"size": FOOT_PT}))],
        name="figure 2 caption")

    x, y, w, h = block(slide, 3, 30.36, 3.94, "The verified scenes are a coastal stress test")
    tiles = [
        (f"{n['final_f1_min']}–{n['final_f1_max']}", f"verified F1, all {n['cells']} cells\n(test: {n['test_f1_min']}–{n['test_f1_max']})"),
        (f"≤ {n['final_dark_recall_max']}", f"dark-vessel recall\n({n['final_dark']} dark vessels)"),
        (f"≤ {n['final_near_shore_f1_max']}", f"near-shore F1\n({n['final_near_shore']} vessels within 2 km)"),
    ]
    tile_w = w / 3
    for i, (big, label) in enumerate(tiles):
        tx = x + i * tile_w
        text(slide, tx, y - 0.05, tile_w, 0.95, [P(B(big, size=50, color=NAVY), align=PP_ALIGN.CENTER, space_after=0)],
             name=f"stat {i + 1} value")
        text(slide, tx, y + 0.95, tile_w, 0.9, [P((line, {"size": FOOT_PT}), align=PP_ALIGN.CENTER, space_after=0)
                                                 for line in label.split("\n")], name=f"stat {i + 1} label")
    text(slide, x, y + 1.95, w, 0.6, [P((
        f"Precision stays at {n['final_precision_min']}–{n['final_precision_max']} while recall falls to "
        f"{n['final_recall_min']}–{n['final_recall_max']}: the detectors miss coastal vessels rather than over-fire.",
        {"size": FOOT_PT}))], name="stat note")

    # ---------------------------------------------------------------- column 4
    x, y, w, h = block(slide, 4, 6.11, 12.33, "Findings")
    text(slide, x, y, w, 1.1, [P(B("12 ≈ 111", size=60, color=ROLE["imagenet"]), align=PP_ALIGN.CENTER, space_after=0)],
         name="hero number")
    text(slide, x, y + 1.15, w, 1.3, [P(
        f"On the verified scenes, ImageNet ViT with 12 training scenes matches random ViT with 111 "
        f"({n['final_vit_best12']} vs. {n['final_vit_random_111']} F1).", align=PP_ALIGN.CENTER)], name="hero caption")
    text(slide, x, y + 2.75, w, h - 2.75, [
        P(B("Pretraining pays off with few labels."), f" At 12 scenes every pretrained arm beats random: "
          f"{n['test_gain12_min']} to {n['test_gain12_max']} test F1, {n['final_gain12_min']} to "
          f"{n['final_gain12_max']} verified F1.", numbered=True, space_after=12),
        P(B("SAR beats optical only while labels are scarce."), f" CNN: {n['test_cnn_sar_minus_opt_10']} test F1 at 12 "
          f"scenes, {n['test_cnn_sar_minus_opt_100']} at 111. ViT: no stable order.", numbered=True, space_after=12),
        P(B("The input adapter matters."), " With a seeded 3-channel stem, BigEarthNet-S2 beats random at every budget "
          f"({n['cnn_optical_test_gain_min']} to {n['cnn_optical_test_gain_max']} test F1).", numbered=True, space_after=12),
        P(B("Coastal and dark vessels stay hard."), f" Verified F1 is {n['final_f1_min']}–{n['final_f1_max']} (test: "
          f"{n['test_f1_min']}–{n['test_f1_max']}). Dark-vessel recall ≤ {n['final_dark_recall_max']}; near-shore F1 ≤ "
          f"{n['final_near_shore_f1_max']}.", numbered=True),
    ], name="findings")
    x, y, w, h = block(slide, 4, 19.07, 6.23, "Limits")
    text(slide, x, y, w, h, [
        P("One seed per cell; 8 dev scenes pick checkpoints and thresholds.", bullet=True),
        P("The predeclared monotonicity gate failed: two SAR test curves drop by more than 0.02 as labels grow. "
          "We report the grid as descriptive.", bullet=True),
        P("Checkpoints differ in objective and corpus as well as domain. We compare released checkpoints, "
          "not a causal domain effect.", bullet=True),
    ], name="limits")
    x, y, w, h = block(slide, 4, 25.93, 4.94, "Code and checkpoints")
    qr_w = 2.6
    for i, (asset, label) in enumerate((("qr_github.png", "GitHub: code, evidence"),
                                        ("qr_huggingface.png", "Hugging Face: weights"))):
        qx = x + 0.3 + i * (w / 2)
        picture(slide, HERE / "assets" / asset, qx, y, qr_w, f"QR {label}")
        text(slide, qx - 0.6, y + qr_w + 0.1, qr_w + 1.2, 0.5, [P((label, {"size": FOOT_PT}), align=PP_ALIGN.CENTER)],
             name=f"QR label {i + 1}")
    x, y, w, h = block(slide, 4, 31.50, 2.66, "References")
    text(slide, x, y, w, h, [P((
        "Paolo et al., xView3-SAR, NeurIPS 2022. Zhou et al., Objects as Points, 2019. Steiner et al., How to Train "
        "Your ViT?, TMLR 2022. Woo et al., ConvNeXt V2, CVPR 2023. Straka & Gruber, SatDINO, 2025. Liu et al., "
        "SARMAE, CVPR 2026. Clasen et al., reBEN, IGARSS 2025.", {"size": 17}))], name="references")
    return prs


def pipeline(slide, x: float, y: float, w: float) -> None:
    """Two-row detector diagram, native shapes (mirrors figures/detector_pipeline.tex)."""

    grey_fill, grey_line = RGBColor(0xF0, 0xF0, 0xF0), RGBColor(0x8C, 0x8C, 0x8C)
    blue_fill, blue_line = RGBColor(0xE6, 0xF1, 0xF7), RGBColor(0x00, 0x5B, 0x8E)
    neck_fill = RGBColor(0xF2, 0xF8, 0xFB)
    cal_fill, cal_line = RGBColor(0xFC, 0xF1, 0xDB), CALLOUT_TITLE
    end_fill, end_line = RGBColor(0xE0, 0xF4, 0xEE), RGBColor(0x00, 0x7E, 0x5C)
    row1, box_h = y + 0.35, 1.25
    node(slide, x, row1 + 0.35, 1.9, box_h, ["[VH, VV,", "VH − VV]", "512² crop"], fill=grey_fill, line=grey_line, name="input")
    enc_x, enc_w = x + 2.45, 3.6
    group = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(enc_x - 0.15), Inches(row1 - 0.05),
                                   Inches(enc_w + 0.3), Inches(2.15))
    group.name = "encoder group"
    group.fill.background()
    group.line.color.rgb = grey_line
    group.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    group.line.width = Pt(1.5)
    group.shadow.inherit = False
    text(slide, enc_x - 0.15, row1 - 0.45, enc_w + 0.3, 0.4, [P(("one encoder per track", {"size": 15, "color": MUTED}),
                                                                align=PP_ALIGN.CENTER)], name="encoder group label")
    node(slide, enc_x, row1 + 0.05, enc_w, 0.85, "ViT-B/16, stride 16", fill=blue_fill, line=blue_line, name="ViT encoder")
    node(slide, enc_x, row1 + 1.15, enc_w, 0.85, "ConvNeXt-V2-B, s32 + up", fill=blue_fill, line=blue_line, name="CNN encoder")
    head_x = enc_x + enc_w + 0.65
    head_w = x + w - head_x
    node(slide, head_x, row1 + 0.35, head_w, box_h, ["shared", "256-ch head"], fill=neck_fill, line=blue_line, name="head")
    arrow(slide, x + 1.9, row1 + 1.0, enc_x, row1 + 0.475, elbow=True, name="input to ViT")
    arrow(slide, x + 1.9, row1 + 1.0, enc_x, row1 + 1.575, elbow=True, name="input to CNN")
    arrow(slide, enc_x + enc_w + 0.15, row1 + 1.0, head_x, row1 + 1.0, name="encoders to head")
    row2 = row1 + 2.45
    seg_w = (w - 2 * 0.45) / 3
    boxes = [("F1, 200 m geo match", end_fill, end_line, "scoring"),
             ("threshold τ picked on dev", cal_fill, cal_line, "threshold"),
             ("heatmap peaks ≥ 0.05, 120 m NMS", neck_fill, blue_line, "decode")]
    for i, (label, fill, line, name) in enumerate(boxes):
        node(slide, x + i * (seg_w + 0.45), row2, seg_w, box_h, label, fill=fill, line=line, name=name)
    arrow(slide, head_x + head_w / 2, row1 + 0.35 + box_h, head_x + head_w / 2, row2, name="head to decode")
    arrow(slide, x + 2 * (seg_w + 0.45), row2 + box_h / 2, x + seg_w + 0.45 + seg_w, row2 + box_h / 2, name="decode to threshold")
    arrow(slide, x + seg_w + 0.45, row2 + box_h / 2, x + seg_w, row2 + box_h / 2, name="threshold to scoring")


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
