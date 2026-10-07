"""Poster v2 visuals that replace text: scoreboard, budget waffle, scene map, dark-vessel crop.

* ``figure_scoreboard`` turns the 32 paired-bootstrap SAR differences into
  two 16-dot grids (vs optical, vs ImageNet). A dot is a win only when the
  whole 95% interval sits on one side of zero, the rule the answer strip uses.
* ``figure_budget_waffle`` draws the 111 training scenes as squares, shaded
  by the smallest nested budget that contains them (12 < 28 < 56 < 111).
* ``figure_scene_map`` places each study and verified scene at the mean
  position of its labels on Natural Earth land polygons (an input file kept
  outside the repository; its SHA-256 is checked).
* ``figure_dark_vessels`` marks AIS-matched and dark (no AIS) verified vessels
  on one radar window chosen by a fixed rule.
* ``reference_numbers`` reads the two reference detectors (YOLO26 at 111
  scenes; zero-shot LocateAnything-3B) after checking their SHA-256 sums.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.analysis.heldout_results import EvidenceError

BUDGETS = (12, 28, 56, 111)
FRACTION_OF = {12: 10, 28: 25, 56: 50, 111: 100}
WAFFLE_ROWS = 4
LAND_SHA256 = "e874b27a51d146452be360cafb3cc50c86001074a67d534113e6534682f9826b"  # ne_50m_land.geojson
AIS_COLOR, DARK_COLOR = "#F0E442", "#D55E00"  # Okabe-Ito yellow and vermillion
SPLIT_STYLE = {  # split -> (label, marker, size, color)
    "train": ("Train (111)", "o", 70, "#9AA3B2"),
    "dev": ("Dev (23)", "^", 150, "#0072B2"),
    "test": ("Test (16)", "s", 150, "#CC79A7"),
    "eval_final": ("Verified (50)", "D", 150, "#009E73"),
}
GUINEA_BOX = (-4.0, 16.0, -3.0, 9.0)  # lon0, lon1, lat0, lat1 of the Gulf of Guinea inset
EUROPE_BOX = (-25.0, 18.0, 36.0, 69.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- scoreboard
def scoreboard(intervals: Mapping[str, Mapping[str, float]], rival: str) -> np.ndarray:
    """4 budgets x 4 (ViT test, ViT verified, CNN test, CNN verified): +1 SAR win, -1 rival win, 0 tie."""

    grid = np.zeros((4, 4), dtype=int)
    for row, fraction in enumerate((10, 25, 50, 100)):
        for col, (track, metric) in enumerate((("vit", "test"), ("vit", "final"), ("cnn", "test"), ("cnn", "final"))):
            ci = intervals[f"{metric}_{track}_sar_minus_{rival}_{fraction}"]
            grid[row, col] = 1 if ci["low"] > 0 else -1 if ci["high"] < 0 else 0
    return grid


def figure_scoreboard(intervals, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from src.analysis.heldout_results import ROLE_COLOR
    from src.analysis.poster_figures import ANNOT_PT, GRID, INK, MUTED, save_both

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9))
    for axis, (rival, name) in zip(axes, (("opt", "optical"), ("imagenet", "imagenet"))):
        grid = scoreboard(intervals, rival)
        rival_color = ROLE_COLOR[name]
        for row in range(4):
            for col in range(4):
                x, y = col + (0.35 if col >= 2 else 0.0), 3 - row
                value = grid[row, col]
                if value == 0:
                    axis.plot(x, y, "o", ms=30, mfc="white", mec=MUTED, mew=3)
                else:
                    axis.plot(x, y, "o", ms=30, color=ROLE_COLOR["sar"] if value > 0 else rival_color)
        wins, losses = int((grid > 0).sum()), int((grid < 0).sum())
        rival_label = "optical" if name == "optical" else "ImageNet"
        axis.set_title(f"SAR vs {rival_label}\n{wins} wins · {losses} losses · {16 - wins - losses} ties",
                       color=INK, fontsize=ANNOT_PT + 3, linespacing=1.3)
        axis.set_xlim(-0.6, 3.95)
        axis.set_ylim(-0.6, 3.6)
        axis.set_yticks(range(4))
        axis.set_yticklabels(["111", "56", "28", "12"] if axis is axes[0] else [])
        axis.set_xticks([0, 1, 2.35, 3.35])
        axis.set_xticklabels(["test", "verif.", "test", "verif."], fontsize=ANNOT_PT - 3)
        for x, label in ((0.5, "ViT"), (2.85, "CNN")):
            axis.text(x, -1.15, label, ha="center", va="top", fontsize=ANNOT_PT, fontweight="bold", color=INK)
        axis.tick_params(length=0)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_facecolor("none")
        axis.axvline(1.68, color=GRID, lw=2)
    axes[0].set_ylabel("Training scenes")
    handles = [Line2D([], [], marker="o", ms=20, color=ROLE_COLOR["sar"], ls="none"),
               Line2D([], [], marker="o", ms=20, color=ROLE_COLOR["optical"], ls="none"),
               Line2D([], [], marker="o", ms=20, color=ROLE_COLOR["imagenet"], ls="none"),
               Line2D([], [], marker="o", ms=20, mfc="white", mec=MUTED, mew=3, ls="none")]
    fig.legend(handles, ["SAR wins", "optical wins", "ImageNet wins", "tie (95% CI crosses 0)"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), handletextpad=0.1, columnspacing=0.8,
               fontsize=ANNOT_PT - 3)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.78, bottom=0.27, wspace=0.08)
    paths = save_both(fig, out_dir, "poster_scoreboard")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- budget waffle
def figure_budget_waffle(out_dir: Path) -> list[Path]:
    """111 squares, one per training scene, filled column by column."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from src.analysis.poster_figures import ANNOT_PT, INK, save_both

    shades = {12: "#14213D", 28: "#3D5A80", 56: "#7D97B8", 111: "#C9D3E0"}
    columns = int(np.ceil(BUDGETS[-1] / WAFFLE_ROWS))
    fig, axis = plt.subplots(figsize=(9.6, 2.1))
    for index in range(BUDGETS[-1]):
        col, row = divmod(index, WAFFLE_ROWS)
        budget = next(b for b in BUDGETS if index < b)
        axis.add_patch(Rectangle((col + 0.08, WAFFLE_ROWS - 1 - row + 0.08), 0.84, 0.84, color=shades[budget], lw=0))
    for budget in BUDGETS:
        end = budget / WAFFLE_ROWS  # the column boundary where this budget ends
        axis.text(end, WAFFLE_ROWS + 0.2, f"{budget}", ha="right", va="bottom", fontsize=ANNOT_PT + 2,
                  fontweight="bold", color=shades[budget] if budget != 111 else "#5B6B80")
        if budget != BUDGETS[-1]:
            axis.plot([end, end], [-0.1, WAFFLE_ROWS + 0.15], color=INK, lw=2.2)
    axis.text(0, -0.25, "each square = one training scene; every budget contains the smaller ones",
              ha="left", va="top", fontsize=ANNOT_PT - 3, color=INK)
    axis.set_xlim(-0.1, columns + 0.1)
    axis.set_ylim(-1.4, WAFFLE_ROWS + 1.25)
    axis.set_aspect("equal")
    axis.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    paths = save_both(fig, out_dir, "poster_budget_waffle")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- scene map
def scene_positions(label_csvs: Sequence[Path], scene_ids: set[str]) -> dict[str, tuple[float, float]]:
    """Mean (lon, lat) of each scene's labels."""

    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for path in label_csvs:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                scene = row["scene_id"]
                if scene in scene_ids and row.get("detect_lat") and row.get("detect_lon"):
                    entry = sums[scene]
                    entry[0] += float(row["detect_lon"])
                    entry[1] += float(row["detect_lat"])
                    entry[2] += 1
    return {scene: (lon / n, lat / n) for scene, (lon, lat, n) in sums.items() if n}


def _land_polygons(land_geojson: Path):
    if _sha256(land_geojson) != LAND_SHA256:
        raise EvidenceError("land polygons do not match the pinned ne_50m_land.geojson")
    features = json.loads(land_geojson.read_text(encoding="utf-8"))["features"]
    for feature in features:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            yield np.asarray(polygon[0])


def figure_scene_map(splits: Mapping[str, Sequence[str]], label_csvs: Sequence[Path], land_geojson: Path,
                     out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    from src.analysis.poster_figures import ANNOT_PT, INK, save_both

    wanted = {str(s) for ids in splits.values() for s in ids}
    positions = scene_positions(label_csvs, wanted)
    missing = wanted - set(positions)
    if missing:
        raise EvidenceError(f"{len(missing)} scenes have no labelled positions for the map")
    rings = list(_land_polygons(land_geojson))
    fig = plt.figure(figsize=(9.6, 4.3))
    main = fig.add_axes((0.0, 0.0, 0.66, 1.0))
    inset = fig.add_axes((0.68, 0.02, 0.31, 0.42))
    for axis, (lon0, lon1, lat0, lat1) in ((main, EUROPE_BOX), (inset, GUINEA_BOX)):
        axis.set_facecolor("#E8F1F8")
        for ring in rings:
            if ring[:, 0].max() < lon0 or ring[:, 0].min() > lon1 or ring[:, 1].max() < lat0 or ring[:, 1].min() > lat1:
                continue
            axis.add_patch(Polygon(ring, closed=True, facecolor="#F4F1EA", edgecolor="#B8B2A6", lw=0.8))
        for split, ids in splits.items():
            label, marker, size, color = SPLIT_STYLE[split]
            pts = np.array([positions[str(s)] for s in ids])
            axis.scatter(pts[:, 0], pts[:, 1], s=size, marker=marker, c=color, edgecolors="white", linewidths=1.2,
                         zorder=3, label=label if axis is main else None)
        axis.set_xlim(lon0, lon1)
        axis.set_ylim(lat0, lat1)
        axis.set_aspect(1.3)  # mild stretch toward an equal-area look at these latitudes
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#9AA3B2")
    inset.set_title("Gulf of Guinea", fontsize=ANNOT_PT - 2, color=INK, pad=6)
    fig.legend(*main.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.67, 0.99),
               fontsize=ANNOT_PT - 2, frameon=False, handletextpad=0.2, labelspacing=0.5)
    paths = save_both(fig, out_dir, "poster_scene_map")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- dark vessels
def choose_dark_window(rows_by_scene: Mapping[str, list[dict]], width: int, height: int):
    """The offshore window with the most dark vessels among windows that also hold AIS vessels."""

    best = None
    for scene, rows in rows_by_scene.items():
        vessels = []
        for row in rows:
            if row["is_vessel"].strip().lower() != "true" or row["confidence"].strip().upper() == "LOW":
                continue
            distance = float(row["distance_from_shore_km"]) if row["distance_from_shore_km"] else None
            if distance is None or distance <= 5.0:
                continue
            dark = row["source"].strip().lower() == "manual"
            vessels.append((int(row["detect_scene_column"]), int(row["detect_scene_row"]), dark))
        for cx, cy, _ in vessels:
            x0, y0 = cx - width // 2, cy - height // 2
            inside = [(x, y, d) for x, y, d in vessels if x0 <= x < x0 + width and y0 <= y < y0 + height]
            n_dark = sum(d for *_, d in inside)
            n_ais = len(inside) - n_dark
            if n_ais >= 3 and n_dark >= 3:
                key = (min(n_dark, n_ais), n_dark + n_ais)
                if best is None or key > best[0]:
                    best = (key, scene, x0, y0)
    if best is None:
        raise EvidenceError("no offshore window holds both AIS and dark vessels")
    return best[1:]


def figure_dark_vessels(verified_csv: Path, archive_dir: Path, cache_dir: Path, out_dir: Path,
                        width: int = 1400, height: int = 560) -> tuple[list[Path], dict[str, object]]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from src.analysis.poster_coastal import read_vh_window
    from src.analysis.poster_figures import save_both

    rows_by_scene: dict[str, list[dict]] = defaultdict(list)
    with verified_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows_by_scene[row["scene_id"]].append(row)
    scene, x0, y0 = choose_dark_window(rows_by_scene, width, height)
    import rasterio
    from rasterio.windows import Window

    read_vh_window(archive_dir, cache_dir, scene, x0, y0)  # extracts the scene's VH raster into the cache once
    with rasterio.open(cache_dir / f"{scene}_VH_dB.tif") as dataset:
        x0 = max(0, min(x0, dataset.width - width))
        y0 = max(0, min(y0, dataset.height - height))
        image = dataset.read(1, window=Window(x0, y0, width, height)).astype(float)
    image = np.where(np.isfinite(image) & (image > -100), image, np.nan)
    low, high = np.nanpercentile(image, [2, 99.5])
    fig, axis = plt.subplots(figsize=(9.6, 4.5))
    axis.imshow(np.clip(image, low, high), cmap="gray", extent=(0, width / 100, height / 100, 0),
                interpolation="nearest")
    counts = {"ais": 0, "dark": 0}
    for row in rows_by_scene[scene]:
        if row["is_vessel"].strip().lower() != "true" or row["confidence"].strip().upper() == "LOW":
            continue
        x, y = int(row["detect_scene_column"]) - x0, int(row["detect_scene_row"]) - y0
        if not (0 <= x < width and 0 <= y < height):
            continue
        dark = row["source"].strip().lower() == "manual"
        counts["dark" if dark else "ais"] += 1
        if dark:
            axis.plot(x / 100, y / 100, marker="s", ms=22, mfc="none", mec=DARK_COLOR, mew=3.5)
        else:
            axis.plot(x / 100, y / 100, marker="o", ms=24, mfc="none", mec=AIS_COLOR, mew=3.5)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    handles = [Line2D([], [], marker="o", ms=20, mfc="none", mec=AIS_COLOR, mew=3.5, ls="none"),
               Line2D([], [], marker="s", ms=18, mfc="none", mec=DARK_COLOR, mew=3.5, ls="none")]
    fig.legend(handles, ["broadcasting AIS", "dark: no AIS match"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.5)
    axis.text(0.01, 0.97, f"{width // 100} × {height // 100} km, VH", transform=axis.transAxes, ha="left",
              va="top", color="white", fontsize=20)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.17)
    paths = save_both(fig, out_dir, "poster_dark_vessels")
    plt.close(fig)
    return paths, {"dark_example_scene": scene, "dark_example_ais": counts["ais"],
                   "dark_example_dark": counts["dark"]}


# --------------------------------------------------------------------------- reference detectors
REFERENCE_FILES = ("yolo26-f100/final_metrics.json", "locateanything-zs/final_metrics.json")


def reference_numbers(references_root: Path, expected_test_positives: int) -> dict[str, object]:
    """YOLO26 test F1 and the zero-shot LocateAnything F1, each file checked against SHA256SUMS."""

    sums = {}
    for line in (references_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip().lstrip("*").removeprefix("references/")] = digest
    for name in REFERENCE_FILES:
        if _sha256(references_root / name) != sums.get(name):
            raise EvidenceError(f"reference file does not match SHA256SUMS: {name}")
    yolo = json.loads((references_root / REFERENCE_FILES[0]).read_text(encoding="utf-8"))
    test = yolo["test"]["aggregate"]
    if test["tp"] + test["fn"] != expected_test_positives:
        raise EvidenceError("YOLO26 test counts disagree with the audited test positives")
    zero_shot = json.loads((references_root / REFERENCE_FILES[1]).read_text(encoding="utf-8"))
    best = zero_shot["per_prompt"][zero_shot["best_prompt"]]
    best = best.get("aggregate", best)
    return {
        "yolo_test_f1": float(test["f1"]),
        "yolo_test_near_shore_f1": float(yolo["test_near_shore_f1"]),
        "zero_shot_f1": float(best["f1"]),
        "zero_shot_prompt": str(zero_shot["best_prompt"]),
        "zero_shot_vessels": int(best["tp"] + best["fn"]),
    }
