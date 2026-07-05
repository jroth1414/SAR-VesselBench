"""LS-SSDD-v1.0 fetcher -> data/raw/lsssdd/ (DEVPLAN P1.2).

LS-SSDD-v1.0 (Zhang et al. 2020) is the ONLY auxiliary dataset: native
Sentinel-1, 15 large scenes pre-cut into 9,000 800x800 sub-images, the
labeled supervised-transfer source for Arms 4 and 8.

ACQUISITION REALITY (surfaced 2026-07-04): the GitHub repo the plan cites
(``TianwenZhang0825/LS-SSDD-v1.0-OPEN``) is a 48 KB landing page holding the
LICENSE and a README that points to the actual distribution at
https://radars.ac.cn (Journal of Radars dataset page; Baidu/direct links).
This script fetches the pinned repo (satisfying the P1.2 revision pin and the
LICENSE gate) and then tells you the imagery itself must be pulled from
radars.ac.cn — a manual step until that download is automated.

License gate: the license text found in the repo is recorded to
``LICENSE.note`` at download; downstream code must refuse to use the dataset
when that note is missing (P1.2).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import zipfile
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen

LSSSDD_REPO = "TianwenZhang0825/LS-SSDD-v1.0-OPEN"


def download_lsssdd(out_root: Path, *, revision: str) -> None:
    """Download the pinned LS-SSDD repo zip and unpack it."""

    if not revision:
        raise SystemExit(
            "a pinned --revision (commit sha or tag) is mandatory (P1.2)"
        )
    url = f"https://codeload.github.com/{LSSSDD_REPO}/zip/{revision}"
    print(f"downloading {url} ...")
    with urlopen(url) as response:  # noqa: S310 — pinned github codeload URL
        payload = response.read()
    print(f"downloaded {len(payload) / 2**20:.1f} MiB, unpacking")

    out_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(out_root)

    _write_note(
        out_root / "SOURCE.note",
        f"github:{LSSSDD_REPO}@{revision} (codeload zip)",
    )

    license_texts = sorted(out_root.rglob("LICENSE*")) + sorted(
        out_root.rglob("license*")
    )
    if license_texts:
        _write_note(
            out_root / "LICENSE.note",
            f"license file(s) in repo: {[str(p) for p in license_texts]}",
        )
    else:
        print(
            "WARNING: no license file found in the LS-SSDD repo — record the "
            "license terms manually in LICENSE.note before using the data "
            "(P1.2 gate: do not proceed on a source whose license note is missing)"
        )

    print(
        "NOTE: the repo is a landing page only — the 9,000 sub-images are "
        "distributed via https://radars.ac.cn (links in the repo README). "
        "Fetch them into data/raw/lsssdd/ before building data/lsssdd_split.json."
    )


def _write_note(path: Path, detail: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {detail}\n")
    print(f"note updated: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--revision", default=None, help="LS-SSDD repo commit sha or tag (mandatory)"
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    download_lsssdd(
        Path(config["paths"]["raw_lsssdd"]),
        revision=args.revision or config["download"].get("lsssdd_revision") or "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
