#!/usr/bin/env python3
"""Validate the class-report filename and five-page main-body boundary."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

LABEL = re.compile(r"\\newlabel\{(?P<name>mainbody:(?:start|end))\}\{\{.*?\}\{(?P<page>\d+)\}")


def page_labels(aux: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for line in aux.read_text(encoding="utf-8", errors="strict").splitlines():
        match = LABEL.search(line)
        if match:
            labels[match.group("name")] = int(match.group("page"))
    missing = {"mainbody:start", "mainbody:end"} - labels.keys()
    if missing:
        raise SystemExit("missing resolved report page labels: " + ", ".join(sorted(missing)))
    return labels


def pdf_pages(pdf: Path) -> int:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit("pypdf is required to validate final_report.pdf") from exc
    return len(PdfReader(str(pdf)).pages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--aux", required=True, type=Path)
    args = parser.parse_args()

    if args.pdf.name != "final_report.pdf":
        raise SystemExit("class report must be named exactly final_report.pdf")
    if not args.pdf.is_file() or args.pdf.is_symlink():
        raise SystemExit("final_report.pdf is missing or unsafe")
    if not args.aux.is_file() or args.aux.is_symlink():
        raise SystemExit("report AUX file is missing or unsafe")

    labels = page_labels(args.aux)
    start = labels["mainbody:start"]
    end = labels["mainbody:end"]
    span = end - start + 1
    if start < 1 or end < start:
        raise SystemExit(f"invalid main-body page range: {start}--{end}")
    if span > 5:
        raise SystemExit(f"main body occupies {span} pages ({start}--{end}); maximum is 5")

    total = pdf_pages(args.pdf)
    if total < end:
        raise SystemExit("PDF page count is smaller than the resolved conclusion page")
    print(f"report_validation=success main_body_pages={span} range={start}-{end} total_pages={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
