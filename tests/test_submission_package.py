from pathlib import Path, PurePosixPath

import pytest

from tools.build_submission import FIXED_TIME, ROOT_FILES, scan, select, zip_info
from tools.check_report import page_labels


def test_page_labels_measure_intro_through_conclusion(tmp_path: Path) -> None:
    aux = tmp_path / "final_report.aux"
    aux.write_text(
        "\\newlabel{mainbody:start}{{1}{2}{Introduction}{section.1}{}}\n"
        "\\newlabel{mainbody:end}{{6}{6}{Conclusion}{section.6}{}}\n",
        encoding="utf-8",
    )
    assert page_labels(aux) == {"mainbody:start": 2, "mainbody:end": 6}


def test_page_labels_fail_closed_when_boundary_is_missing(tmp_path: Path) -> None:
    aux = tmp_path / "final_report.aux"
    aux.write_text(
        "\\newlabel{mainbody:start}{{1}{2}{Introduction}{section.1}{}}\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="mainbody:end"):
        page_labels(aux)


def test_submission_scan_rejects_private_site_path(tmp_path: Path) -> None:
    source = tmp_path / "log.txt"
    source.write_text("scratch=/projects/" "geofam/example\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="private or credential-like"):
        scan(source, PurePosixPath("logs/progress.txt"))


@pytest.mark.parametrize("suffix", [".log", ".toml", ".sh", ".cls", ".bst"])
def test_submission_scan_checks_every_selected_text_type(
    tmp_path: Path, suffix: str
) -> None:
    source = tmp_path / f"member{suffix}"
    source.write_bytes(b"BOX" b"_FOLDER_ID=forbidden\n")
    with pytest.raises(SystemExit, match="private or credential-like"):
        scan(source, PurePosixPath(source.name))


def test_submission_allowlist_excludes_data_and_build_debris(tmp_path: Path) -> None:
    tracked: set[str] = set()
    for name in ROOT_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n", encoding="utf-8")
        tracked.add(name)

    config = tmp_path / "configs" / "arms.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("arms: {}\n", encoding="utf-8")
    tracked.add("configs/arms.yaml")
    chipper = tmp_path / "src" / "data" / "chipper.py"
    chipper.parent.mkdir(parents=True)
    chipper.write_text("# scientific source\n", encoding="utf-8")
    tracked.add("src/data/chipper.py")

    data = tmp_path / "data" / "labels.csv"
    data.parent.mkdir(parents=True)
    data.write_text("secret data\n", encoding="utf-8")

    build_debris = tmp_path / "docs" / "class_report" / "final_report.aux"
    build_debris.parent.mkdir(parents=True)
    build_debris.write_text("aux\n", encoding="utf-8")
    class_pdf = build_debris.with_suffix(".pdf")
    class_pdf.write_bytes(b"pdf")
    aipr_pdf = tmp_path / "docs" / "aipr2026" / "paper.pdf"
    aipr_pdf.parent.mkdir(parents=True)
    aipr_pdf.write_bytes(b"pdf")

    names = {path.relative_to(tmp_path).as_posix() for path in select(tmp_path, tracked)}
    assert "configs/arms.yaml" in names
    assert "src/data/chipper.py" in names

    scorer = tmp_path / "scripts" / "score_test_cohort.py"
    scorer.parent.mkdir(parents=True)
    scorer.write_text("# held-out cohort scorer\n", encoding="utf-8")
    tracked.add("scripts/score_test_cohort.py")
    names = {path.relative_to(tmp_path).as_posix() for path in select(tmp_path, tracked)}
    assert "scripts/score_test_cohort.py" in names
    assert "docs/class_report/final_report.pdf" in names
    assert "docs/aipr2026/paper.pdf" in names
    assert "docs/class_report/final_report.aux" not in names
    assert not any(name.startswith("data/") for name in names)


def test_zip_metadata_is_reproducible() -> None:
    info = zip_info("README.md", executable=False)
    assert info.date_time == FIXED_TIME
    assert info.compress_type > 0
    assert info.external_attr >> 16 & 0o777 == 0o644
