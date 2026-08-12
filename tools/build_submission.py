#!/usr/bin/env python3
"""Build the deterministic, data-free class submission archive."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE = "Roth_John_final_project.zip"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)
BLOCKED_SUFFIXES = {
    ".ckpt", ".env", ".jwt", ".key", ".npy", ".npz", ".parquet", ".pem",
    ".pt", ".pth", ".safetensors", ".tif", ".tiff", ".zip",
}
BLOCKED_ANYWHERE = {
    ".git", ".pytest_cache", "__pycache__",
}
BLOCKED_TOP_LEVEL = {
    "data", "runs", "weights", "checkpoints", "slurm", "containers", "handoff",
}
PRIVATE_PATTERNS = (
    re.compile(rb"/projects/" rb"geofam/", re.I),
    re.compile(rb"/nfs/" rb"WRIVA/", re.I),
    re.compile(rb"BOX" rb"_FOLDER_ID", re.I),
    re.compile(rb"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY", re.I),
    re.compile(rb"AutomationUser_[A-Za-z0-9_]+@boxdevedition\.com", re.I),
)

ROOT_FILES = (
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "Makefile", "pyproject.toml",
    "requirements-ci.txt",
)
TREES = (
    "configs", "locks", ".github/workflows", "src", "scripts", "tests", "tools", "docs/class_report",
    "docs/aipr2026", "docs/results/generated", "results/h100",
)


def tracked(repo: Path) -> set[str]:
    output = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "ls-files", "-z"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    return {item.decode("utf-8") for item in output.split(b"\0") if item}


def select(repo: Path, tracked_paths: set[str]) -> list[Path]:
    chosen: set[Path] = set()
    for name in ROOT_FILES:
        path = repo / name
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"required submission file is missing or unsafe: {name}")
        chosen.add(path)
    for prefix in TREES:
        root = repo / prefix
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("*")):
            rel_text = candidate.relative_to(repo).as_posix()
            if (
                rel_text in tracked_paths
                and candidate.is_file()
                and not candidate.is_symlink()
            ):
                chosen.add(candidate)

    # Built PDFs may be untracked; all source members come from the Git index.
    for rel in ("docs/class_report/final_report.pdf", "docs/aipr2026/paper.pdf"):
        path = repo / rel
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"build the required PDF before packaging: {rel}")
        chosen.add(path)

    result: list[Path] = []
    for path in sorted(chosen, key=lambda p: p.relative_to(repo).as_posix()):
        rel = path.relative_to(repo)
        parts = set(rel.parts)
        if parts & BLOCKED_ANYWHERE or rel.parts[0] in BLOCKED_TOP_LEVEL:
            raise SystemExit(f"blocked directory entered allowlist: {rel}")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            raise SystemExit(f"blocked file type entered allowlist: {rel}")
        rel_text = rel.as_posix()
        if rel_text not in tracked_paths and path.suffix.lower() != ".pdf":
            raise SystemExit(f"untracked non-PDF file entered submission: {rel}")
        result.append(path)
    return result


def scan(path: Path, rel: PurePosixPath) -> None:
    if path.stat().st_size > 100_000_000:
        raise SystemExit(f"submission member exceeds 100 MB: {rel}")
    payload = path.read_bytes()
    for pattern in PRIVATE_PATTERNS:
        if pattern.search(payload):
            raise SystemExit(f"private or credential-like text in {rel}: {pattern.pattern!r}")


def zip_info(name: str, executable: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (args.output or repo / "dist" / ARCHIVE).resolve()
    if output.name != ARCHIVE:
        raise SystemExit(f"submission archive must be named exactly {ARCHIVE}")
    subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "diff", "--cached", "--quiet"],
        cwd=repo,
        check=True,
    )
    if output.parent != repo / "dist":
        raise SystemExit("submission archive must be written directly under the repository dist directory")

    subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "diff", "--quiet"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "diff", "--cached", "--check"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [os.environ.get("PYTHON", "python"), str(repo / "tools/check_report.py"),
         str(repo / "docs/class_report/final_report.pdf"), "--aux",
         str(repo / "docs/class_report/final_report.aux")],
        cwd=repo,
        check=True,
    )

    members = select(repo, tracked(repo))
    hashes: list[str] = []
    for path in members:
        rel = PurePosixPath(path.relative_to(repo).as_posix())
        scan(path, rel)
        hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel}")
    checksum_payload = ("\n".join(hashes) + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise SystemExit(f"partial archive already exists: {partial}")
    try:
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in members:
                rel = path.relative_to(repo).as_posix()
                archive.writestr(zip_info(rel, False), path.read_bytes())
            archive.writestr(zip_info("SHA256SUMS", False), checksum_payload)
            inventory = "\n".join(path.relative_to(repo).as_posix() for path in members) + "\n"
            archive.writestr(zip_info("SUBMISSION_INVENTORY.txt", False), inventory.encode("utf-8"))
        with zipfile.ZipFile(partial) as archive:
            bad = archive.testzip()
            if bad:
                raise SystemExit(f"ZIP CRC validation failed: {bad}")
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise SystemExit("ZIP contains duplicate member names")
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"submission_archive={output}")
    print(f"submission_sha256={digest}")
    print(f"submission_members={len(members) + 2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
