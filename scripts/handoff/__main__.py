"""CLI for the Sprint 7d H100 Box handoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .box import (
    MINIMUM_BOX_FREE_BYTES,
    client_from_environment,
    download_package,
    download_package_with_verifier,
    preflight_box,
    upload_package,
    upload_package_with_verifier,
)
from .package import (
    BuildOptions,
    EXPECTED_BRANCH,
    PackageError,
    _git_value,
    _hash_file,
    build_package,
    build_wheelhouse,
    extract_package,
    verify_package,
)
from .runtime_amendment import (
    build_runtime_amendment,
    extract_runtime_amendment,
    prepare_runtime_verifier,
    verify_runtime_amendment,
)


def _client(repo: Path) -> tuple[object, str]:
    return client_from_environment(repo.resolve())


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="check Box folder/quota/limit")
    preflight.add_argument("--repo", type=Path, default=Path.cwd())
    preflight.add_argument(
        "--minimum-free-bytes", type=int, default=MINIMUM_BOX_FREE_BYTES
    )

    wheelhouse = commands.add_parser(
        "wheelhouse", help="resolve the exact cu126 lock to binary wheels"
    )
    wheelhouse.add_argument("--repo", type=Path, required=True)
    wheelhouse.add_argument("--output", type=Path, required=True)
    wheelhouse.add_argument("--python", type=Path, default=Path(sys.executable))

    build = commands.add_parser("build", help="build a production source handoff")
    build.add_argument("--repo", type=Path, required=True)
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--wheelhouse", type=Path, required=True)
    build.add_argument("--apptainer-definition", type=Path, required=True)
    build.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="parent; builder creates xview3-h100-fp32-<full git SHA>",
    )
    build.add_argument("--max-part-bytes", type=int)

    upload = commands.add_parser("upload", help="verified Box upload, READY last")
    upload.add_argument("--repo", type=Path, required=True)
    upload.add_argument("--package-root", type=Path, required=True)
    upload.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="new outside-repository path for the atomic remote upload receipt",
    )

    download = commands.add_parser(
        "download", help="verified Box download through .partial files"
    )
    download.add_argument("--repo", type=Path, required=True)
    download.add_argument("--package-root", type=Path, required=True)
    download.add_argument("--expected-ready-sha256", required=True)
    download.add_argument("--expected-manifest-sha256", required=True)
    download.add_argument("--expected-sha256sums-sha256", required=True)
    download.add_argument("--expected-package-id", required=True)

    verify = commands.add_parser("verify", help="verify a received package")
    verify.add_argument("--package-root", type=Path, required=True)

    extract = commands.add_parser("extract", help="verify and stream-extract")
    extract.add_argument("--package-root", type=Path, required=True)
    extract.add_argument("--destination", type=Path, required=True)

    build_runtime = commands.add_parser(
        "build-runtime", help="build the code-only native-venv amendment"
    )
    build_runtime.add_argument("--repo", type=Path, required=True)
    build_runtime.add_argument("--base-package-root", type=Path, required=True)
    build_runtime.add_argument("--output-dir", type=Path, required=True)
    build_runtime.add_argument("--max-part-bytes", type=int)

    verify_runtime = commands.add_parser(
        "verify-runtime", help="verify the 7d base and 7e runtime amendment"
    )
    verify_runtime.add_argument("--base-package-root", type=Path, required=True)
    verify_runtime.add_argument("--package-root", type=Path, required=True)

    extract_runtime = commands.add_parser(
        "extract-runtime", help="verify and extract the native runtime bundle"
    )
    extract_runtime.add_argument("--base-package-root", type=Path, required=True)
    extract_runtime.add_argument("--package-root", type=Path, required=True)
    extract_runtime.add_argument("--destination", type=Path, required=True)

    upload_runtime = commands.add_parser(
        "upload-runtime", help="verified Box upload of the runtime amendment"
    )
    upload_runtime.add_argument("--repo", type=Path, required=True)
    upload_runtime.add_argument("--base-package-root", type=Path, required=True)
    upload_runtime.add_argument("--package-root", type=Path, required=True)
    upload_runtime.add_argument("--receipt", type=Path, required=True)

    download_runtime = commands.add_parser(
        "download-runtime", help="verified Box download of the runtime amendment"
    )
    download_runtime.add_argument("--repo", type=Path, required=True)
    download_runtime.add_argument("--base-package-root", type=Path, required=True)
    download_runtime.add_argument("--package-root", type=Path, required=True)
    download_runtime.add_argument("--expected-ready-sha256", required=True)
    download_runtime.add_argument("--expected-manifest-sha256", required=True)
    download_runtime.add_argument("--expected-sha256sums-sha256", required=True)
    download_runtime.add_argument("--expected-package-id", required=True)

    results = commands.add_parser(
        "build-results", help="package all 32 completed H100 cells for return"
    )
    results.add_argument("--repo", type=Path, required=True)
    results.add_argument("--runs-root", type=Path, required=True)
    results.add_argument("--campaign-manifest", type=Path, required=True)
    results.add_argument("--output", type=Path, required=True)
    results.add_argument("--max-part-bytes", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            client, folder_id = _client(args.repo)
            result = preflight_box(
                client,
                folder_id,
                minimum_free_bytes=args.minimum_free_bytes,
            )
            _print(result.as_dict())
        elif args.command == "wheelhouse":
            client, folder_id = _client(args.repo)
            result = preflight_box(client, folder_id)
            build_wheelhouse(
                repo_root=args.repo.resolve(),
                output=args.output.resolve(),
                python=args.python.resolve(),
            )
            _print(
                {
                    "wheelhouse": str(args.output.resolve()),
                    "box_preflight": result.as_dict(),
                }
            )
        elif args.command == "build":
            repo = args.repo.resolve()
            client, folder_id = _client(repo)
            result = preflight_box(client, folder_id)
            requested = args.max_part_bytes or result.maximum_file_bytes
            maximum = min(requested, result.maximum_file_bytes)
            commit = _git_value(repo, "rev-parse", "HEAD")
            package_root = (
                args.output_dir.resolve() / f"xview3-h100-fp32-{commit}"
            )
            built = build_package(
                BuildOptions(
                    repo_root=repo,
                    data_root=args.data_root.resolve(),
                    package_root=package_root,
                    wheelhouse=args.wheelhouse.resolve(),
                    apptainer_definition=args.apptainer_definition.resolve(),
                    max_part_bytes=maximum,
                    branch=EXPECTED_BRANCH,
                    production=True,
                )
            )
            _print(
                {
                    "package_root": str(built),
                    "box_preflight": result.as_dict(),
                    "part_bytes": maximum,
                }
            )
        elif args.command == "upload":
            client, folder_id = _client(args.repo)
            result = upload_package(
                client,
                folder_id,
                args.package_root,
                repo_root=args.repo,
                receipt_path=args.receipt,
                minimum_free_bytes=0,
            )
            _print(result)
        elif args.command == "download":
            client, folder_id = _client(args.repo)
            downloaded = download_package(
                client,
                folder_id,
                args.package_root,
                repo_root=args.repo,
                expected_ready_sha256=args.expected_ready_sha256,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_sha256sums_sha256=args.expected_sha256sums_sha256,
                expected_package_id=args.expected_package_id,
            )
            _print({"package_root": str(downloaded)})
        elif args.command == "verify":
            manifest = verify_package(args.package_root)
            _print(
                {
                    "status": "verified",
                    "package_id": manifest["package_id"],
                    "package_type": manifest["package_type"],
                }
            )
        elif args.command == "extract":
            destination = extract_package(
                args.package_root, args.destination
            )
            _print({"status": "extracted", "destination": str(destination)})
        elif args.command == "build-runtime":
            repo = args.repo.resolve()
            client, folder_id = _client(repo)
            result = preflight_box(client, folder_id, minimum_free_bytes=0)
            requested = args.max_part_bytes or result.maximum_file_bytes
            maximum = min(requested, result.maximum_file_bytes)
            package = build_runtime_amendment(
                repo_root=repo,
                base_package_root=args.base_package_root.resolve(),
                output_dir=args.output_dir.resolve(),
                maximum_physical_file_bytes=maximum,
            )
            manifest = json.loads(
                (package / "manifest.json").read_text(encoding="utf-8")
            )
            _print(
                {
                    "package_root": str(package),
                    "package_id": manifest["package_id"],
                    "ready_sha256": _hash_file(package / "READY.json"),
                    "manifest_sha256": _hash_file(package / "manifest.json"),
                    "sha256sums_sha256": _hash_file(package / "SHA256SUMS"),
                    "box_preflight": result.as_dict(),
                    "part_bytes": maximum,
                }
            )
        elif args.command == "verify-runtime":
            manifest = verify_runtime_amendment(
                args.package_root, args.base_package_root
            )
            _print(
                {
                    "status": "verified",
                    "package_id": manifest["package_id"],
                    "package_type": manifest["package_type"],
                    "runtime_identity_sha256": manifest[
                        "runtime_identity_sha256"
                    ],
                }
            )
        elif args.command == "extract-runtime":
            destination = extract_runtime_amendment(
                args.package_root,
                args.base_package_root,
                args.destination,
            )
            _print({"status": "extracted", "destination": str(destination)})
        elif args.command == "upload-runtime":
            client, folder_id = _client(args.repo)
            verifier = prepare_runtime_verifier(args.base_package_root)
            result = upload_package_with_verifier(
                client,
                folder_id,
                args.package_root,
                repo_root=args.repo,
                receipt_path=args.receipt,
                verifier=verifier,
                minimum_free_bytes=0,
            )
            _print(result)
        elif args.command == "download-runtime":
            client, folder_id = _client(args.repo)
            verifier = prepare_runtime_verifier(args.base_package_root)
            downloaded = download_package_with_verifier(
                client,
                folder_id,
                args.package_root,
                repo_root=args.repo,
                expected_ready_sha256=args.expected_ready_sha256,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_sha256sums_sha256=args.expected_sha256sums_sha256,
                expected_package_id=args.expected_package_id,
                verifier=verifier,
            )
            _print({"package_root": str(downloaded)})
        elif args.command == "build-results":
            # Keep preflight/upload/download importable in the minimal transfer
            # environment without importing the training/runtime dependency graph.
            from .results import build_results_package

            client, folder_id = _client(args.repo)
            result = preflight_box(client, folder_id, minimum_free_bytes=0)
            if args.max_part_bytes > result.maximum_file_bytes:
                raise PackageError(
                    "--max-part-bytes exceeds the current Box maximum file size"
                )
            package = build_results_package(
                repo=args.repo,
                runs_root=args.runs_root,
                campaign_manifest=args.campaign_manifest,
                output=args.output,
                max_part_bytes=args.max_part_bytes,
            )
            _print(
                {
                    "package_root": str(package),
                    "box_preflight": result.as_dict(),
                }
            )
        else:  # pragma: no cover - argparse makes this unreachable
            parser.error(f"unknown command: {args.command}")
    except (PackageError, RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
