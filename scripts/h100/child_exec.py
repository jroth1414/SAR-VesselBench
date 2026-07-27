"""Fail unless H100 sitecustomize is active, then exec an unchanged command."""

from __future__ import annotations

import argparse
import os

from scripts.h100.precision import assert_sitecustomize_active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    assert_sitecustomize_active()
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
