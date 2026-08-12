"""Launch one core training cell in a fresh strict-FP32 Python process."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from src.runtime.precision import STRICT_SENTINEL


CHILD_MODULE = "src.runtime.train_child"


def main(argv: Sequence[str] | None = None) -> int:
    """Set the driver-level TF32 policy, then replace this process."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    environment = os.environ.copy()
    environment["NVIDIA_TF32_OVERRIDE"] = "0"
    environment.pop(STRICT_SENTINEL, None)
    command = [sys.executable, "-m", CHILD_MODULE, *arguments]
    os.execvpe(command[0], command, environment)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
