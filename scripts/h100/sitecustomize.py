"""Auto-loaded, fail-closed strict-FP32 guard for every H100 subprocess."""

import os
import sys

try:
    from scripts.h100.precision import apply_strict_fp32

    apply_strict_fp32()
except BaseException as exc:  # Python otherwise only warns and keeps running.
    sys.stderr.write(f"FATAL H100 strict-FP32 initialization failed: {exc}\n")
    sys.stderr.flush()
    os._exit(86)
