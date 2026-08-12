"""Initialize strict IEEE FP32 before importing the shared trainer."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Apply and assert backend policy, then enter the shared trainer."""

    from src.runtime.precision import apply_strict_fp32

    apply_strict_fp32()

    from src.runtime.lightning_contract import assert_launch_process_contract

    assert_launch_process_contract()

    from src.train.finetune import main as finetune_main

    return finetune_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
