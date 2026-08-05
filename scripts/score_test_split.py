"""Compatibility entrypoint for cohort-gated TEST scoring.

The historical implementation mutated ``final_metrics.json`` and paired
``best.ckpt`` with ``last_dev.threshold``.  That path is intentionally gone;
all callers are routed to the all-32 cohort barrier and immutable
``test_metrics.json`` implementation.
"""

from scripts.score_test_cohort import main, score_run

__all__ = ["main", "score_run"]


if __name__ == "__main__":
    raise SystemExit(main())
