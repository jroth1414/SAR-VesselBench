"""Live Trackio dashboard as a SIDECAR — zero changes to training code.

Lightning's CSVLogger flushes ``runs/<exp>/metrics/metrics.csv`` continuously
during every run; this tailer watches those files and streams new rows into
trackio (Hugging Face's local-first tracker). The frozen training recipe,
the training venv, and the lockfiles are untouched — trackio lives in its
own venv (``.venv-trackio``) and this process can die/restart freely.

Usage (dev box):
    .venv-trackio\\Scripts\\python scripts/trackio_tail.py --backfill  # once
    .venv-trackio\\Scripts\\python scripts/trackio_tail.py             # live tail
    .venv-trackio\\Scripts\\trackio show                               # dashboard

The queue is serial (one training at a time), so the tailer tracks the
single active run and finishes it when its final_metrics.json appears.
``--backfill`` imports every already-completed run once (state kept in
runs/.trackio_seen.json). On the 8-wide node, run one tailer per GPU log
dir or just backfill between waves.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

PROJECT = "jhu-xview3"
RUNS = Path("runs")
SEEN_PATH = RUNS / ".trackio_seen.json"
# columns worth charting (anything else in metrics.csv is ignored)
METRICS = ("train_loss", "dev_f1", "val_loss", "lr-AdamW", "epoch")


def load_seen() -> dict:
    return json.loads(SEEN_PATH.read_text()) if SEEN_PATH.exists() else {}


def save_seen(seen: dict) -> None:
    SEEN_PATH.write_text(json.dumps(seen, indent=1), newline="\n")


def metric_rows(csv_path: Path, start_row: int) -> tuple[list[dict], int]:
    try:
        table = pd.read_csv(csv_path, on_bad_lines="skip")
    except Exception:
        return [], start_row
    table = table.iloc[start_row:]
    rows = []
    for _, row in table.iterrows():
        payload = {
            key: float(row[key])
            for key in METRICS
            if key in row and pd.notna(row[key])
        }
        if payload and "step" in row and pd.notna(row["step"]):
            payload["step"] = int(row["step"])
            rows.append(payload)
    return rows, start_row + len(table)


def import_run(trackio, run_dir: Path, *, live: bool, seen: dict) -> None:
    csv_path = run_dir / "metrics" / "metrics.csv"
    if not csv_path.exists():
        return
    exp_id = run_dir.name
    state = seen.get(exp_id, {"rows": 0, "finished": False})
    if state["finished"]:
        return

    config = {}
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        config = {"config_file": str(config_path)}

    trackio.init(project=PROJECT, name=exp_id, config=config, resume="allow")
    while True:
        rows, new_count = metric_rows(csv_path, state["rows"])
        for payload in rows:
            step = payload.pop("step")
            trackio.log(payload, step=step)
        state["rows"] = new_count
        seen[exp_id] = state
        save_seen(seen)

        done = (run_dir / "final_metrics.json").exists()
        if done:
            final = json.loads((run_dir / "final_metrics.json").read_text())
            summary = {
                key: value
                for key, value in final.items()
                if isinstance(value, (int, float))
            }
            if summary:
                trackio.log(summary)
            state["finished"] = True
            seen[exp_id] = state
            save_seen(seen)
            break
        if not live:
            break
        time.sleep(30)
    trackio.finish()


def active_run() -> Path | None:
    candidates = []
    for csv_path in RUNS.glob("*/metrics/metrics.csv"):
        run_dir = csv_path.parents[1]
        if (run_dir / "final_metrics.json").exists():
            continue
        candidates.append((csv_path.stat().st_mtime, run_dir))
    if not candidates:
        return None
    mtime, run_dir = max(candidates)
    return run_dir if (time.time() - mtime) < 300 else None


def main() -> int:
    import trackio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill", action="store_true", help="import finished runs once, then exit")
    args = parser.parse_args()

    seen = load_seen()
    if args.backfill:
        for run_dir in sorted(RUNS.iterdir()):
            if run_dir.is_dir() and (run_dir / "final_metrics.json").exists():
                print(f"backfill: {run_dir.name}")
                import_run(trackio, run_dir, live=False, seen=seen)
        print("backfill complete — dashboard: trackio show")
        return 0

    print("live tail started (30 s poll); dashboard: trackio show")
    while True:
        run_dir = active_run()
        if run_dir is None:
            time.sleep(60)
            continue
        print(f"tailing {run_dir.name}")
        import_run(trackio, run_dir, live=True, seen=seen)


if __name__ == "__main__":
    raise SystemExit(main())
