"""Export small result artifacts from runs/ into results/ for git sharing.

runs/ stays gitignored (checkpoints, logs, media — ground rule 11); this
copies ONLY the small, license-clean result data so training/performance
numbers can be shared with project partners through the repo and synced
back from a GPU node the same way:

- runs/summary/grid.csv + label_efficiency.png (pure plot)
- every current-manifest run's final_metrics.json, resolved config.yaml,
  runtime_provenance.json, and per-epoch metrics.csv
- p36_summary.json (the P3.6 gate record)

Deliberately EXCLUDED: checkpoints, chip/prediction galleries (they contain
xView3 imagery — add manually if the team decides to), anything large.
Re-run after each wave; idempotent.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def current_exp_ids() -> set[str]:
    """Return only active core/reference ids; retired runs stay archived."""

    import yaml

    config = yaml.safe_load(Path("configs/arms.yaml").read_text())
    exp_ids = set()
    rerun_fracs = set(config["seeds"]["rerun_fracs"])
    for meta in config["arms"].values():
        for frac in config["label_fracs"]:
            seeds = list(config["seeds"]["core"])
            if frac in rerun_fracs:
                seeds += list(config["seeds"]["reruns"])
            for seed in seeds:
                exp_ids.add(f"{meta['short']}-f{int(round(frac * 100))}-s{seed}")
    exp_ids.update(ref["exp_id"] for ref in config["references"].values())
    return exp_ids

RUNS = Path("runs")
OUT = Path("results")


def main() -> int:
    copied = 0
    summary_out = OUT / "summary"
    summary_out.mkdir(parents=True, exist_ok=True)
    for name in ("grid.csv", "label_efficiency.png"):
        src = RUNS / "summary" / name
        if src.exists():
            shutil.copy2(src, summary_out / name)
            copied += 1
    if (RUNS / "p36_summary.json").exists():
        shutil.copy2(RUNS / "p36_summary.json", OUT / "p36_summary.json")
        copied += 1

    allowed = current_exp_ids()
    for final in sorted(RUNS.glob("*/final_metrics.json")):
        if final.parent.name not in allowed:
            continue
        run_dir = OUT / final.parent.name
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, run_dir / "final_metrics.json")
        copied += 1
        config_yaml = final.parent / "config.yaml"
        if config_yaml.exists():
            shutil.copy2(config_yaml, run_dir / "config.yaml")
            copied += 1
        runtime_provenance = final.parent / "runtime_provenance.json"
        if runtime_provenance.exists():
            shutil.copy2(runtime_provenance, run_dir / "runtime_provenance.json")
            copied += 1
        metrics_csv = final.parent / "metrics" / "metrics.csv"
        if metrics_csv.exists():
            shutil.copy2(metrics_csv, run_dir / "metrics.csv")
            copied += 1

    print(f"exported {copied} files -> {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
