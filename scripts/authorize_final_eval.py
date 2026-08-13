"""Publish the owner's hash-bound all-32 final-evaluation amendment.

This command validates only frozen campaign/TEST/control evidence. It never
opens ``validation.csv`` or any eval-final raster. The final evaluator consumes
the resulting canonical receipt only after independently rebuilding it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from scripts.h100.contracts import load_cells
from src.analysis.curves import training_fraction_counts
from src.eval.final_authorization import (
    AUTHORIZATION_FILENAME,
    AUTHORIZED_OWNER,
    write_authorization,
)
from src.eval.final_eval import (
    _campaign_git_sha_from_cohort,
    _require_clean_evaluator,
    _validated_final_checkpoint,
    owner_authorization_evidence,
    validate_grid_contract,
)
from src.eval.heldout_contract import (
    COHORT_FILENAME,
    HeldoutContractError,
    cohort_record,
    validate_complete_test_cohort,
    validate_training_cohort,
)


CONFIRMATION = "I_AUTHORIZE_POST_TEST_ALL_32_FINAL_ONCE"


def _canonical_directory(path: Path, description: str) -> Path:
    absolute = path.absolute()
    if (
        absolute.is_symlink()
        or not absolute.is_dir()
        or absolute.resolve() != absolute
    ):
        raise HeldoutContractError(
            f"{description} must be an existing canonical non-symlink directory"
        )
    return absolute


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--owner", default=AUTHORIZED_OWNER)
    parser.add_argument(
        "--confirm",
        required=True,
        help=f"must be the literal {CONFIRMATION}",
    )
    args = parser.parse_args(argv)
    if args.confirm != CONFIRMATION:
        parser.error(f"--confirm must be the literal {CONFIRMATION}")
    if args.owner != AUTHORIZED_OWNER:
        parser.error(f"--owner must be {AUTHORIZED_OWNER}")

    repo = _canonical_directory(args.repo, "repository")
    runs_root = _canonical_directory(args.runs_root, "runs root")
    lockfile = runs_root / "final_eval.lock"
    if lockfile.exists() or lockfile.is_symlink():
        raise HeldoutContractError(
            "final_eval.lock already exists; authorization cannot be created after access"
        )
    output = runs_root / ".h100" / AUTHORIZATION_FILENAME

    detector_path = repo / "configs/detector.yaml"
    detector_cfg = yaml.safe_load(detector_path.read_text(encoding="utf-8"))
    detector_sha256 = hashlib.sha256(detector_path.read_bytes()).hexdigest()
    cells = load_cells(repo)
    cohort_path = runs_root / ".h100" / COHORT_FILENAME
    campaign_git_sha = _campaign_git_sha_from_cohort(cohort_path)
    evaluator_git_sha = _require_clean_evaluator(
        repo, campaign_git_sha=campaign_git_sha
    )
    cohort, cohort_sha256 = validate_training_cohort(
        path=cohort_path,
        cells=cells,
        runs_root=runs_root,
        git_sha=campaign_git_sha,
        detector_sha256=detector_sha256,
        candidate_floor=float(detector_cfg["decode"]["candidate_floor"]),
    )

    data_cfg = yaml.safe_load((repo / "configs/data.yaml").read_text(encoding="utf-8"))
    splits = json.loads(
        (repo / str(data_cfg["paths"]["splits"])).read_text(encoding="utf-8")
    )["splits"]
    test_scene_ids = tuple(sorted(map(str, splits["test"])))
    test_results = validate_complete_test_cohort(
        cells=cells,
        runs_root=runs_root,
        cohort=cohort,
        cohort_sha256=cohort_sha256,
        test_scene_ids=test_scene_ids,
    )
    arms = yaml.safe_load((repo / "configs/arms.yaml").read_text(encoding="utf-8"))[
        "arms"
    ]
    fraction_counts = training_fraction_counts(
        repo=repo,
        fractions=tuple(sorted({float(cell.fraction) for cell in cells})),
    )
    grid_audit = validate_grid_contract(
        runs_root / "summary" / "grid.csv",
        cells=cells,
        test_results=test_results,
        cohort=cohort,
        runs_root=runs_root,
        arms=arms,
        fraction_counts=fraction_counts,
        git_sha=campaign_git_sha,
        detector_sha256=detector_sha256,
        precision=str(detector_cfg["schedule"]["precision"]),
    )
    if grid_audit["monotonicity_ok"] is not False:
        raise HeldoutContractError(
            "post-TEST amendment is valid only for the disclosed failed diagnostic"
        )
    for cell in cells:
        _validated_final_checkpoint(
            record=cohort_record(cohort, cell.exp_id),
            runs_root=runs_root,
            exp_id=cell.exp_id,
        )

    payload = owner_authorization_evidence(
        owner=args.owner,
        created_utc=None,
        evaluator_git_sha=evaluator_git_sha,
        campaign_git_sha=campaign_git_sha,
        detector_sha256=detector_sha256,
        cohort_sha256=cohort_sha256,
        grid_audit=grid_audit,
        cells=cells,
        runs_root=runs_root,
    )
    digest = write_authorization(output, payload)
    print(
        json.dumps(
            {
                "status": "owner-amendment-published",
                "path": str(output),
                "sha256": digest,
                "campaign_git_sha": campaign_git_sha,
                "evaluator_git_sha": evaluator_git_sha,
                "selected_cell_count": len(cells),
                "violations": grid_audit["violations"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
