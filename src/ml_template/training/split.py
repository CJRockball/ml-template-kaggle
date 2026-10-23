"""Create, store and load the holdout and CV split.

The split is created once per (Silver version, split policy, key and target
columns). ``split_id`` is a hash of those, so the training step recomputes the
same ID from its config. Assignments are stored as Parquet plus a manifest, and
registered in the ``splits`` table of project.db.

Partitions: ``dev`` rows carry a CV fold 0..n_folds-1. ``holdout`` rows carry
fold -1 and are used for the final evaluation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml_template.data.contract import load_dataset_contract
from ml_template.paths import DATA_SILVER, DATA_SPLITS
from ml_template.training.run_config import SplitConfig, load_config, resolve_path

logger = logging.getLogger("ml_template.training.split")

HOLDOUT_FOLD = -1
MIN_POSITIVES = 5  # per fold and for the holdout


def split_policy(cfg: SplitConfig) -> dict[str, Any]:
    """Exclude an unset cv_seed to preserve existing split IDs."""
    policy = cfg.model_dump(mode="json")

    if policy.get("cv_seed") is None:
        policy.pop("cv_seed", None)

    return policy


def compute_split_id(
    silver_dataset_version_id: str,
    cfg: SplitConfig,
    key_col: str,
    target_col: str,
) -> str:
    payload = json.dumps(
        {
            "silver": silver_dataset_version_id,
            "policy": split_policy(cfg),
            "key": key_col,
            "target": target_col,
        },
        sort_keys=True,
    )
    return "split_" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def make_split(df: pd.DataFrame, cfg: SplitConfig, key_col: str, target_col: str) -> pd.DataFrame:
    """Return assignments with columns [key_col, partition, fold], sorted by key."""
    for col in (key_col, target_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in data")
    data = df[[key_col, target_col]].sort_values(key_col).reset_index(drop=True)
    if data[key_col].duplicated().any():
        raise ValueError(f"{key_col} is not unique")
    if data[target_col].isna().any() or not data[target_col].isin([0, 1]).all():
        raise ValueError(f"{target_col} must be binary 0/1 with no nulls")

    y = data[target_col].to_numpy().astype(int)
    idx = np.arange(len(data))
    dev_idx, _ = train_test_split(
        idx, test_size=cfg.holdout_fraction, stratify=y, random_state=cfg.seed
    )
    dev_idx = np.sort(dev_idx)

    fold = np.full(len(data), HOLDOUT_FOLD, dtype=int)
    cv_seed = cfg.seed if cfg.cv_seed is None else cfg.cv_seed

    skf = StratifiedKFold(
        n_splits=cfg.n_folds,
        shuffle=True,
        random_state=cv_seed,
    )
    for k, (_, val) in enumerate(skf.split(dev_idx, y[dev_idx])):
        fold[dev_idx[val]] = k

    for k in [HOLDOUT_FOLD, *range(cfg.n_folds)]:
        positives = int(y[fold == k].sum())
        if positives < MIN_POSITIVES:
            name = "holdout" if k == HOLDOUT_FOLD else f"fold {k}"
            raise ValueError(
                f"{name} has {positives} positives (minimum {MIN_POSITIVES}); "
                "reduce n_folds or holdout_fraction"
            )

    return pd.DataFrame(
        {
            key_col: data[key_col].to_numpy(),
            "partition": np.where(fold == HOLDOUT_FOLD, "holdout", "dev"),
            "fold": fold,
        }
    )


def _assignments_sha256(assignments: pd.DataFrame) -> str:
    return hashlib.sha256(assignments.to_csv(index=False).encode()).hexdigest()


def _manifest(
    assignments: pd.DataFrame,
    split_id: str,
    silver_id: str,
    cfg: SplitConfig,
    key_col: str,
    target_col: str,
    target: pd.Series | None,
) -> dict[str, Any]:
    hold = (assignments["partition"] == "holdout").to_numpy()
    manifest: dict[str, Any] = {
        "split_id": split_id,
        "silver_dataset_version_id": silver_id,
        "policy": split_policy(cfg),
        "key_col": key_col,
        "target_col": target_col,
        "n_rows": int(len(assignments)),
        "n_holdout": int(hold.sum()),
        "fold_sizes": {
            str(k): int(v)
            for k, v in assignments.loc[~hold, "fold"].value_counts().sort_index().items()
        },
        "assignments_sha256": _assignments_sha256(assignments),
        "created_at": datetime.now(UTC).isoformat(),
    }
    if target is not None:
        y = target.to_numpy().astype(int)
        manifest["n_positive_total"] = int(y.sum())
        manifest["n_positive_holdout"] = int(y[hold].sum())
    return manifest


def save_split(
    assignments: pd.DataFrame,
    silver_id: str,
    cfg: SplitConfig,
    key_col: str,
    target_col: str,
    target: pd.Series | None = None,
    root: Path = DATA_SPLITS,
) -> tuple[str, Path, dict[str, Any]]:
    """Write the split. Immutable: an existing split must be identical."""
    split_id = compute_split_id(silver_id, cfg, key_col, target_col)
    out_dir = Path(root) / split_id
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["assignments_sha256"] != _assignments_sha256(assignments):
            raise FileExistsError(f"{split_id} exists with different assignments")
        logger.info("Split %s already exists and matches; reusing", split_id)
        return split_id, out_dir, existing

    out_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(out_dir / "assignments.parquet", index=False)
    manifest = _manifest(assignments, split_id, silver_id, cfg, key_col, target_col, target)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote split %s to %s", split_id, out_dir)
    return split_id, out_dir, manifest


def load_split(split_id: str, root: Path = DATA_SPLITS) -> pd.DataFrame:
    path = Path(root) / split_id / "assignments.parquet"
    if not path.exists():
        raise FileNotFoundError(f"split {split_id} not found at {path}; run the split step")
    return pd.read_parquet(path)


def holdout_keys(assignments: pd.DataFrame, key_col: str) -> np.ndarray:
    return assignments.loc[assignments["partition"] == "holdout", key_col].to_numpy()


def cv_folds(assignments: pd.DataFrame, key_col: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """List of (train_keys, validation_keys) per fold, dev rows only."""
    dev = assignments[assignments["partition"] == "dev"]
    folds = []
    for k in sorted(dev["fold"].unique()):
        val = dev.loc[dev["fold"] == k, key_col].to_numpy()
        train = dev.loc[dev["fold"] != k, key_col].to_numpy()
        folds.append((train, val))
    return folds


def register_split(engine: Any, split_id: str, path: Path, manifest: dict[str, Any]) -> None:
    """Insert a row in the ``splits`` table (created if missing). Idempotent."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    from ml_template.db.tracking_tables import ensure_tracking_tables, splits

    ensure_tracking_tables(engine)
    stmt = (
        sqlite_insert(splits)
        .values(
            split_id=split_id,
            silver_dataset_version_id=manifest["silver_dataset_version_id"],
            path=str(path),
            policy_json=json.dumps(manifest["policy"], sort_keys=True),
            key_col=manifest["key_col"],
            target_col=manifest["target_col"],
            n_rows=manifest["n_rows"],
            n_holdout=manifest["n_holdout"],
            assignments_sha256=manifest["assignments_sha256"],
            created_at=manifest["created_at"],
        )
        .on_conflict_do_nothing(index_elements=["split_id"])
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def main(argv: list[str] | None = None) -> None:
    from ml_template.db.connection import get_engine
    from ml_template.tracking.pipeline_logging import setup_logging

    p = argparse.ArgumentParser(description="Create the holdout and CV split for a Silver version.")
    p.add_argument("--config", default="configs/experiment/xgb_baseline.yaml", type=Path)
    p.add_argument("--silver-version-id", default=None, help="Overrides the config value")
    p.add_argument("--silver-root", default=DATA_SILVER, type=Path)
    p.add_argument("--split-root", default=DATA_SPLITS, type=Path)
    args = p.parse_args(argv)

    setup_logging()
    cfg = load_config(args.config)
    contract = load_dataset_contract(resolve_path(cfg.data.dataset_config))
    key_col, target_col = contract.entity_id, contract.target
    silver_id = args.silver_version_id or cfg.data.silver_dataset_version_id
    parquet = args.silver_root / cfg.data.dataset_name / silver_id / "data.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"{parquet} not found; build Silver first")
    logger.info("Creating split for %s from %s", silver_id, parquet)

    df = pd.read_parquet(parquet, columns=[key_col, target_col])
    assignments = make_split(df, cfg.split, key_col, target_col)

    if cfg.split.cv_seed is not None:
        original_policy = cfg.split.model_copy(update={"cv_seed": None})
        original_id = compute_split_id(
            silver_id,
            original_policy,
            key_col,
            target_col,
        )
        original = load_split(original_id, args.split_root)

        original_keys = set(original[key_col])
        new_keys = set(assignments[key_col])
        if original_keys != new_keys:
            raise ValueError("New assignments do not cover the same rows as the original split")

        original_holdout = set(holdout_keys(original, key_col))
        new_holdout = set(holdout_keys(assignments, key_col))
        if original_holdout != new_holdout:
            raise ValueError("Holdout membership changed; refusing to save the new CV split")

        logger.info(
            "Verified unchanged holdout against original split %s",
            original_id,
        )

    target = df.sort_values(key_col)[target_col].reset_index(drop=True)
    split_id, out_dir, manifest = save_split(
        assignments, silver_id, cfg.split, key_col, target_col, target, args.split_root
    )
    register_split(get_engine(), split_id, out_dir, manifest)
    logger.info(
        "Split %s: %d rows, %d holdout, fold sizes %s",
        split_id,
        manifest["n_rows"],
        manifest["n_holdout"],
        manifest["fold_sizes"],
    )


if __name__ == "__main__":
    main()
