"""Resolve a run config into model inputs: features, target and stored split.

Shared by training and holdout evaluation so both see identical data. The model
sees only the key, the feature columns and the single target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ml_template.data.contract import DatasetContract, load_dataset_contract
from ml_template.data.silver_loader import SilverData, load_silver
from ml_template.features.feature_set_store import resolve_feature_set
from ml_template.models.model_factory import prepare_features
from ml_template.paths import DATA_FEATURES, DATA_SILVER, DATA_SPLITS
from ml_template.training.run_config import RunConfig, resolve_path
from ml_template.training.split import compute_split_id, load_split


@dataclass(frozen=True)
class RunInputs:
    contract: DatasetContract
    silver: SilverData
    X: pd.DataFrame  # feature columns only, indexed by entity key
    y: pd.Series  # binary target, same index
    assignments: pd.DataFrame  # key, partition, fold
    split_id: str


def split_id_for(cfg: RunConfig, contract: DatasetContract) -> str:
    return compute_split_id(
        cfg.data.silver_dataset_version_id, cfg.split, contract.entity_id, contract.target
    )


def load_run_inputs(
    cfg: RunConfig,
    *,
    silver_root: Path = DATA_SILVER,
    split_root: Path = DATA_SPLITS,
    feature_root: Path = DATA_FEATURES,
) -> RunInputs:
    contract = load_dataset_contract(resolve_path(cfg.data.dataset_config))
    silver = load_silver(
        cfg.data.silver_dataset_version_id, cfg.data.dataset_name, contract.entity_id, silver_root
    )
    if silver.target != contract.target or set(silver.features) != set(contract.features):
        raise ValueError(
            "Silver manifest roles and the dataset config disagree; "
            "rebuild Silver or restore the config used for it"
        )

    split_id = split_id_for(cfg, contract)
    assignments = load_split(split_id, split_root)

    frame = silver.model_frame()
    raw = frame[silver.features]
    if cfg.data.feature_set_id is not None:
        raw = resolve_feature_set(
            cfg.data.feature_set_id, raw, silver.silver_dataset_version_id, feature_root
        )
    X = prepare_features(raw)
    y = frame[silver.target].astype(int)
    if set(assignments[contract.entity_id]) != set(frame.index):
        raise ValueError(
            f"split {split_id} does not cover the rows of {silver.silver_dataset_version_id}"
        )
    return RunInputs(contract, silver, X, y, assignments, split_id)
