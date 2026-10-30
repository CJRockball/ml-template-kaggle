#!/usr/bin/env python3
"""Apply the LightGBM and CatBoost edits to existing project files.

    python patches/apply_model_patch.py            # check only (default)
    python patches/apply_model_patch.py --apply    # write the edits (backups are kept)
    python patches/apply_model_patch.py --revert   # restore the backups

Each edit replaces an exact block of text, so it does not depend on line numbers. All edits are
checked before anything is written: if one block is missing or has been changed locally, the
script stops and leaves every file untouched. Running it again after --apply reports
"already applied". Backups are saved next to each file as <name>.orig-modelpatch.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

BACKUP_SUFFIX = ".orig-modelpatch"


@dataclass(frozen=True)
class Edit:
    path: str
    old: str
    new: str
    marker: str  # text that is present once the edit has been applied


RUN_CONFIG_OLD_MODEL = """class XGBModelConfig(_Strict):
    type: Literal["xgb"] = "xgb"
    params: XGBParams = Field(default_factory=XGBParams)
"""

RUN_CONFIG_NEW_MODEL = (
    RUN_CONFIG_OLD_MODEL
    + """

class LGBMParams(_Strict):
    objective: Literal["binary"] = "binary"
    eval_metric: Literal["average_precision", "binary_logloss", "auc"] = "average_precision"
    n_estimators: int = Field(2000, ge=1)  # a cap; early stopping picks the real count
    early_stopping_rounds: int = Field(50, ge=1)
    num_leaves: int = Field(15, ge=2)
    max_depth: int = Field(-1, ge=-1)  # -1 = no limit
    learning_rate: float = Field(0.05, gt=0)
    subsample: float = Field(0.8, gt=0, le=1)
    subsample_freq: int = Field(1, ge=0)  # bagging happens only when this is above 0
    colsample_bytree: float = Field(0.8, gt=0, le=1)
    min_child_samples: int = Field(20, ge=1)
    reg_lambda: float = Field(1.0, ge=0)
    reg_alpha: float = Field(0.0, ge=0)
    max_bin: int = Field(255, ge=2)
    # "auto" = negatives / positives of the rows being fitted.
    scale_pos_weight: Annotated[float, Field(gt=0)] | Literal["auto"] = "auto"


class LGBMModelConfig(_Strict):
    type: Literal["lgbm"] = "lgbm"
    params: LGBMParams = Field(default_factory=LGBMParams)


class CatBoostParams(_Strict):
    loss_function: Literal["Logloss"] = "Logloss"
    eval_metric: Literal["PRAUC", "Logloss", "AUC"] = "PRAUC"  # watched by early stopping
    n_estimators: int = Field(2000, ge=1)  # a cap; early stopping picks the real count
    early_stopping_rounds: int = Field(50, ge=1)
    depth: int = Field(4, ge=1, le=16)
    learning_rate: float = Field(0.05, gt=0)
    subsample: float = Field(0.8, gt=0, le=1)  # Bernoulli row sampling
    rsm: float = Field(0.8, gt=0, le=1)  # share of features per split
    l2_leaf_reg: float = Field(3.0, ge=0)
    border_count: int = Field(254, ge=1, le=65535)
    scale_pos_weight: Annotated[float, Field(gt=0)] | Literal["auto"] = "auto"


class CatBoostModelConfig(_Strict):
    type: Literal["catboost"] = "catboost"
    params: CatBoostParams = Field(default_factory=CatBoostParams)


ModelConfig = Annotated[
    XGBModelConfig | LGBMModelConfig | CatBoostModelConfig, Field(discriminator="type")
]
"""
)

RUN_CONFIG_OLD_FIELD = """    # Add further model types here later as a discriminated union on ``type``.
    model: XGBModelConfig = Field(default_factory=XGBModelConfig)
"""
RUN_CONFIG_NEW_FIELD = """    # Add further model types to ``ModelConfig`` (a union keyed on ``type``).
    model: ModelConfig = Field(default_factory=XGBModelConfig)
"""

FACTORY_OLD = """def build_adapter_factory(cfg: RunConfig) -> AdapterFactory:
    \"\"\"Return a callable that makes a fresh, unfitted adapter for this config.\"\"\"
    if cfg.model.type == "xgb":
        from ml_template.models.xgb_classifier import XGBAdapter

        def base() -> ModelAdapter:
            return XGBAdapter(cfg.model.params, cfg.seed)

    else:
        raise ValueError(f"unknown model type {cfg.model.type!r}")

    if not cfg.fitted_features:
        return base
    from ml_template.features.fitted_features import TransformingAdapter

    return lambda: TransformingAdapter(base(), cfg.fitted_features)


def load_adapter(cfg: RunConfig, path: Path) -> ModelAdapter:
    if cfg.model.type == "xgb":
        from ml_template.models.xgb_classifier import XGBAdapter

        inner = XGBAdapter.load(cfg.model.params, cfg.seed, path)
    else:
        raise ValueError(f"unknown model type {cfg.model.type!r}")

    if not cfg.fitted_features:
        return inner
    from ml_template.features.fitted_features import TransformingAdapter

    return TransformingAdapter.load(inner, cfg.fitted_features, path)
"""

FACTORY_NEW = """def _adapter_class(model_type: str) -> type:
    \"\"\"The adapter class for a model type. Imports only the library that is needed.\"\"\"
    if model_type == "xgb":
        from ml_template.models.xgb_classifier import XGBAdapter

        return XGBAdapter
    if model_type == "lgbm":
        from ml_template.models.lgbm_classifier import LGBMAdapter

        return LGBMAdapter
    if model_type == "catboost":
        from ml_template.models.catboost_classifier import CatBoostAdapter

        return CatBoostAdapter
    raise ValueError(f"unknown model type {model_type!r}")


def build_adapter_factory(cfg: RunConfig) -> AdapterFactory:
    \"\"\"Return a callable that makes a fresh, unfitted adapter for this config.\"\"\"
    adapter_class = _adapter_class(cfg.model.type)

    def base() -> ModelAdapter:
        return adapter_class(cfg.model.params, cfg.seed)

    if not cfg.fitted_features:
        return base
    from ml_template.features.fitted_features import TransformingAdapter

    return lambda: TransformingAdapter(base(), cfg.fitted_features)


def load_adapter(cfg: RunConfig, path: Path) -> ModelAdapter:
    inner = _adapter_class(cfg.model.type).load(cfg.model.params, cfg.seed, path)
    if not cfg.fitted_features:
        return inner
    from ml_template.features.fitted_features import TransformingAdapter

    return TransformingAdapter.load(inner, cfg.fitted_features, path)
"""

PYPROJECT_OLD = "# Dependency groups = DEVELOPER-only tooling, never published (PEP 735).\n"
PYPROJECT_NEW = (
    "# Optional model libraries. Install with: uv sync --extra lgbm --extra catboost\n"
    "[project.optional-dependencies]\n"
    'lgbm = ["lightgbm>=4.3"]\n'
    'catboost = ["catboost>=1.2"]\n'
    "\n" + PYPROJECT_OLD
)

EDITS = [
    Edit(
        "src/ml_template/training/run_config.py",
        RUN_CONFIG_OLD_MODEL,
        RUN_CONFIG_NEW_MODEL,
        "class LGBMModelConfig(",
    ),
    Edit(
        "src/ml_template/training/run_config.py",
        RUN_CONFIG_OLD_FIELD,
        RUN_CONFIG_NEW_FIELD,
        "model: ModelConfig = Field(",
    ),
    Edit(
        "src/ml_template/models/model_factory.py", FACTORY_OLD, FACTORY_NEW, "def _adapter_class("
    ),
    Edit("pyproject.toml", PYPROJECT_OLD, PYPROJECT_NEW, "[project.optional-dependencies]"),
]


def plan(root: Path) -> tuple[dict[Path, str], list[str], list[str]]:
    """Return new file contents, messages and problems. Reads only."""
    texts: dict[Path, str] = {}
    messages: list[str] = []
    problems: list[str] = []
    for edit in EDITS:
        path = root / edit.path
        if not path.is_file():
            problems.append(f"{edit.path}: file not found")
            continue
        text = texts.get(path, path.read_text(encoding="utf-8"))
        if edit.marker in text:
            messages.append(f"{edit.path}: already applied ({edit.marker!r})")
        elif text.count(edit.old) != 1:
            found = text.count(edit.old)
            problems.append(
                f"{edit.path}: expected block found {found} times (needs exactly 1); "
                "the file differs from the version this patch was written for"
            )
        else:
            texts[path] = text.replace(edit.old, edit.new)
            messages.append(f"{edit.path}: will apply {edit.marker!r}")
            continue
        texts.setdefault(path, text)
    changed = {p: t for p, t in texts.items() if t != p.read_text(encoding="utf-8")}
    return changed, messages, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write the edits")
    mode.add_argument("--revert", action="store_true", help="restore the backups")
    args = parser.parse_args(argv)
    root = args.root.resolve()

    if args.revert:
        backups = [root / e.path for e in EDITS]
        restored = 0
        for path in dict.fromkeys(backups):
            backup = path.with_name(path.name + BACKUP_SUFFIX)
            if backup.is_file():
                path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
                backup.unlink()
                restored += 1
                print(f"restored {path.relative_to(root)}")
        print(f"{restored} file(s) restored")
        return 0

    changed, messages, problems = plan(root)
    for line in messages:
        print(line)
    if problems:
        for line in problems:
            print(f"PROBLEM  {line}", file=sys.stderr)
        print("nothing was changed", file=sys.stderr)
        return 1
    if not changed:
        print("nothing to apply: every edit is already in place")
        return 0
    if not args.apply:
        print(f"check passed: {len(changed)} file(s) would change. Re-run with --apply.")
        return 0
    for path, text in changed.items():
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(root)}")
    print(f"done: {len(changed)} file(s) changed. Next: uv lock, then make check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
