#!/usr/bin/env bash
# Project installer: uv environment, optional model libraries, folders and checks.
# Works on Linux, macOS and WSL. Run from anywhere:  bash scripts/install.sh --help
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/install.sh [options]

  --lgbm          also install LightGBM        (extra "lgbm")
  --catboost      also install CatBoost        (extra "catboost")
  --all-models    install both optional model libraries
  --install-uv    install uv first if it is missing
  --skip-checks   do not run "make check" (ruff and pytest)
  --pipeline      run "make pipeline" afterwards (needs the raw CSV in data/raw)
  --dry-run       print the commands without running them
  -h, --help      show this help
EOF
}

EXTRAS=()
WANT_LGBM=0
WANT_CATBOOST=0
INSTALL_UV=0
RUN_CHECKS=1
RUN_PIPELINE=0
DRY_RUN=0
PYTHON_VERSION="3.11"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lgbm) EXTRAS+=(--extra lgbm); WANT_LGBM=1 ;;
    --catboost) EXTRAS+=(--extra catboost); WANT_CATBOOST=1 ;;
    --all-models) EXTRAS+=(--extra lgbm --extra catboost); WANT_LGBM=1; WANT_CATBOOST=1 ;;
    --install-uv) INSTALL_UV=1 ;;
    --skip-checks) RUN_CHECKS=0 ;;
    --pipeline) RUN_PIPELINE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."

step() { printf '\n==> %s\n' "$*"; }
fail() { printf 'error: %s\n' "$*" >&2; exit 1; }
run() {
  printf '+ %s\n' "$*"
  if [[ $DRY_RUN -eq 0 ]]; then "$@"; fi
}

[[ -f pyproject.toml ]] || fail "pyproject.toml not found; run this from the repository"

step "uv"
if ! command -v uv >/dev/null 2>&1; then
  if [[ $INSTALL_UV -eq 1 ]]; then
    run sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    export PATH="$HOME/.local/bin:$PATH"
  else
    fail "uv not found. Install it (https://docs.astral.sh/uv/) or rerun with --install-uv"
  fi
fi

step "Python ${PYTHON_VERSION} and dependencies"
run uv python install "$PYTHON_VERSION"
run uv sync ${EXTRAS[@]+"${EXTRAS[@]}"}

step "Folders"
for dir in data/raw data/silver data/features data/splits artifacts logs; do
  run mkdir -p "$dir"
done

step "Raw data"
if compgen -G "data/raw/*.csv" >/dev/null; then
  echo "found a CSV in data/raw"
else
  echo "warning: no CSV in data/raw. Download the AI4I 2020 dataset (UCI) into data/raw"
  echo "         before running 'make pipeline'."
  [[ $RUN_PIPELINE -eq 0 ]] || fail "--pipeline needs the raw CSV in data/raw"
fi

if [[ $WANT_LGBM -eq 1 ]]; then
  step "Check LightGBM"
  run uv run python -c "import lightgbm; print('lightgbm', lightgbm.__version__)"
fi
if [[ $WANT_CATBOOST -eq 1 ]]; then
  step "Check CatBoost"
  run uv run python -c "import catboost; print('catboost', catboost.__version__)"
fi

if [[ $RUN_CHECKS -eq 1 ]]; then
  step "Checks (ruff and pytest)"
  run make check
fi

if [[ $RUN_PIPELINE -eq 1 ]]; then
  step "Pipeline"
  run make pipeline
fi

step "Done"
cat <<'EOF'
Next steps:
  make pipeline                         # ingest, eda, silver, split, train
  make train EXP=configs/experiment/xgb_baseline.yaml
  make sweep SWEEP=configs/sweep/xgb_grid.yaml
EOF
