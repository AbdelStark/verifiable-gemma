#!/usr/bin/env bash
# Provision a rented vast.ai GPU instance for verifiable-gemma (backup to Modal).
# Run on the instance (CUDA 12 PyTorch image, 1 GPU, >= 200 GB disk):
#   curl -fsSL https://raw.githubusercontent.com/<you>/verifiable-gemma/main/infra/vast/setup.sh | bash -s -- <repo-url>
# or copy the repository over with rsync and run: bash infra/vast/setup.sh
set -euo pipefail

REPO_URL="${1:-}"
WORKDIR="${VG_WORKDIR:-/workspace}"
export HF_HOME="${HF_HOME:-$WORKDIR/hf}"

mkdir -p "$WORKDIR" "$HF_HOME"
cd "$WORKDIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ -n "$REPO_URL" ] && [ ! -d verifiable-gemma ]; then
  git clone "$REPO_URL" verifiable-gemma
fi
cd "${VG_REPO_DIR:-$WORKDIR/verifiable-gemma}"

uv sync --locked --extra gpu
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.cuda.get_device_name(0))"

cat <<MSG

verifiable-gemma is installed. HF_HOME=$HF_HOME
  uv run pytest -q
  uv run vg demo --tiny
  uv run vg keygen --model google/gemma-4-12B-it --out $WORKDIR/keys/gemma-4-12B-it --public-out $WORKDIR/state/gemma-4-12B-it/public.json
  uv run vg demo --model google/gemma-4-12B-it --device cuda --workdir $WORKDIR/state/demo
  uv run vg bench --model google/gemma-4-12B-it --device cuda --workdir $WORKDIR/state/bench --calibrate
MSG
