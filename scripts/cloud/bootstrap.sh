#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${REPO_ROOT}/configs/cloud/taboo_ship_smoke.toml"
UV_VERSION="0.11.10"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' \
    "check Linux and NVIDIA RTX A6000" \
    "install uv ${UV_VERSION} if absent" \
    "uv sync --locked --extra cloud" \
    "model-audits preflight --check-access" \
    "stop before any model-weight download or paid smoke execution"
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: cloud bootstrap requires Linux" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi is unavailable" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --disable-pip-version-check "uv==${UV_VERSION}"
fi

cd "${REPO_ROOT}"
uv sync --locked --extra cloud
uv run --extra cloud model-audits preflight --config "${CONFIG}" --check-access

echo "Bootstrap complete. No model weights were downloaded by this script."
echo "Do not run the paid smoke command without the fresh Milestone 1B approval."
